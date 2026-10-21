#include "test_framework.h"

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/config.h"
#include "edge_prefill/draft_scorer.h"
#include "edge_prefill/selector.h"

using namespace edge_prefill;

namespace {

std::string test_model_path() {
    const char* env = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return env ? env : "";
}

// Stub that returns a fixed per-token score: blocks whose tokens contain
// "signal" get 1.0, everything else gets 0.0.
struct SignalStubScorer : public DraftScorer {
    Status score(const std::vector<llama_token>& tokens,
                 const std::string& query,
                 std::vector<double>* per_token_scores) override {
        (void)query;
        per_token_scores->assign(tokens.size(), 0.0);
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i] == 1) (*per_token_scores)[i] = 1.0;
        }
        return Status::success();
    }
};

struct FailingScorer : public DraftScorer {
    Status score(const std::vector<llama_token>& tokens,
                 const std::string& query,
                 std::vector<double>* per_token_scores) override {
        (void)tokens;
        (void)query;
        (void)per_token_scores;
        return Status::error(ErrorCode::Unsupported, "stub failure");
    }
};

struct SizeMismatchScorer : public DraftScorer {
    Status score(const std::vector<llama_token>& tokens,
                 const std::string& query,
                 std::vector<double>* per_token_scores) override {
        (void)tokens;
        (void)query;
        per_token_scores->assign(3, 0.5);
        return Status::success();
    }
};

SelectionOptions sel_options(double budget = 0.5) {
    SelectionOptions opts;
    opts.block_size = 32;
    opts.budget = budget;
    opts.min_dense_tokens = 512;
    return opts;
}

} // namespace

TEST_CASE(draft_scorer_missing_model_file) {
    LlamaDraftScorer scorer;
    CHECK_ERR(scorer.load_model("definitely/not/a/model.gguf"), ErrorCode::ModelLoad);
    CHECK_ERR(scorer.load_model(""), ErrorCode::InvalidArgument);
    CHECK(!scorer.loaded());
}

TEST_CASE(draft_scorer_empty_query_or_tokens) {
    LlamaDraftScorer scorer;
    std::vector<double> scores;
    CHECK_ERR(scorer.score({}, "query", &scores), ErrorCode::InvalidArgument);
    std::vector<llama_token> tokens = {1, 2, 3};
    CHECK_ERR(scorer.score(tokens, "", &scores), ErrorCode::InvalidArgument);
    // Model not loaded yet: valid input surfaces Unsupported.
    CHECK_ERR(scorer.score(tokens, "query", &scores), ErrorCode::Unsupported);
    CHECK_ERR(scorer.score(tokens, "query", nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(draft_scorer_load_and_score) {
    const std::string path = test_model_path();
    if (path.empty()) {
        std::cout << "SKIP draft_scorer_load_and_score (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend::init();
    LlamaDraftScorer scorer;
    DraftScorerOptions opts;
    opts.block_size = 32;
    CHECK_OK(scorer.load_model(path, opts));
    CHECK(scorer.loaded());
    CHECK(scorer.n_vocab() > 0);
    CHECK(scorer.n_ctx() > 0);

    // Reuse the draft model itself as its own tokenizer/vocab source: the
    // token ids are guaranteed to be covered.
    LlamaBackend backend;
    CHECK_OK(backend.load_model(path));
    std::vector<llama_token> tokens;
    CHECK_OK(backend.tokenize(
        "The quick brown fox jumps over the lazy dog. Rain falls quietly on the "
        "rooftops. Numbers 1 2 3 4 5 6 7 8 9 10 repeat until the end of this test. "
        "Meanwhile the old library keeper sorted dusty books about mountains, rivers, "
        "and distant cities before closing the wooden gate at sunset.",
        &tokens));
    CHECK(tokens.size() >= 64);

    std::vector<double> scores;
    CHECK_OK(scorer.score(tokens, "what color is the fox?", &scores));
    CHECK_EQ(scores.size(), tokens.size());
    for (double s : scores) {
        CHECK(std::isfinite(s));
        CHECK(s >= -1.0 - 1e-9);
        CHECK(s <= 1.0 + 1e-9);
    }
    // Scores are block-constant by construction.
    CHECK_NEAR(scores[0], scores[31], 1e-12);

    // Out-of-vocab token ids must surface Unsupported (caller falls back
    // to lexical-only selection).
    auto bad_tokens = tokens;
    bad_tokens[0] = scorer.n_vocab() + 5;
    CHECK_ERR(scorer.score(bad_tokens, "query", &scores), ErrorCode::Unsupported);

    LlamaBackend::shutdown();
}

TEST_CASE(block_selector_draft_fusion) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // Mark one mid-prompt block as the draft-signal block.
    constexpr uint32_t signal_block = 100;
    for (uint32_t i = signal_block * 32; i < signal_block * 32 + 32; ++i) {
        tokens[i] = 1;
        pieces[i] = "signal";
    }

    SignalStubScorer stub;
    BlockSelector selector;
    Selection sel;
    CHECK_OK(selector.select(tokens, "unrelated query terms", pieces,
                             sel_options(0.3), &stub, &sel));
    CHECK(!sel.dense);
    CHECK_EQ(sel.reason, std::string("selected"));
    // The draft-signal block has fused score boosted by w_draft and must
    // be among the selected blocks.
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), signal_block) !=
          sel.block_indices.end());
    CHECK_NEAR(sel.signals[signal_block].draft, 1.0, 1e-9);
    // Fused must be the weighted combination of the normalized signals.
    const auto& s = sel.signals[signal_block];
    SelectorWeights w = sel_options(0.3).weights;
    const double total = w.lexical + w.diversity + w.position + w.draft;
    CHECK_NEAR(s.fused,
               (w.lexical / total) * s.lexical + (w.diversity / total) * s.diversity +
                   (w.position / total) * s.position + (w.draft / total) * s.draft,
               1e-9);
    // Determinism with the same draft scores.
    Selection sel2;
    CHECK_OK(selector.select(tokens, "unrelated query terms", pieces,
                             sel_options(0.3), &stub, &sel2));
    CHECK(sel.token_indices == sel2.token_indices);
}

TEST_CASE(draft_scorer_missing_model_fallback) {
    constexpr uint32_t n = 4096;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);

    FailingScorer failing;
    BlockSelector selector;
    Selection with_stub;
    CHECK_OK(selector.select(tokens, "query", pieces, sel_options(0.5), &failing, &with_stub));
    CHECK(!with_stub.dense);
    CHECK_EQ(with_stub.reason, std::string("draft-disabled"));
    // No fabricated draft signal: every draft signal stays zero.
    for (const auto& s : with_stub.signals) {
        CHECK_NEAR(s.draft, 0.0, 1e-12);
    }

    // The fallback must equal pure lexical-only selection.
    Selection lexical_only;
    CHECK_OK(selector.select(tokens, "query", pieces, sel_options(0.5), nullptr, &lexical_only));
    CHECK(lexical_only.token_indices == with_stub.token_indices);

    SizeMismatchScorer mismatch;
    Selection with_mismatch;
    CHECK_OK(selector.select(tokens, "query", pieces, sel_options(0.5), &mismatch,
                             &with_mismatch));
    CHECK_EQ(with_mismatch.reason, std::string("draft-disabled"));
    CHECK(with_mismatch.token_indices == lexical_only.token_indices);
}
