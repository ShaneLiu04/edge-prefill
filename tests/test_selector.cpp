#include "test_framework.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/embedding_scorer.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/semantic_gate.h"

using namespace edge_prefill;

namespace {

struct Fixture {
    std::vector<llama_token> tokens;
    std::vector<std::string> pieces;

    explicit Fixture(uint32_t n, uint32_t needle_block = 37) {
        tokens.resize(n);
        pieces.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            tokens[i] = static_cast<llama_token>(i % 1000);
            pieces[i] = "filler";
        }
        if (needle_block * 32 + 5 < n) {
            pieces[needle_block * 32 + 5] = "needle";
        }
    }
};

SelectionOptions options(double budget = 0.5) {
    SelectionOptions opts;
    opts.block_size = 32;
    opts.budget = budget;
    opts.min_dense_tokens = 512;
    opts.sink_blocks = 1;
    opts.recent_blocks = 2;
    opts.coverage_stride = 8;
    return opts;
}

} // namespace

TEST_CASE(selector_budget_and_mandatory) {
    Fixture f(8192);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, options(0.5), nullptr, &sel));
    CHECK(!sel.dense);
    const double target = 8192 * 0.5;
    CHECK(sel.selected_tokens >= target * 0.95);
    CHECK(sel.selected_tokens <= target * 1.05 + 1);
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 0) !=
          sel.mandatory_blocks.end());
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 37) !=
          sel.mandatory_blocks.end());
    CHECK(std::is_sorted(sel.token_indices.begin(), sel.token_indices.end()));
    CHECK_EQ(sel.token_indices.back(), 8191u);
}

TEST_CASE(selector_query_suffix_does_not_dominate_lexical) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[37 * 32 + 5] = "needle";
    for (uint32_t i = 0; i < 10; ++i) {
        pieces[(n / 32 - 1) * 32 + i] = "needle";
    }

    Selection sel;
    BlockSelector selector;
    auto opts = options(0.2);
    opts.coverage_stride = 256;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &sel));
    CHECK(!sel.dense);
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 37) !=
          sel.mandatory_blocks.end());
}

TEST_CASE(selector_deterministic) {
    Fixture f(4096);
    Selection a;
    Selection b;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, options(0.4), nullptr, &a));
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, options(0.4), nullptr, &b));
    CHECK(a.token_indices == b.token_indices);
}

TEST_CASE(selector_short_prompt_is_dense) {
    Fixture f(256);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, options(0.5), nullptr, &sel));
    CHECK(sel.dense);
    CHECK_EQ(sel.selected_tokens, 256u);
    CHECK_EQ(sel.reason, std::string("short-prompt"));
}

TEST_CASE(selector_budget_095_is_dense) {
    Fixture f(8192);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "", f.pieces, options(0.95), nullptr, &sel));
    CHECK(sel.dense);
}

TEST_CASE(selector_single_token_is_dense) {
    Fixture f(1, 0);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "", f.pieces, options(0.5), nullptr, &sel));
    CHECK(sel.dense);
    CHECK_EQ(sel.selected_tokens, 1u);
}

TEST_CASE(selector_invalid_inputs) {
    Fixture f(1024);
    Selection sel;
    BlockSelector selector;
    CHECK_ERR(selector.select({}, "", {}, options(0.5), nullptr, &sel),
              ErrorCode::InvalidArgument);
    CHECK_ERR(selector.select(f.tokens, "", f.pieces, options(0.0), nullptr, &sel),
              ErrorCode::InvalidArgument);
    CHECK_ERR(selector.select(f.tokens, "", f.pieces, options(1.1), nullptr, &sel),
              ErrorCode::InvalidArgument);
    auto bad = options(0.5);
    bad.block_size = 7;
    CHECK_ERR(selector.select(f.tokens, "", f.pieces, bad, nullptr, &sel),
              ErrorCode::InvalidArgument);
    bad = options(0.5);
    bad.min_dense_tokens = 16;
    CHECK_ERR(selector.select(f.tokens, "", f.pieces, bad, nullptr, &sel),
              ErrorCode::InvalidArgument);
    CHECK_ERR(selector.select(f.tokens, "", {}, options(0.5), nullptr, &sel),
              ErrorCode::InvalidArgument);
    CHECK_ERR(selector.select(f.tokens, "", f.pieces, options(0.5), nullptr, nullptr),
              ErrorCode::InvalidArgument);
}

TEST_CASE(selector_16k_budget) {
    Fixture f(16384);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, options(0.5), nullptr, &sel));
    const double target = 16384 * 0.5;
    CHECK(sel.selected_tokens >= target * 0.95);
    CHECK(sel.selected_tokens <= target * 1.05 + 1);
}

TEST_CASE(broadcast_block_scores_geometry) {
    std::vector<double> per_token;

    // 100 tokens / 32 = 3 full blocks + a 4-token trailing half block that
    // still consumes a whole block score (its own).
    CHECK_OK(broadcast_block_scores({1.0, 2.0, 3.0, 4.0}, 32, 100, &per_token));
    CHECK_EQ(per_token.size(), static_cast<size_t>(100));
    for (uint32_t i = 0; i < 100; ++i) {
        const double expected = static_cast<double>(i / 32 + 1);
        CHECK_NEAR(per_token[i], expected, 1e-12);
    }
    // Broadcast property: every token in a block shares the block score
    // exactly (bit-identical copy, not a recomputed value).
    for (uint32_t b = 0; b < 4; ++b) {
        for (uint32_t i = b * 32 + 1; i < std::min((b + 1) * 32, 100u); ++i) {
            CHECK_EQ(per_token[i], per_token[b * 32]);
        }
    }

    // Exact coverage (96 == 3 * 32) is legal.
    per_token.clear();
    CHECK_OK(broadcast_block_scores({5.0, 6.0, 7.0}, 32, 96, &per_token));
    CHECK_EQ(per_token.size(), static_cast<size_t>(96));
    CHECK_NEAR(per_token[95], 7.0, 1e-12);

    // Violations: empty scores, zero block size, zero tokens, a token count
    // the scores cannot cover (2 blocks < 200 tokens), a score count that
    // does not match the blocking (4 blocks for 96 tokens), null output.
    CHECK_ERR(broadcast_block_scores({}, 32, 96, &per_token),
              ErrorCode::InvalidArgument);
    CHECK_ERR(broadcast_block_scores({1.0, 2.0, 3.0}, 0, 96, &per_token),
              ErrorCode::InvalidArgument);
    CHECK_ERR(broadcast_block_scores({1.0, 2.0, 3.0}, 32, 0, &per_token),
              ErrorCode::InvalidArgument);
    CHECK_ERR(broadcast_block_scores({1.0, 2.0}, 32, 200, &per_token),
              ErrorCode::InvalidArgument);
    CHECK_ERR(broadcast_block_scores({1.0, 2.0, 3.0, 4.0}, 32, 96, &per_token),
              ErrorCode::InvalidArgument);
    CHECK_ERR(broadcast_block_scores({1.0, 2.0, 3.0}, 32, 96, nullptr),
              ErrorCode::InvalidArgument);
}

TEST_CASE(embedding_selection_scorer_validation) {
    EmbeddingScorer unloaded;
    std::vector<std::string> pieces = {"alpha", "beta", "gamma"};
    std::vector<llama_token> tokens = {1, 2, 3};
    std::vector<double> out;

    // The constructor only stores pointers (it has no Status return), so
    // every failure mode must surface from score() as InvalidArgument.
    EmbeddingSelectionScorer null_scorer(nullptr, &pieces, 32);
    CHECK_ERR(null_scorer.score(tokens, "query", &out), ErrorCode::InvalidArgument);

    EmbeddingSelectionScorer not_loaded(&unloaded, &pieces, 32);
    CHECK_ERR(not_loaded.score(tokens, "query", &out), ErrorCode::InvalidArgument);

    EmbeddingSelectionScorer null_pieces(&unloaded, nullptr, 32);
    CHECK_ERR(null_pieces.score(tokens, "query", &out), ErrorCode::InvalidArgument);

    EmbeddingSelectionScorer zero_block(&unloaded, &pieces, 0);
    CHECK_ERR(zero_block.score(tokens, "query", &out), ErrorCode::InvalidArgument);

    EmbeddingSelectionScorer adapter(&unloaded, &pieces, 32);
    CHECK_ERR(adapter.score(tokens, "", &out), ErrorCode::InvalidArgument);
    CHECK_ERR(adapter.score({}, "query", &out), ErrorCode::InvalidArgument);
    CHECK_ERR(adapter.score(tokens, "query", nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(semantic_priority_weights_default) {
    const SelectorWeights w = semantic_priority_weights();
    CHECK_NEAR(w.lexical, 0.20, 1e-9);
    CHECK_NEAR(w.diversity, 0.15, 1e-9);
    CHECK_NEAR(w.position, 0.15, 1e-9);
    CHECK_NEAR(w.draft, 0.50, 1e-9);
}

TEST_CASE(semantic_selection_needle_guard) {
    const char* env = std::getenv("EDGE_PREFILL_TEST_EMBEDDING");
    const std::string path = env ? env : "";
    if (path.empty()) {
        std::cout << "SKIP semantic_selection_needle_guard "
                     "(EDGE_PREFILL_TEST_EMBEDDING unset)\n";
        return;
    }

    LlamaBackend::init();
    EmbeddingScorer scorer;
    CHECK_OK(scorer.load_model(path));

    constexpr uint32_t n_tokens = 32768;
    constexpr uint32_t block_size = 32;
    constexpr uint32_t n_blocks = n_tokens / block_size;
    constexpr uint32_t needle_block = 600;
    const std::string query = "What is the magic passcode stored in the archive?";
    const std::string needle_text = "The magic passcode stored in the archive is 4271.";
    const char* filler_words[] = {"the", "quiet", "river", "flows", "past",
                                  "the", "old",   "stone", "bridge"};
    constexpr uint32_t n_filler = 9;

    std::vector<llama_token> tokens(n_tokens);
    std::vector<std::string> pieces(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        tokens[i] = static_cast<llama_token>(i % 1000);
        pieces[i] = filler_words[i % n_filler];
    }
    for (uint32_t i = needle_block * block_size;
         i < needle_block * block_size + block_size; ++i) {
        pieces[i] = needle_text;
    }

    // Adapter success contract on the real engine: one score per token,
    // block-constant values (broadcast), and the needle block semantically
    // outranking every filler block for the passcode query.
    EmbeddingSelectionScorer adapter(&scorer, &pieces, block_size);
    std::vector<double> per_token;
    CHECK_OK(adapter.score(tokens, query, &per_token));
    CHECK_EQ(per_token.size(), static_cast<size_t>(n_tokens));
    for (uint32_t b = 0; b < n_blocks; ++b) {
        for (uint32_t i = b * block_size + 1; i < (b + 1) * block_size; ++i) {
            CHECK_EQ(per_token[i], per_token[b * block_size]);
        }
    }
    const double needle_score = per_token[needle_block * block_size];
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (b == needle_block) continue;
        CHECK(per_token[b * block_size] < needle_score);
    }

    // End to end: the semantic signal must carry the needle block into the
    // selection under a tight budget, and the selection must record that
    // its per-position scores came from the embedding engine.
    BlockSelector selector;
    Selection sel;
    CHECK_OK(selector.select(tokens, query, pieces, options(0.15), &adapter, &sel));
    CHECK(!sel.dense);
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(),
                    needle_block) != sel.block_indices.end());
    CHECK_EQ(sel.score_source, std::string("embedding"));

    LlamaBackend::shutdown();
}
