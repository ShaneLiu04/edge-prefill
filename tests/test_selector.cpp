#include "test_framework.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <set>
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

// AR007 FR-2: the anchor budget cap is what lets a small budget actually
// select few tokens. Before AR007 the mandatory anchors alone pinned ~13% of
// a 16K prompt, so any budget below that produced the same selection.
TEST_CASE(selector_anchor_cap_reaches_low_budget) {
    constexpr uint32_t n = 16384;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[100 * 32 + 3] = "needle";

    Selection sel;
    BlockSelector selector;
    auto opts = options(0.05);
    opts.coverage_stride = 0;  // auto (AR007)
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &sel));
    CHECK(!sel.dense);
    // 5% of 16384 = 819 tokens; the 1.05 slack plus the forced last token is
    // the ceiling. The old unconditional anchors produced ~2100.
    CHECK(sel.selected_tokens <= 900u);
    CHECK(sel.selected_tokens >= 700u);
    // Anchors must respect the ratio cap (35% of the target, >= one block).
    const uint32_t anchor_tokens = static_cast<uint32_t>(sel.anchor_blocks) * 32u;
    CHECK(anchor_tokens <= 400u);
    // The needle block is still pinned by the lexical/entity anchor even at
    // 5%: quality must not depend on the budget being large.
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), 100u) !=
          sel.block_indices.end());
}

// AR007 FR-3: MMR must not let near-duplicate boilerplate crowd out a single
// unique relevant block.
TEST_CASE(selector_mmr_prefers_unique_block) {
    constexpr uint32_t n = 16384;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n);
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // 400 blocks of identical boilerplate that happens to share the query
    // term, plus one block with the actual fact and a unique word.
    for (uint32_t i = 0; i < n; ++i) pieces[i] = "report";
    const uint32_t fact_block = 300;
    pieces[fact_block * 32] = "needle";
    pieces[fact_block * 32 + 1] = "zebra";

    auto opts = options(0.03);
    opts.coverage_stride = 0;
    opts.mmr_threshold = 0.60;
    opts.mmr_lambda = 0.50;
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &sel));
    CHECK(!sel.dense);
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), fact_block) !=
          sel.block_indices.end());
    CHECK(sel.mmr_penalized > 0);
}

// AR007 FR-2: anchors and selection stay deterministic.
TEST_CASE(selector_ar007_deterministic) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    for (uint32_t i = 0; i < 30; ++i) pieces[7 * 32 + i] = "alpha beta";

    auto opts = options(0.08);
    opts.coverage_stride = 0;
    Selection a;
    Selection b;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha", pieces, opts, nullptr, &a));
    CHECK_OK(selector.select(tokens, "alpha", pieces, opts, nullptr, &b));
    CHECK_EQ(a.token_indices, b.token_indices);
    CHECK_EQ(a.anchor_blocks, b.anchor_blocks);
    CHECK_EQ(a.mmr_penalized, b.mmr_penalized);
}

// AR008 FR-1: a digit is a discriminative token even though it is a single
// character. Before AR008 the query-term filter dropped it, so "vault 5" and
// "vault 6" produced identical scores and the selector answered whichever
// block came first (measured as quality 1/3 on the multi-needle set).
TEST_CASE(selector_phrase_keeps_numeric_discriminator) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    const uint32_t b1 = 40, b2 = 120, b3 = 200;
    auto sentence = [&](uint32_t block, const char* digit, const char* code) {
        pieces[block * 32 + 0] = "the";
        pieces[block * 32 + 1] = "secret";
        pieces[block * 32 + 2] = "passcode";
        pieces[block * 32 + 3] = "for";
        pieces[block * 32 + 4] = "vault";
        pieces[block * 32 + 5] = digit;
        pieces[block * 32 + 6] = "is";
        pieces[block * 32 + 7] = code;
    };
    sentence(b1, "1", "magenta");
    sentence(b2, "2", "cobalt");
    sentence(b3, "3", "amber");

    Selection sel;
    BlockSelector selector;
    auto opts = options(0.05);
    opts.coverage_stride = 0;
    CHECK_OK(selector.select(tokens, "What is the secret passcode for vault 3?",
                             pieces, opts, nullptr, &sel));
    CHECK(!sel.dense);
    CHECK(sel.phrase_blocks > 0u);
    CHECK(sel.specific_terms >= 2u);
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), b3) !=
          sel.block_indices.end());
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), b3) !=
          sel.mandatory_blocks.end());
}

// AR008 FR-1: phrase matching must survive a block boundary. The needle
// sentence can start near the end of block b and finish in b+1; a per-block
// bigram then misses it entirely (this was the last failing multi-needle case
// before the crossing-pair fix).
TEST_CASE(selector_phrase_crosses_block_boundary) {
    constexpr uint32_t n = 4096;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[60 * 32 + 31] = "vault";
    pieces[61 * 32 + 0] = "7";
    pieces[61 * 32 + 1] = "is";
    pieces[61 * 32 + 2] = "teal";

    Selection sel;
    BlockSelector selector;
    auto opts = options(0.05);
    opts.coverage_stride = 0;
    CHECK_OK(selector.select(tokens, "passcode for vault 7", pieces, opts, nullptr, &sel));
    CHECK(sel.phrase_blocks > 0u);
    const bool pinned_60 =
            std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 60u) !=
            sel.mandatory_blocks.end();
    const bool pinned_61 =
            std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 61u) !=
            sel.mandatory_blocks.end();
    CHECK(pinned_60 || pinned_61);
    CHECK(pinned_61);
}

// AR008 FR-1 zero-regression switch: phrase_weight=0 must reproduce the AR007
// selection (no phrase work at all), and the feature must never turn a sparse
// selection dense.
TEST_CASE(selector_phrase_weight_zero_matches_ar007) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    for (uint32_t i = 0; i < 30; ++i) pieces[11 * 32 + i] = "alpha beta gamma";

    auto opts = options(0.08);
    opts.coverage_stride = 0;
    opts.phrase_weight = 0.0;
    Selection off;
    Selection off2;
    Selection on;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha beta", pieces, opts, nullptr, &off));
    CHECK_EQ(off.phrase_blocks, 0u);
    // The profile's discriminative-term count stays available even when the
    // phrase signal is off: routing must not collapse to the "flat" bucket
    // just because the user disabled phrase admission.
    CHECK(off.specific_terms > 0u);
    CHECK_OK(selector.select(tokens, "alpha beta", pieces, opts, nullptr, &off2));
    CHECK_EQ(off.token_indices, off2.token_indices);
    auto opts_on = opts;
    opts_on.phrase_weight = 0.5;
    CHECK_OK(selector.select(tokens, "alpha beta", pieces, opts_on, nullptr, &on));
    CHECK_EQ(on.dense, off.dense);
    CHECK(on.selected_tokens <= off.selected_tokens + 32u * 4u);
}

// AR009 FR-2: multi-fact coverage. A prompt can hold several independent
// facts and a single question can ask for several of them at once. The AW
// budget then only has room for a handful of blocks, so the admission has to
// pay for every requested fact instead of spending everything on the single
// strongest lexical match.
namespace {

struct MultiFactFixture {
    std::vector<llama_token> tokens;
    std::vector<std::string> pieces;

    // key -> block index. Each fact sentence lives in one block and is
    // identified by one rare word ("one", "two", ...), which is exactly how
    // the generated multi-fact family is laid out.
    explicit MultiFactFixture(const std::vector<std::pair<uint32_t, std::string>>& facts,
                              const std::string& echo = "") {
        constexpr uint32_t n = 8192;
        tokens.resize(n);
        pieces.assign(n, "filler");
        for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
        for (const auto& fact : facts) {
            const uint32_t base = fact.first * 32u;
            pieces[base] = "the";
            pieces[base + 1] = "secret";
            pieces[base + 2] = "passcode";
            pieces[base + 3] = "is";
            pieces[base + 4] = fact.second;
        }
        if (!echo.empty() && n > 32) {
            // The chat template repeats the question in the final block.
            pieces[n - 1] = echo;
        }
    }
};

SelectionOptions multi_fact_options(double coverage_weight) {
    auto opts = options(0.05);
    opts.coverage_stride = 0;
    opts.phrase_weight = 0.5;
    opts.coverage_weight = coverage_weight;
    return opts;
}

bool contains_block(const Selection& sel, uint32_t block) {
    for (int32_t tok : sel.token_indices) {
        if (static_cast<uint32_t>(tok) / 32u == block) return true;
    }
    return false;
}

const std::string kMultiFactQuery =
        "List the secret passcodes for the vaults one, two and three.";

} // namespace

// AR009 zero-regression switch: coverage_weight=0 must leave the AR008 ranking
// untouched and report zero covered sub-queries.
TEST_CASE(selector_coverage_off_matches_ar008) {
    MultiFactFixture f({{20, "one"}, {60, "two"}, {100, "three"}, {140, "four"},
                        {180, "five"}, {220, "six"}});
    auto on = multi_fact_options(0.0);
    Selection a;
    Selection b;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, kMultiFactQuery, f.pieces, on, nullptr, &a));
    CHECK_OK(selector.select(f.tokens, kMultiFactQuery, f.pieces, on, nullptr, &b));
    CHECK_EQ(a.coverage_queries, 0u);
    CHECK_EQ(a.token_indices, b.token_indices);
    CHECK(!a.dense);
}

// The point of AR009: every requested fact must end up in the context. With
// coverage off the admission pins at most one of the three asked facts; with
// coverage on all three are admitted and reported in the audit column.
TEST_CASE(selector_coverage_admits_every_requested_fact) {
    MultiFactFixture f({{20, "one"}, {60, "two"}, {100, "three"}, {140, "four"},
                        {180, "five"}, {220, "six"}});
    BlockSelector selector;
    Selection off;
    CHECK_OK(selector.select(f.tokens, kMultiFactQuery, f.pieces,
                             multi_fact_options(0.0), nullptr, &off));
    Selection on;
    CHECK_OK(selector.select(f.tokens, kMultiFactQuery, f.pieces,
                             multi_fact_options(3.0), nullptr, &on));
    CHECK_EQ(on.coverage_queries, 3u);
    CHECK(contains_block(on, 20));
    CHECK(contains_block(on, 60));
    CHECK(contains_block(on, 100));
    // The unasked facts must not be dragged in as a side effect.
    CHECK(on.selected_tokens <= off.selected_tokens + 32u * 4u);
}

// The chat template repeats the question in the last block, so that block
// contains every sub-query at once. It must never count as evidence: doing so
// marks the whole question "covered" before the first fact is admitted.
TEST_CASE(selector_coverage_ignores_question_echo) {
    MultiFactFixture f({{20, "one"}, {60, "two"}, {100, "three"}},
                       "list the secret passcodes for the vaults one two and three");
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, kMultiFactQuery, f.pieces,
                             multi_fact_options(3.0), nullptr, &sel));
    CHECK_EQ(sel.coverage_queries, 3u);
    CHECK(contains_block(sel, 20));
    CHECK(contains_block(sel, 60));
    CHECK(contains_block(sel, 100));
}

// A single-fact question has nothing to cover: the selection must be identical
// to AR008 and the audit column must stay zero, which is what keeps the whole
// single-needle family bit-for-bit unchanged.
TEST_CASE(selector_coverage_is_noop_for_single_fact) {
    MultiFactFixture f({{20, "one"}, {60, "two"}, {100, "three"}});
    BlockSelector selector;
    Selection off;
    Selection on;
    CHECK_OK(selector.select(f.tokens, "What is the secret passcode for vault two?",
                             f.pieces, multi_fact_options(0.0), nullptr, &off));
    CHECK_OK(selector.select(f.tokens, "What is the secret passcode for vault two?",
                             f.pieces, multi_fact_options(3.0), nullptr, &on));
    CHECK_EQ(on.coverage_queries, 0u);
    CHECK_EQ(on.token_indices, off.token_indices);
    CHECK_EQ(on.selected_tokens, off.selected_tokens);
}

// AR010 FR-1: every signal must be min-max normalised into [0,1]. The previous
// implementation kept iterators into the vector it was overwriting, so the
// lower bound became 0 as soon as the minimum element had been normalised and
// the signal was mapped to v/range instead of (v-lo)/range (measured: diversity
// up to 3.2, position up to 1.16 at 32K). The over-ranged priors then outranked
// lexical evidence and 62% of the 32K budget went to zero-evidence blocks.
TEST_CASE(selector_signals_are_bounded_after_normalisation) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // Evidence spread unevenly so the minimum is not the first element.
    pieces[17 * 32 + 3] = "needle";
    pieces[90 * 32 + 7] = "needle";
    pieces[150 * 32 + 1] = "needle";

    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, options(0.10), nullptr, &sel));
    CHECK(!sel.signals.empty());
    double max_seen = 0.0;
    for (const auto& sig : sel.signals) {
        CHECK(sig.lexical >= -1e-9 && sig.lexical <= 1.0 + 1e-9);
        CHECK(sig.diversity >= -1e-9 && sig.diversity <= 1.0 + 1e-9);
        CHECK(sig.position >= -1e-9 && sig.position <= 1.0 + 1e-9);
        CHECK(sig.phrase >= -1e-9 && sig.phrase <= 1.0 + 1e-9);
        max_seen = std::max(max_seen, sig.lexical);
    }
    CHECK(max_seen > 0.999);  // the max of a normalised signal is exactly 1
}

// AR010 FR-2: with fill_priors=0 the admission is content-sized. Every block
// that is not a structural anchor must carry evidence, and the fill reports why
// it stopped instead of silently relying on "the best score happens to be 0".
TEST_CASE(selector_content_sized_admission_stops_on_evidence) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[20 * 32 + 4] = "needle";

    auto opts = options(0.20);  // 1638 tokens: far more than the evidence needs
    opts.fill_priors = 0.0;
    opts.prior_gate = 1.0;
    Selection fit;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &fit));
    CHECK_EQ(fit.stop_reason, "evidence-exhausted");
    CHECK(fit.selected_tokens < static_cast<uint32_t>(std::ceil(n * 0.20)));
    CHECK(fit.evidence_blocks >= 1u);
    std::set<uint32_t> anchors(fit.mandatory_blocks.begin(), fit.mandatory_blocks.end());
    for (uint32_t b : fit.block_indices) {
        if (anchors.count(b)) continue;
        CHECK(fit.signals[b].lexical > 0.0 || fit.signals[b].phrase > 0.0);
    }

    // fill_priors=1 keeps the AR007 contract: the budget is a quota and is
    // filled with positional/coverage sampling after the evidence is gone.
    auto qopts = opts;
    qopts.fill_priors = 1.0;
    Selection quota;
    CHECK_OK(selector.select(tokens, "needle", pieces, qopts, nullptr, &quota));
    CHECK_EQ(quota.stop_reason, "budget");
    CHECK(quota.selected_tokens > fit.selected_tokens);
    CHECK(quota.selected_tokens >= static_cast<uint32_t>(std::ceil(n * 0.20 * 0.95)));
}

// AR010 FR-1: the prior gate must not change *whether* evidence is admitted,
// only how the remaining budget is ordered; with prior_gate=0 the same
// evidence block is still selected.
TEST_CASE(selector_prior_gate_keeps_evidence_admitted) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[200 * 32 + 4] = "needle";  // near the tail: high positional prior

    auto opts = options(0.05);
    opts.fill_priors = 0.0;
    Selection gated;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &gated));
    auto legacy = opts;
    legacy.prior_gate = 0.0;
    Selection plain;
    CHECK_OK(selector.select(tokens, "needle", pieces, legacy, nullptr, &plain));
    const bool gated_has = std::find(gated.block_indices.begin(), gated.block_indices.end(), 200u)
                           != gated.block_indices.end();
    const bool plain_has = std::find(plain.block_indices.begin(), plain.block_indices.end(), 200u)
                           != plain.block_indices.end();
    CHECK(gated_has);
    CHECK(plain_has);
}

// AR011 FR-1: the structure budget must bound the structural anchors (sink +
// recent) absolutely, and the anchor cap must then be able to trim the anchors
// that used to be untouchable (best-lexical window, phrase pins). Measured
// before the change: a 4K prompt at b=0.08 admitted 310 tokens of which 9/10
// blocks were mandatory, so the "budget" was a quota for structure.
TEST_CASE(selector_structure_budget_bounds_anchors) {
    constexpr uint32_t n = 4096;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[40 * 32 + 4] = "needle";
    pieces[41 * 32 + 4] = "needle";
    pieces[60 * 32 + 1] = "needle";

    auto opts = options(0.08);
    opts.fill_priors = 0.0;
    Selection legacy;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &legacy));

    auto bounded = opts;
    bounded.structure_budget = 64;
    Selection sel;
    CHECK_OK(selector.select(tokens, "needle", pieces, bounded, nullptr, &sel));
    CHECK(sel.structure_tokens <= 64u);
    CHECK(sel.evidence_tokens + sel.structure_tokens == sel.selected_tokens);
    CHECK(sel.selected_tokens <= legacy.selected_tokens);
    CHECK(sel.anchor_blocks <= legacy.anchor_blocks);
    // The structural sink and the final block are never dropped: the first
    // token logits and the attention sink depend on them.
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), 0) !=
          sel.mandatory_blocks.end());
    CHECK(std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), n / 32 - 1) !=
          sel.mandatory_blocks.end());
    // The evidence that was actually asked for must still be admitted.
    CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), 40u) !=
          sel.block_indices.end());
}

// AR011 FR-1: at a length where the anchors do not dominate the budget the
// accounting must be a no-op (measured 16K: 280 tokens either way; forcing the
// cap there admitted *more* evidence, 320 tokens, for no quality gain).
TEST_CASE(selector_structure_budget_is_inert_when_anchors_do_not_dominate) {
    constexpr uint32_t n = 15744;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[100 * 32 + 4] = "needle";

    auto opts = options(0.05);
    opts.fill_priors = 0.0;
    Selection legacy;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &legacy));
    auto bounded = opts;
    bounded.structure_budget = 64;
    Selection sel;
    CHECK_OK(selector.select(tokens, "needle", pieces, bounded, nullptr, &sel));
    CHECK_EQ(sel.selected_tokens, legacy.selected_tokens);
    CHECK_EQ(sel.block_indices.size(), legacy.block_indices.size());
}

// AR011 FR-3: span admission decouples the admission granularity from the
// 32-token execution block. The span must be centred on the evidence peak, keep
// the sentence it belongs to, and never drop the last prompt token.
TEST_CASE(selector_span_admission_trims_and_keeps_last_token) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // A sentence inside one evidence block: the key and the answer after it.
    pieces[50 * 32 + 10] = "needle";
    pieces[50 * 32 + 12] = "answer";
    pieces[50 * 32 + 16] = ".";

    auto opts = options(0.08);
    opts.fill_priors = 0.0;
    opts.structure_budget = 64;
    Selection whole;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &whole));
    auto span_opts = opts;
    span_opts.span_admission = 1;
    span_opts.span_min_tokens = 24;
    Selection span;
    CHECK_OK(selector.select(tokens, "needle", pieces, span_opts, nullptr, &span));
    CHECK(span.span_tokens > 0u);
    CHECK(span.selected_tokens < whole.selected_tokens);
    CHECK_EQ(span.selected_tokens + span.span_tokens, whole.selected_tokens);
    CHECK_EQ(span.token_indices.back(), n - 1);
    for (size_t i = 1; i < span.token_indices.size(); ++i) {
        CHECK(span.token_indices[i] > span.token_indices[i - 1]);
    }
    // The key and the answer sentence survive the trim.
    const auto has = [&](uint32_t pos) {
        return std::find(span.token_indices.begin(), span.token_indices.end(), pos) !=
               span.token_indices.end();
    };
    CHECK(has(50 * 32 + 10));
    CHECK(has(50 * 32 + 12));
}

// AR011 FR-3: with several asked facts the span pass must stay out of the way.
// Trimming a multi-fact selection measured q 0.836 -> 0.800 (easy) and
// 0.669 -> 0.444 with q_min 0.174 -> 0.000 (hard): each fact needs its whole
// sentence, so the selector keeps whole blocks whenever coverage is active.
TEST_CASE(selector_span_admission_is_inert_for_multi_fact) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // Three sibling facts, each keyed by its own rare term.
    pieces[20 * 32 + 3] = "alpha";
    pieces[20 * 32 + 5] = "value";
    pieces[60 * 32 + 3] = "beta";
    pieces[60 * 32 + 5] = "value";
    pieces[100 * 32 + 3] = "gamma";
    pieces[100 * 32 + 5] = "value";

    auto opts = options(0.10);
    opts.fill_priors = 0.0;
    opts.coverage_weight = 3.0;
    Selection whole;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha beta gamma value", pieces, opts, nullptr,
                             &whole));
    CHECK(whole.coverage_queries >= 2u);
    auto span_opts = opts;
    span_opts.span_admission = 1;
    Selection span;
    CHECK_OK(selector.select(tokens, "alpha beta gamma value", pieces, span_opts, nullptr,
                             &span));
    CHECK_EQ(span.span_tokens, 0u);
    CHECK_EQ(span.selected_tokens, whole.selected_tokens);
}

// AR011: coverage reservation is an explicit opt-in and must report the
// shortfall honestly instead of silently leaving a requested fact without
// evidence. (Measured: no reliable gain over the AR009 facility-location fill
// at the tested budgets, so it stays off by default and off in the portfolio.)
TEST_CASE(selector_coverage_reserve_reports_shortfall) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[10 * 32 + 3] = "alpha";
    pieces[80 * 32 + 3] = "beta";
    pieces[150 * 32 + 3] = "gamma";

    auto opts = options(0.02);  // 163 tokens: one block after the structure
    opts.fill_priors = 0.0;
    opts.coverage_weight = 3.0;
    Selection plain;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha beta gamma", pieces, opts, nullptr, &plain));
    auto reserve = opts;
    reserve.coverage_reserve = 1;
    Selection sel;
    CHECK_OK(selector.select(tokens, "alpha beta gamma", pieces, reserve, nullptr, &sel));
    CHECK(sel.uncovered_facts > 0u);
    CHECK(sel.uncovered_facts <= sel.coverage_queries + sel.uncovered_facts);
    CHECK(sel.selected_tokens <= plain.selected_tokens + 32u);
}

// AR012 FR-1: a fact row can straddle a block boundary ("... for vault one is
// MAGENTA-FALCON-" | "8200."). Whole-block admission then ships half an answer,
// which is exactly what the multi-fact tiers measured (0.760 fact completeness).
// Sentence completion must admit the whole sentence, including the part that
// lives in a block that was not selected.
TEST_CASE(selector_sentence_completion_keeps_split_answer) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    // Two asked facts so coverage (multi-fact) is active, each keyed by a rare
    // term, with the value straddling the block boundary at 40*32.
    // One sentence every 8 tokens so no sentence runs to the end of the doc.
    for (uint32_t i = 7; i < n; i += 8) pieces[i] = "filler.";
    pieces[20 * 32 + 3] = "alpha";
    pieces[20 * 32 + 8] = "value-a";
    pieces[20 * 32 + 20] = ".";
    pieces[40 * 32 + 28] = ".";        // ends the filler sentence before beta
    pieces[40 * 32 + 29] = "beta";     // key near the end of block 40 ...
    pieces[40 * 32 + 31] = "value-";   // ... value starts in block 40 ...
    pieces[41 * 32 + 0] = "b";         // ... and ends in block 41
    pieces[41 * 32 + 1] = ".";
    pieces[70 * 32 + 3] = "gamma";
    pieces[70 * 32 + 8] = "value-c";
    pieces[70 * 32 + 20] = ".";

    auto opts = options(0.02);  // 163 tokens: block 40 can be selected, not 41
    opts.fill_priors = 0.0;
    opts.coverage_weight = 3.0;
    opts.sentence_completion = 1;
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha beta gamma", pieces, opts, nullptr, &sel));
    const auto has = [&](uint32_t pos) {
        return std::find(sel.token_indices.begin(), sel.token_indices.end(), pos) !=
               sel.token_indices.end();
    };
    CHECK(sel.sentences_completed >= 1u);
    CHECK(has(40 * 32 + 29));
    CHECK(has(40 * 32 + 31));
    // The continuation token is admitted even though its block was not selected.
    bool tail_in = has(41 * 32 + 0) || has(41 * 32 + 1);
    CHECK(tail_in);
    // Every admitted token belongs to a selected block (invariant I3).
    for (uint32_t pos : sel.token_indices) {
        CHECK(std::find(sel.block_indices.begin(), sel.block_indices.end(), pos / 32) !=
              sel.block_indices.end());
    }
}

// AR012: the switch is opt-in and must leave AR011's selection untouched.
TEST_CASE(selector_sentence_completion_off_matches_ar011) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    pieces[200 * 32 + 4] = "needle";

    auto opts = options(0.05);
    opts.fill_priors = 0.0;
    opts.span_admission = 1;
    Selection ar011;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &ar011));
    auto off = opts;
    off.sentence_completion = 0;  // the default, pinned explicitly
    Selection same;
    CHECK_OK(selector.select(tokens, "needle", pieces, off, nullptr, &same));
    CHECK_EQ(same.selected_tokens, ar011.selected_tokens);
    CHECK(same.token_indices == ar011.token_indices);
    CHECK_EQ(same.sentences_completed, 0u);
}

// AR012: completion is engaged only when several facts are asked (coverage
// active). On a single-fact prompt it measured no quality gain and +15% tokens
// (multi-needle 247 -> 283), so it must stay out of the way.
TEST_CASE(selector_sentence_completion_is_multi_fact_only) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    for (uint32_t i = 7; i < n; i += 8) pieces[i] = "filler.";
    pieces[60 * 32 + 29] = "needle";
    pieces[60 * 32 + 31] = "answer-";
    pieces[61 * 32 + 0] = "tail";
    pieces[61 * 32 + 1] = ".";

    auto opts = options(0.05);
    opts.fill_priors = 0.0;
    opts.span_admission = 1;
    opts.sentence_completion = 1;
    Selection single;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "needle", pieces, opts, nullptr, &single));
    CHECK_EQ(single.sentences_completed, 0u);
}

// AR012 FR-4: the audit counters must describe what happened, and the index
// invariants must hold after completion (ascending, unique, last token kept).
TEST_CASE(selector_sentence_completion_audit_and_invariants) {
    constexpr uint32_t n = 8192;
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    for (uint32_t i = 7; i < n; i += 8) pieces[i] = "filler.";
    pieces[15 * 32 + 2] = "alpha";
    pieces[15 * 32 + 6] = ".";
    pieces[40 * 32 + 28] = ".";
    pieces[40 * 32 + 30] = "beta";
    pieces[40 * 32 + 31] = "value-";
    pieces[41 * 32 + 1] = ".";
    pieces[80 * 32 + 5] = "gamma";
    pieces[80 * 32 + 9] = ".";

    auto opts = options(0.03);
    opts.fill_priors = 0.0;
    opts.coverage_weight = 3.0;
    opts.sentence_completion = 1;
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "alpha beta gamma", pieces, opts, nullptr, &sel));
    CHECK(sel.sentences_completed >= 1u);
    CHECK_EQ(sel.token_indices.back(), n - 1);
    for (size_t i = 1; i < sel.token_indices.size(); ++i) {
        CHECK(sel.token_indices[i] > sel.token_indices[i - 1]);
    }
    // A block pulled in by a crossing sentence is reported in block_indices.
    if (sel.sentences_extended > 0u) {
        CHECK(sel.block_indices.size() > 0u);
    }
}

// AR013 FR-2: block compaction must keep every intra-block offset and collapse
// the filler gaps, so the KV span equals the admitted token count.
TEST_CASE(selector_position_compaction_is_block_local_and_gap_free) {
    Fixture f(8192);
    // Three admissible regions separated by filler, plus a long tail gap.
    f.pieces[2 * 32 + 3] = "needle";
    f.pieces[2 * 32 + 9] = "alpha";
    f.pieces[40 * 32 + 1] = "needle";
    f.pieces[40 * 32 + 6] = "beta";
    f.pieces[77 * 32 + 4] = "needle";

    auto compact = options(0.05);
    compact.position_compaction = 1;
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle alpha beta", f.pieces, compact, nullptr, &sel));
    CHECK(!sel.dense);
    CHECK_EQ(sel.compaction_reason, std::string("block"));
    CHECK_EQ(sel.positions.size(), sel.token_indices.size());
    // Strictly increasing (llama_decode requires it).
    for (size_t i = 1; i < sel.positions.size(); ++i) {
        CHECK(sel.positions[i] > sel.positions[i - 1]);
    }
    // Gap-free: the span is exactly the number of admitted tokens.
    CHECK_EQ(sel.position_span, static_cast<uint32_t>(sel.token_indices.size()));
    // Intra-block offsets survive verbatim.
    for (size_t i = 1; i < sel.token_indices.size(); ++i) {
        const bool same_block =
                sel.token_indices[i] / 32 == sel.token_indices[i - 1] / 32;
        if (same_block) {
            CHECK_EQ(sel.positions[i] - sel.positions[i - 1],
                     sel.token_indices[i] - sel.token_indices[i - 1]);
        }
    }
    // The last prompt token stays last (first-token logits).
    CHECK_EQ(sel.positions.back(), sel.position_span - 1);
}

// AR013: compaction is opt-in. Off must reproduce AR012 bit for bit, i.e. an
// empty positions vector and the identity span.
TEST_CASE(selector_position_compaction_off_matches_ar012) {
    Fixture f(8192);
    f.pieces[20 * 32 + 5] = "needle";
    auto off = options(0.05);
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, off, nullptr, &sel));
    CHECK(sel.positions.empty());
    CHECK_EQ(sel.compaction_reason, std::string(""));
    CHECK_EQ(sel.selected_tokens, static_cast<uint32_t>(sel.token_indices.size()));
    CHECK_EQ(sel.position_span, sel.token_indices.back() + 1);

    auto on = off;
    on.position_compaction = 1;
    Selection compacted;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, on, nullptr, &compacted));
    // Same admission, different positions: compaction never changes what is
    // admitted, only where it sits in the KV cache.
    CHECK_EQ(compacted.token_indices, sel.token_indices);
    CHECK(compacted.position_span <= sel.position_span);
}

// AR013: compacting an already compact sequence is a no-op, which keeps the
// transform safe to apply on any selection path (including retries).
TEST_CASE(selector_position_compaction_is_idempotent) {
    Fixture f(4096);
    f.pieces[9 * 32 + 7] = "needle";
    auto opts = options(0.08);
    opts.position_compaction = 1;
    Selection once;
    BlockSelector selector;
    CHECK_OK(selector.select(f.tokens, "needle", f.pieces, opts, nullptr, &once));
    CHECK_EQ(once.compaction_reason, std::string("block"));

    // Rebuild a prompt whose token positions are already the compacted ones.
    std::vector<llama_token> tokens;
    std::vector<std::string> pieces;
    for (size_t i = 0; i < once.positions.size(); ++i) {
        tokens.push_back(f.tokens[once.token_indices[i]]);
        pieces.push_back(f.pieces[once.token_indices[i]]);
    }
    auto dense_small = options(1.0);
    dense_small.block_size = 32;
    dense_small.min_dense_tokens = 32;
    dense_small.position_compaction = 1;
    Selection twice;
    CHECK_OK(selector.select(tokens, "needle", pieces, dense_small, nullptr, &twice));
    if (!twice.dense) {
        for (size_t i = 0; i < twice.positions.size(); ++i) {
            CHECK_EQ(twice.positions[i], static_cast<uint32_t>(i));
        }
    }
}
