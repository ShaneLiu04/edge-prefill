#include "test_framework.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/selector.h"

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
