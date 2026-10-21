#include "test_framework.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/config.h"
#include "edge_prefill/selector.h"

using namespace edge_prefill;

namespace {

Selection make_selection(uint32_t n = 4096, double budget = 0.5) {
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    constexpr uint32_t needle_pos = 37 * 32 + 5;
    if (needle_pos < n) pieces[needle_pos] = "needle";

    SelectionOptions opts;
    opts.block_size = 32;
    opts.budget = budget;
    opts.min_dense_tokens = 512;
    Selection sel;
    BlockSelector selector;
    auto st = selector.select(tokens, "needle", pieces, opts, nullptr, &sel);
    if (!st.ok()) {
        CHECK_OK(st);
    }
    return sel;
}

AttentionSparsityOptions sparsity_options(uint32_t top_k_blocks) {
    AttentionSparsityOptions opts;
    opts.block_size = 32;
    opts.top_k_blocks = top_k_blocks;
    return opts;
}

} // namespace

TEST_CASE(attention_sparsity_indices_sorted) {
    const Selection sel = make_selection();
    CHECK(!sel.dense);
    AttentionSparsityOptions opts = sparsity_options(16);
    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, opts, &indices, &reason));
    CHECK(!indices.empty());
    CHECK(!reason.empty());
    // Strictly ascending and within token range.
    for (size_t i = 0; i < indices.size(); ++i) {
        CHECK(indices[i] >= 0);
        CHECK(static_cast<uint32_t>(indices[i]) < sel.total_tokens);
        if (i > 0) CHECK(indices[i] > indices[i - 1]);
    }
    // Indices are a subset of the prefill positions.
    for (int32_t idx : indices) {
        const bool found = std::binary_search(sel.token_indices.begin(),
                                              sel.token_indices.end(),
                                              static_cast<uint32_t>(idx));
        CHECK(found);
    }
    // The last prompt token must stay gatherable (first-token logits).
    CHECK_EQ(indices.back(), static_cast<int32_t>(sel.total_tokens - 1));
}

TEST_CASE(attention_sparsity_disabled) {
    const Selection sel = make_selection();
    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, sparsity_options(0), &indices, &reason));
    CHECK(indices.empty());
    CHECK_EQ(reason, std::string("attn-disabled"));
}

TEST_CASE(attention_sparsity_dense_selection) {
    const Selection sel = make_selection(256, 0.5);
    CHECK(sel.dense);
    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, sparsity_options(16), &indices, &reason));
    CHECK(indices.empty());
    CHECK_EQ(reason, std::string("dense"));
}

TEST_CASE(attention_sparsity_budget) {
    const Selection sel = make_selection(8192, 0.5);
    const uint32_t n_selected_blocks = static_cast<uint32_t>(sel.block_indices.size());
    const uint32_t n_mandatory = static_cast<uint32_t>(sel.mandatory_blocks.size());
    // The cap must leave room beyond the mandatory anchors, otherwise the
    // top-k ranking cannot add any optional block.
    const uint32_t cap = n_mandatory + 8;
    CHECK(n_selected_blocks > cap);

    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, sparsity_options(cap), &indices, &reason));
    CHECK(!indices.empty());
    // Kept block count never exceeds the cap plus mandatory anchors.
    std::vector<bool> in_index(sel.total_tokens, false);
    for (int32_t idx : indices) in_index[static_cast<uint32_t>(idx)] = true;
    uint32_t kept_blocks = 0;
    for (uint32_t b : sel.block_indices) {
        bool any = false;
        const uint32_t begin = b * 32;
        const uint32_t end = std::min(sel.total_tokens, begin + 32);
        for (uint32_t i = begin; i < end; ++i) any = any || in_index[i];
        if (any) ++kept_blocks;
    }
    CHECK(kept_blocks <= cap + n_mandatory);
    // Mandatory anchors are never dropped.
    for (uint32_t b : sel.mandatory_blocks) {
        bool any = false;
        const uint32_t begin = b * 32;
        const uint32_t end = std::min(sel.total_tokens, begin + 32);
        for (uint32_t i = begin; i < end; ++i) any = any || in_index[i];
        CHECK(any);
    }
    // Highest fused blocks are preferred: the best non-mandatory block must
    // be kept.
    uint32_t best_block = 0;
    double best_fused = -1.0;
    for (uint32_t b : sel.block_indices) {
        if (std::find(sel.mandatory_blocks.begin(), sel.mandatory_blocks.end(), b) !=
            sel.mandatory_blocks.end()) {
            continue;
        }
        if (sel.signals[b].fused > best_fused) {
            best_fused = sel.signals[b].fused;
            best_block = b;
        }
    }
    bool best_kept = false;
    for (int32_t idx : indices) {
        if (static_cast<uint32_t>(idx) / 32 == best_block) best_kept = true;
    }
    CHECK(best_kept);
}

TEST_CASE(attention_sparsity_all_tokens) {
    const Selection sel = make_selection(4096, 0.5);
    // A cap that covers every selected block must not engage the gather:
    // gathering everything equals the unpatched path.
    const uint32_t all = static_cast<uint32_t>(sel.block_indices.size());
    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, sparsity_options(all), &indices, &reason));
    CHECK(indices.empty());
    CHECK_EQ(reason, std::string("attn-top-k-covers-all"));
    // A cap beyond the block count behaves the same.
    CHECK_OK(AttentionSparsityController::build(sel, sparsity_options(all + 100), &indices,
                                                &reason));
    CHECK(indices.empty());
}

TEST_CASE(attention_sparsity_invalid_inputs) {
    const Selection sel = make_selection();
    std::vector<int32_t> indices;
    std::string reason;

    // Null output.
    CHECK_ERR(AttentionSparsityController::build(sel, sparsity_options(8), nullptr, &reason),
              ErrorCode::InvalidArgument);
    // Zero block size.
    auto bad = sparsity_options(8);
    bad.block_size = 0;
    CHECK_ERR(AttentionSparsityController::build(sel, bad, &indices, &reason),
              ErrorCode::InvalidArgument);
    // per_layer without a layer list.
    bad = sparsity_options(8);
    bad.per_layer = true;
    CHECK_ERR(AttentionSparsityController::build(sel, bad, &indices, &reason),
              ErrorCode::InvalidArgument);
    // per_layer with a list is accepted (layer count is validated by the
    // backend caller).
    bad.layer_top_k = {8, 16, 32};
    CHECK_OK(AttentionSparsityController::build(sel, bad, &indices, &reason));
    CHECK(!indices.empty());

    // Non-dense selection with no token indices.
    Selection broken = sel;
    broken.token_indices.clear();
    CHECK_ERR(AttentionSparsityController::build(broken, sparsity_options(8), &indices, &reason),
              ErrorCode::InvalidArgument);

    // Signals size mismatch.
    Selection bad_signals = sel;
    bad_signals.signals.pop_back();
    CHECK_ERR(AttentionSparsityController::build(bad_signals, sparsity_options(8), &indices,
                                                 &reason),
              ErrorCode::InvalidArgument);

    // Out-of-range token index.
    Selection out_of_range = sel;
    out_of_range.token_indices.back() = sel.total_tokens + 7;
    CHECK_ERR(AttentionSparsityController::build(out_of_range, sparsity_options(8), &indices,
                                                 &reason),
              ErrorCode::InvalidArgument);

    // Non-ascending token indices.
    Selection not_sorted = sel;
    std::reverse(not_sorted.token_indices.begin(), not_sorted.token_indices.end());
    CHECK_ERR(AttentionSparsityController::build(not_sorted, sparsity_options(8), &indices,
                                                 &reason),
              ErrorCode::InvalidArgument);
}

TEST_CASE(attention_sparsity_config_parse) {
    std::vector<uint32_t> values;
    CHECK_OK(parse_u32_list("4,8,16", &values));
    CHECK_EQ(values.size(), 3u);
    CHECK_EQ(values[0], 4u);
    CHECK_EQ(values[1], 8u);
    CHECK_EQ(values[2], 16u);
    CHECK_OK(parse_u32_list(" 7 ", &values));
    CHECK_EQ(values.size(), 1u);
    CHECK_EQ(values[0], 7u);
    CHECK_ERR(parse_u32_list("", &values), ErrorCode::InvalidArgument);
    CHECK_ERR(parse_u32_list("4,,8", &values), ErrorCode::InvalidArgument);
    CHECK_ERR(parse_u32_list("4,x", &values), ErrorCode::InvalidArgument);
    CHECK_ERR(parse_u32_list("-1", &values), ErrorCode::InvalidArgument);
    CHECK_ERR(parse_u32_list("4", nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(attention_sparsity_plan_fields) {
    // The plan command must expose the new attention field end to end.
    // Validated at runtime in integration tests; here we check the Config
    // plumbing only.
    Config cfg;
    CHECK_EQ(cfg.attn_top_k, 0u);
    CHECK(cfg.attn_top_k_layers.empty());
    Plan plan;
    CHECK_EQ(plan.attention_top_k, 0u);
    const std::string json = plan.to_json();
    CHECK(json.find("attention_top_k") != std::string::npos);
}
