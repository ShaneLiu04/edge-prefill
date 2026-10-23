#include "test_framework.h"

#include <algorithm>
#include <numeric>
#include <set>
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

namespace {

HeadProfile make_profile(int32_t n_layer, int32_t n_head, uint32_t retrieval_period) {
    HeadProfile profile;
    profile.n_layer = n_layer;
    profile.n_head = n_head;
    for (int32_t l = 0; l < n_layer; ++l) {
        for (int32_t h = 0; h < n_head; ++h) {
            HeadProfileEntry entry;
            entry.layer = l;
            entry.head = h;
            entry.echo_score = 0.01;
            entry.induction_score = 0.01;
            entry.retrieval = (static_cast<uint32_t>(l * n_head + h) % retrieval_period) == 0;
            profile.heads.push_back(entry);
        }
    }
    return profile;
}

} // namespace

TEST_CASE(head_adaptive_build_pool_and_matrix) {
    const Selection sel = make_selection(8192, 0.5);
    CHECK(!sel.dense);
    HeadProfile profile = make_profile(4, 4, 3);
    HeadAdaptiveOptions opts;
    opts.block_size = 32;
    opts.top_k_rows = 64;
    std::vector<int32_t> pool;
    std::vector<int32_t> matrix;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_head_adaptive(sel, profile, opts, &pool,
                                                              &matrix, &reason));
    CHECK(!pool.empty());
    CHECK(!matrix.empty());
    CHECK(!reason.empty());

    // Matrix: one entry per layer x head, -1 exactly on retrieval heads.
    CHECK_EQ(matrix.size(), profile.heads.size());
    for (size_t i = 0; i < profile.heads.size(); ++i) {
        CHECK_EQ(matrix[i], profile.heads[i].retrieval ? -1 : 64);
    }
    // Exactly two distinct matrix values keeps the graph's group limit.
    std::vector<int32_t> distinct(matrix);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    CHECK_EQ(distinct.size(), 2u);

    // Pool: a permutation of [0, n_selected) — every cache row exactly once.
    CHECK_EQ(pool.size(), sel.token_indices.size());
    std::vector<int32_t> sorted_pool(pool);
    std::sort(sorted_pool.begin(), sorted_pool.end());
    for (size_t r = 0; r < sorted_pool.size(); ++r) {
        CHECK_EQ(sorted_pool[r], static_cast<int32_t>(r));
    }

    // Mandatory block rows come first, in ascending block order.
    size_t mandatory_rows = 0;
    for (uint32_t b : sel.mandatory_blocks) {
        const uint32_t begin = b * 32;
        const uint32_t end = std::min(sel.total_tokens, begin + 32);
        for (uint32_t t : sel.token_indices) {
            if (t >= begin && t < end) ++mandatory_rows;
        }
    }
    CHECK(mandatory_rows > 0);
    CHECK(mandatory_rows < pool.size());
    std::set<int32_t> head_rows(pool.begin(), pool.begin() + mandatory_rows);
    CHECK_EQ(head_rows.size(), mandatory_rows);
    for (uint32_t b : sel.mandatory_blocks) {
        const uint32_t begin = b * 32;
        const uint32_t end = std::min(sel.total_tokens, begin + 32);
        for (uint32_t t : sel.token_indices) {
            if (t >= begin && t < end) {
                const auto it = std::lower_bound(sel.token_indices.begin(),
                                                 sel.token_indices.end(), t);
                const int32_t row = static_cast<int32_t>(it - sel.token_indices.begin());
                CHECK(head_rows.count(row) == 1);
            }
        }
    }
}

TEST_CASE(head_adaptive_invalid_inputs) {
    const Selection sel = make_selection();
    HeadProfile profile = make_profile(2, 2, 2);
    HeadAdaptiveOptions opts;
    opts.block_size = 32;
    opts.top_k_rows = 32;
    std::vector<int32_t> pool;
    std::vector<int32_t> matrix;
    std::string reason;

    CHECK_ERR(AttentionSparsityController::build_head_adaptive(sel, profile, opts, nullptr,
                                                               &matrix, &reason),
              ErrorCode::InvalidArgument);
    auto zero_rows = opts;
    zero_rows.top_k_rows = 0;
    CHECK_ERR(AttentionSparsityController::build_head_adaptive(sel, profile, zero_rows, &pool,
                                                               &matrix, &reason),
              ErrorCode::InvalidArgument);
    auto zero_block = opts;
    zero_block.block_size = 0;
    CHECK_ERR(AttentionSparsityController::build_head_adaptive(sel, profile, zero_block, &pool,
                                                               &matrix, &reason),
              ErrorCode::InvalidArgument);
    HeadProfile broken = profile;
    broken.heads.pop_back();
    CHECK_ERR(AttentionSparsityController::build_head_adaptive(sel, broken, opts, &pool,
                                                               &matrix, &reason),
              ErrorCode::InvalidArgument);

    // Dense selection reports "dense" and leaves outputs empty.
    const Selection dense = make_selection(256, 0.5);
    CHECK(dense.dense);
    CHECK_OK(AttentionSparsityController::build_head_adaptive(dense, profile, opts, &pool,
                                                              &matrix, &reason));
    CHECK(pool.empty());
    CHECK(matrix.empty());
    CHECK_EQ(reason, std::string("dense"));
}

// AR004 FR-4: pattern routing classifies every head into one of four
// classes and maps the class to a gather budget.
TEST_CASE(pattern_matrix_classifies_heads) {
    HeadProfile profile;
    profile.n_layer = 1;
    profile.n_head = 4;
    HeadProfileEntry e;
    e.layer = 0;
    e.head = 0;
    e.retrieval = true;
    profile.heads.push_back(e);
    e = HeadProfileEntry{};
    e.layer = 0;
    e.head = 1;
    e.echo_score = 0.7;
    profile.heads.push_back(e);
    e = HeadProfileEntry{};
    e.layer = 0;
    e.head = 2;
    e.induction_score = 0.6;
    profile.heads.push_back(e);
    e = HeadProfileEntry{};
    e.layer = 0;
    e.head = 3;
    e.echo_score = 0.2;
    e.induction_score = 0.1;
    profile.heads.push_back(e);

    PatternRoutingOptions opts;
    opts.k_landmark = 32;
    opts.k_local = 48;
    opts.k_normal = 96;
    opts.echo_threshold = 0.5;
    opts.induction_threshold = 0.5;
    std::vector<int32_t> matrix;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_pattern_matrix(profile, opts, &matrix,
                                                               &reason));
    CHECK_EQ(matrix.size(), static_cast<size_t>(4));
    // Retrieval wins over every score; echo beats induction.
    CHECK_EQ(matrix[0], -1);
    CHECK_EQ(matrix[1], 32);
    CHECK_EQ(matrix[2], 48);
    CHECK_EQ(matrix[3], 96);
    CHECK(!reason.empty());
    CHECK(reason.find("retrieval=1") != std::string::npos);
    CHECK(reason.find("landmark=1") != std::string::npos);
    CHECK(reason.find("local=1") != std::string::npos);
    CHECK(reason.find("normal=1") != std::string::npos);

    // Boundary: a score exactly at the threshold classifies into the class.
    HeadProfile boundary = profile;
    boundary.heads[1].echo_score = 0.5;
    boundary.heads[2].induction_score = 0.5;
    CHECK_OK(AttentionSparsityController::build_pattern_matrix(boundary, opts, &matrix,
                                                               &reason));
    CHECK_EQ(matrix[1], 32);
    CHECK_EQ(matrix[2], 48);

    // A head above both thresholds is a landmark (echo subsumes copying).
    HeadProfile both = profile;
    both.heads[3].echo_score = 0.9;
    both.heads[3].induction_score = 0.9;
    CHECK_OK(AttentionSparsityController::build_pattern_matrix(both, opts, &matrix,
                                                               &reason));
    CHECK_EQ(matrix[3], 32);
}

// AR004 FR-4: pattern routing rejects null outputs, zero budgets and a
// mismatched profile geometry.
TEST_CASE(pattern_matrix_rejects_bad_inputs) {
    HeadProfile profile = make_profile(2, 2, 2);
    PatternRoutingOptions opts;
    std::vector<int32_t> matrix;
    std::string reason;
    CHECK_ERR(AttentionSparsityController::build_pattern_matrix(profile, opts, nullptr,
                                                                &reason),
              ErrorCode::InvalidArgument);
    auto zero_landmark = opts;
    zero_landmark.k_landmark = 0;
    CHECK_ERR(AttentionSparsityController::build_pattern_matrix(profile, zero_landmark,
                                                                &matrix, &reason),
              ErrorCode::InvalidArgument);
    auto zero_normal = opts;
    zero_normal.k_normal = 0;
    CHECK_ERR(AttentionSparsityController::build_pattern_matrix(profile, zero_normal,
                                                                &matrix, &reason),
              ErrorCode::InvalidArgument);
    HeadProfile broken = profile;
    broken.heads.pop_back();
    CHECK_ERR(AttentionSparsityController::build_pattern_matrix(broken, opts, &matrix,
                                                                &reason),
              ErrorCode::InvalidArgument);
}

// AR004 FR-4: pyramid prior interpolates k_min (middle) to k_max (ends),
// symmetric and monotone towards the middle.
TEST_CASE(pyramid_matrix_monotone_and_bounds) {
    PyramidOptions opts;
    opts.k_min = 48;
    opts.k_max = 128;
    std::vector<int32_t> matrix;
    CHECK_OK(AttentionSparsityController::build_pyramid_matrix(5, 3, opts, &matrix));
    CHECK_EQ(matrix.size(), static_cast<size_t>(15));
    // t = |l - 2| / 2 -> k = [128, 88, 48, 88, 128] per layer.
    const int32_t expected[5] = {128, 88, 48, 88, 128};
    for (uint32_t l = 0; l < 5; ++l) {
        for (uint32_t h = 0; h < 3; ++h) {
            CHECK_EQ(matrix[l * 3 + h], expected[l]);
        }
    }

    // Even layer count: symmetric around the half-integer middle.
    CHECK_OK(AttentionSparsityController::build_pyramid_matrix(28, 1, opts, &matrix));
    CHECK_EQ(matrix.size(), static_cast<size_t>(28));
    CHECK_EQ(matrix[0], 128);
    for (uint32_t l = 0; l < 28; ++l) {
        CHECK_EQ(matrix[l], matrix[27 - l]);
        if (l > 0 && l <= 13) {
            CHECK(matrix[l] <= matrix[l - 1]);
        }
    }
    // Middle layers are the sparsest and the ends the densest.
    CHECK(matrix[13] <= matrix[0]);
    CHECK(matrix[13] <= matrix[27]);

    // A single layer is both first and last -> k_max.
    CHECK_OK(AttentionSparsityController::build_pyramid_matrix(1, 2, opts, &matrix));
    CHECK_EQ(matrix.size(), static_cast<size_t>(2));
    CHECK_EQ(matrix[0], 128);
    CHECK_EQ(matrix[1], 128);
}

// AR004 FR-4: pyramid prior rejects invalid geometry and budgets.
TEST_CASE(pyramid_matrix_rejects_bad_inputs) {
    PyramidOptions opts;
    std::vector<int32_t> matrix;
    CHECK_ERR(AttentionSparsityController::build_pyramid_matrix(0, 4, opts, &matrix),
              ErrorCode::InvalidArgument);
    CHECK_ERR(AttentionSparsityController::build_pyramid_matrix(4, 0, opts, &matrix),
              ErrorCode::InvalidArgument);
    auto swapped = opts;
    swapped.k_min = 200;
    CHECK_ERR(AttentionSparsityController::build_pyramid_matrix(4, 4, swapped, &matrix),
              ErrorCode::InvalidArgument);
    auto zero_min = opts;
    zero_min.k_min = 0;
    CHECK_ERR(AttentionSparsityController::build_pyramid_matrix(4, 4, zero_min, &matrix),
              ErrorCode::InvalidArgument);
}

// AR004 FR-4 refactor guard: the profile-free ranked pool must match the
// pool build_head_adaptive produces for the same selection.
TEST_CASE(ranked_pool_matches_head_adaptive) {
    const Selection sel = make_selection(8192, 0.5);
    CHECK(!sel.dense);
    HeadProfile profile = make_profile(2, 2, 3);
    HeadAdaptiveOptions hopts;
    hopts.block_size = 32;
    hopts.top_k_rows = 64;
    std::vector<int32_t> adaptive_pool;
    std::vector<int32_t> matrix;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_head_adaptive(sel, profile, hopts,
                                                              &adaptive_pool, &matrix,
                                                              &reason));
    std::vector<int32_t> ranked;
    size_t ranked_rows = 0;
    CHECK_OK(AttentionSparsityController::build_ranked_pool(sel, 32, &ranked,
                                                            &ranked_rows));
    CHECK_EQ(ranked.size(), adaptive_pool.size());
    CHECK_EQ(ranked_rows, ranked.size());
    for (size_t i = 0; i < ranked.size(); ++i) {
        CHECK_EQ(ranked[i], adaptive_pool[i]);
    }

    // Dense selections produce an empty pool by contract.
    const Selection dense = make_selection(256, 0.5);
    CHECK(dense.dense);
    ranked.clear();
    CHECK_OK(AttentionSparsityController::build_ranked_pool(dense, 32, &ranked));
    CHECK(ranked.empty());
}

// AR004 FR-4 zero-regression guard: both new switches default to off so a
// default-config Plan matches AR003 field for field.
TEST_CASE(fr4_switches_default_off) {
    Config cfg;
    CHECK(!cfg.head_pattern_routing);
    CHECK(!cfg.pyramid_prior);
    CHECK_EQ(cfg.pyramid_k_min, 48u);
    CHECK_EQ(cfg.pyramid_k_max, 128u);
}

// AR004 FR-2: selected-but-not-gathered full blocks become pool blocks.
TEST_CASE(pool_blocks_selected_not_gathered) {
    const Selection sel = make_selection(4096, 0.5);
    CHECK(!sel.dense);
    AttentionSparsityOptions aopts = sparsity_options(8);
    std::vector<int32_t> indices;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(sel, aopts, &indices, &reason));
    CHECK(!indices.empty());

    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string pool_reason;
    CHECK_OK(AttentionSparsityController::build_pool_blocks(
            sel, indices, 32, &starts, &sizes, &max_pos, &pool_reason));
    CHECK(!starts.empty());
    CHECK_EQ(starts.size(), sizes.size());
    CHECK_EQ(starts.size(), max_pos.size());
    CHECK(!pool_reason.empty());

    std::vector<bool> gathered(sel.total_tokens, false);
    for (int32_t p : indices) gathered[static_cast<uint32_t>(p)] = true;

    // Row space walk: verify each pool block against the selection.
    uint32_t row = 0;
    size_t next_pool = 0;
    const uint32_t block_size = 32;
    const uint32_t n_blocks =
            (sel.total_tokens + block_size - 1) / block_size;
    std::vector<uint32_t> count_of(n_blocks, 0);
    for (uint32_t t : sel.token_indices) ++count_of[t / block_size];
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (count_of[b] == 0) continue;
        const bool full = count_of[b] == block_size;
        bool is_gathered = false;
        if (full) {
            for (uint32_t j = 0; j < block_size; ++j) {
                if (gathered[b * block_size + j]) { is_gathered = true; break; }
            }
        }
        if (full && row % block_size == 0 && !is_gathered) {
            CHECK(next_pool < starts.size());
            if (next_pool < starts.size()) {
                CHECK_EQ(starts[next_pool], static_cast<int32_t>(row));
                CHECK_EQ(sizes[next_pool], 32);
                CHECK_EQ(max_pos[next_pool],
                         static_cast<int32_t>(b * block_size + block_size - 1));
                CHECK(starts[next_pool] % 32 == 0);
                CHECK(max_pos[next_pool] % 32 == 31);
                if (next_pool > 0) {
                    CHECK(starts[next_pool] > starts[next_pool - 1]);
                }
            }
            ++next_pool;
        }
        row += count_of[b];
    }
    CHECK_EQ(next_pool, starts.size());
}

// AR004 FR-2: dense selections pool nothing (pooled rows would duplicate
// full attention).
TEST_CASE(pool_blocks_dense_empty) {
    const Selection dense = make_selection(256, 0.5);
    CHECK(dense.dense);
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_pool_blocks(
            dense, {}, 32, &starts, &sizes, &max_pos, &reason));
    CHECK(starts.empty());
    CHECK_EQ(reason, std::string("dense"));
}

// AR004 FR-2: pool_1d geometry is fixed; only block_size 32 is legal.
TEST_CASE(pool_blocks_invalid_block_size) {
    const Selection sel = make_selection(4096, 0.5);
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string reason;
    CHECK_ERR(AttentionSparsityController::build_pool_blocks(
            sel, {}, 16, &starts, &sizes, &max_pos, &reason),
              ErrorCode::InvalidArgument);
    CHECK_ERR(AttentionSparsityController::build_pool_blocks(
            sel, {}, 0, &starts, &sizes, &max_pos, &reason),
              ErrorCode::InvalidArgument);
    // Null outputs are rejected up front.
    CHECK_ERR(AttentionSparsityController::build_pool_blocks(
            sel, {}, 32, nullptr, &sizes, &max_pos, &reason),
              ErrorCode::InvalidArgument);
}

// AR004 FR-2: with nothing gathered, every full aligned block is pooled.
TEST_CASE(pool_blocks_no_gather_pools_all_full) {
    const Selection sel = make_selection(4096, 0.5);
    CHECK(!sel.dense);
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_pool_blocks(
            sel, {}, 32, &starts, &sizes, &max_pos, &reason));
    CHECK(!starts.empty());
    for (size_t i = 0; i < starts.size(); ++i) {
        CHECK(starts[i] % 32 == 0);
        CHECK_EQ(sizes[i], 32);
        CHECK(max_pos[i] % 32 == 31);
    }
}

// AR004 FR-2 zero-regression guard: the switch defaults to off.
TEST_CASE(pool_branches_switch_default_off) {
    Config cfg;
    CHECK(!cfg.pool_branches);
}

// AR004 FR-3: groups partition the row space and every group carries the
// mandatory anchors.
TEST_CASE(qg_planner_groups_partition) {
    const Selection sel = make_selection(4096, 0.5);
    CHECK(!sel.dense);
    const uint32_t n_rows = static_cast<uint32_t>(sel.token_indices.size());
    CHECK(n_rows > 128);

    std::vector<double> scores(sel.total_tokens, 0.0);
    constexpr uint32_t needle_pos = 37 * 32 + 5;
    for (uint32_t i = needle_pos & ~31u; i < (needle_pos | 31u); ++i) {
        scores[i] = 1.0;
    }

    QueryGatingOptions opts;
    opts.query_chunk_size = 128;
    opts.query_chunk_max = 16;
    opts.top_k_per_group = 256;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> bounds;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            sel, scores, opts, &groups, &bounds, &reason));
    CHECK_EQ(bounds.front(), 0);
    CHECK_EQ(static_cast<uint32_t>(bounds.back()), n_rows);
    CHECK_EQ(groups.size() + 1, bounds.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        CHECK(bounds[g] < bounds[g + 1]);
        CHECK(!groups[g].empty());
        // Ascending rows inside the row space.
        for (size_t i = 0; i < groups[g].size(); ++i) {
            CHECK(groups[g][i] >= 0);
            CHECK(static_cast<uint32_t>(groups[g][i]) < n_rows);
            if (i > 0) CHECK(groups[g][i] > groups[g][i - 1]);
        }
        // Mandatory anchors: the sink row 0 and (for later groups) the
        // recent row directly preceding the group.
        CHECK_EQ(groups[g].front(), 0);
        if (g > 0) {
            bool has_recent = false;
            for (int32_t r : groups[g]) {
                if (r == bounds[g] - 1) has_recent = true;
            }
            CHECK(has_recent);
        }
    }
    // The top-scored needle block must be gathered by the last group
    // (its first row precedes the group's last query row).
    uint32_t needle_row_begin = n_rows;
    uint32_t needle_row_end = 0;
    for (size_t i = 0; i < sel.token_indices.size(); ++i) {
        const uint32_t pos = sel.token_indices[i];
        if (pos >= (needle_pos & ~31u) && pos <= (needle_pos | 31u)) {
            needle_row_begin = std::min(needle_row_begin, static_cast<uint32_t>(i));
            needle_row_end = std::max(needle_row_end, static_cast<uint32_t>(i) + 1);
        }
    }
    CHECK(needle_row_begin < needle_row_end);
    bool last_group_has_needle = false;
    for (int32_t r : groups.back()) {
        if (r >= static_cast<int32_t>(needle_row_begin) &&
            r < static_cast<int32_t>(needle_row_end)) {
            last_group_has_needle = true;
        }
    }
    CHECK(last_group_has_needle);
}

// AR004 FR-3: score length must match the prompt.
TEST_CASE(qg_planner_invalid_score_length) {
    const Selection sel = make_selection(4096, 0.5);
    QueryGatingOptions opts;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> bounds;
    std::string reason;
    CHECK_ERR(AttentionSparsityController::build_query_gated(
                  sel, {}, opts, &groups, &bounds, &reason),
              ErrorCode::InvalidArgument);
    std::vector<double> short_scores(sel.total_tokens - 1, 0.0);
    CHECK_ERR(AttentionSparsityController::build_query_gated(
                  sel, short_scores, opts, &groups, &bounds, &reason),
              ErrorCode::InvalidArgument);
    // Null outputs and degenerate options are rejected.
    CHECK_ERR(AttentionSparsityController::build_query_gated(
                  sel, std::vector<double>(sel.total_tokens, 0.0), opts,
                  nullptr, &bounds, &reason),
              ErrorCode::InvalidArgument);
    QueryGatingOptions zero_chunk;
    zero_chunk.query_chunk_size = 0;
    CHECK_ERR(AttentionSparsityController::build_query_gated(
                  sel, std::vector<double>(sel.total_tokens, 0.0), zero_chunk,
                  &groups, &bounds, &reason),
              ErrorCode::InvalidArgument);
    QueryGatingOptions zero_max;
    zero_max.query_chunk_max = 0;
    CHECK_ERR(AttentionSparsityController::build_query_gated(
                  sel, std::vector<double>(sel.total_tokens, 0.0), zero_max,
                  &groups, &bounds, &reason),
              ErrorCode::InvalidArgument);
}

// AR004 FR-3: too many groups degrades to a single global set.
TEST_CASE(qg_planner_group_cap) {
    const Selection sel = make_selection(4096, 0.5);
    const uint32_t n_rows = static_cast<uint32_t>(sel.token_indices.size());
    QueryGatingOptions opts;
    opts.query_chunk_size = 64;  // 2048 rows -> 32 groups > cap 16
    opts.query_chunk_max = 16;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> bounds;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            sel, std::vector<double>(sel.total_tokens, 0.0), opts,
            &groups, &bounds, &reason));
    CHECK_EQ(groups.size(), 1u);
    CHECK_EQ(bounds.size(), 2u);
    CHECK_EQ(bounds.front(), 0);
    CHECK_EQ(static_cast<uint32_t>(bounds.back()), n_rows);
    CHECK(reason.find("capped") != std::string::npos);
}

// AR004 FR-3: dense selections gate nothing.
TEST_CASE(qg_planner_dense_empty) {
    const Selection dense = make_selection(256, 0.5);
    CHECK(dense.dense);
    QueryGatingOptions opts;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> bounds;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            dense, {}, opts, &groups, &bounds, &reason));
    CHECK(groups.empty());
    CHECK(bounds.empty());
    CHECK_EQ(reason, std::string("dense"));
}

// AR004 FR-3 zero-regression guard: the switch defaults to off.
TEST_CASE(qg_switch_default_off) {
    Config cfg;
    CHECK(!cfg.query_gating);
    CHECK_EQ(cfg.query_chunk_size, 128u);
    CHECK_EQ(cfg.query_chunk_max, 16u);
}
