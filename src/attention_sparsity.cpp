#include "edge_prefill/attention_sparsity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>

namespace edge_prefill {

Status AttentionSparsityController::build(const Selection& selection,
                                          const AttentionSparsityOptions& opts,
                                          std::vector<int32_t>* attention_indices,
                                          std::string* reason) {
    if (!attention_indices) {
        return Status::error(ErrorCode::InvalidArgument, "null attention indices output");
    }
    if (opts.block_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "block size must be > 0");
    }
    if (opts.per_layer && opts.layer_top_k.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "per_layer requires a non-empty layer_top_k");
    }

    attention_indices->clear();
    if (reason) reason->clear();

    if (selection.dense) {
        // Dense prefill computes full attention; gathering is pure overhead.
        if (reason) *reason = "dense";
        return Status::success();
    }

    if (selection.token_indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "non-dense selection has no token indices");
    }
    const uint32_t n_tokens = selection.total_tokens;
    if (n_tokens == 0) {
        return Status::error(ErrorCode::InvalidArgument, "selection has no tokens");
    }
    const uint32_t block_size = opts.block_size;
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    if (selection.signals.size() != n_blocks) {
        return Status::error(ErrorCode::InvalidArgument, "selection signals size mismatch");
    }

    for (uint32_t t : selection.token_indices) {
        if (t >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
    }
    for (size_t i = 1; i < selection.token_indices.size(); ++i) {
        if (selection.token_indices[i] <= selection.token_indices[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "token indices must be strictly ascending");
        }
    }

    if (opts.top_k_blocks == 0) {
        if (reason) *reason = "attn-disabled";
        return Status::success();
    }

    std::set<uint32_t> mandatory(selection.mandatory_blocks.begin(),
                                 selection.mandatory_blocks.end());
    std::set<uint32_t> selected(selection.block_indices.begin(), selection.block_indices.end());
    for (uint32_t b : mandatory) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "mandatory block out of range");
        }
    }
    for (uint32_t b : selected) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "selected block out of range");
        }
    }

    // Rank selected blocks by fused score (descending, stable on index) and
    // keep the top_k_blocks; mandatory blocks are always kept.
    std::vector<uint32_t> order(selected.begin(), selected.end());
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (std::abs(selection.signals[a].fused - selection.signals[b].fused) > 1e-12) {
            return selection.signals[a].fused > selection.signals[b].fused;
        }
        return a < b;
    });

    std::set<uint32_t> kept(mandatory.begin(), mandatory.end());
    for (uint32_t b : order) {
        if (kept.size() >= opts.top_k_blocks) break;
        kept.insert(b);
    }
    // Mandatory blocks may exceed the cap; that is intentional: anchors are
    // never sacrificed to meet top_k_blocks.

    if (kept.size() >= selected.size()) {
        // Gathering every selected block is equivalent to the unpatched
        // path, so skip the patch entirely.
        if (reason) *reason = "attn-top-k-covers-all";
        return Status::success();
    }

    // Only tokens that actually get prefilled can be gathered.
    std::vector<bool> in_kept_block(n_blocks, false);
    for (uint32_t b : kept) in_kept_block[b] = true;
    for (uint32_t t : selection.token_indices) {
        if (in_kept_block[t / block_size]) {
            attention_indices->push_back(static_cast<int32_t>(t));
        }
    }

    // The last prompt token must stay gatherable: it produces the
    // first-token logits. It lives in a mandatory (recent) block, but the
    // selector may have trimmed it below block granularity, so re-check.
    const int32_t last = static_cast<int32_t>(n_tokens - 1);
    if (attention_indices->empty() || attention_indices->back() != last) {
        auto it = std::lower_bound(attention_indices->begin(), attention_indices->end(), last);
        if (it == attention_indices->end() || *it != last) {
            attention_indices->insert(it, last);
        }
    }

    if (reason) {
        *reason = "attn-top-k-blocks:" + std::to_string(kept.size()) + "/" +
                  std::to_string(selected.size());
    }
    return Status::success();
}

Status AttentionSparsityController::validate_indices(const std::vector<int32_t>& indices,
                                                     const Selection& selection) {
    if (indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "attention indices are empty");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        const int32_t idx = indices[i];
        if (idx < 0 || static_cast<uint32_t>(idx) >= selection.total_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "attention index out of range");
        }
        if (i > 0 && idx <= indices[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "attention indices must be strictly ascending");
        }
        if (!std::binary_search(selection.token_indices.begin(),
                                selection.token_indices.end(), static_cast<uint32_t>(idx))) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "attention index is not a selected position");
        }
    }
    // The final prompt token produces the first-token logits and needs its
    // own key row gathered for self-attention.
    if (indices.back() != static_cast<int32_t>(selection.total_tokens - 1)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "attention indices must include the last prompt token");
    }
    return Status::success();
}

void AttentionSparsityController::chunk_indices(const std::vector<int32_t>& indices,
                                                llama_pos chunk_end_pos,
                                                std::vector<int32_t>* allowed) {
    if (!allowed) return;
    allowed->clear();
    if (chunk_end_pos < 0) return;
    // indices are ascending by contract, so the safe set for this chunk is
    // the prefix of positions already written to the KV cache.
    const auto end = std::upper_bound(indices.begin(), indices.end(), chunk_end_pos);
    allowed->assign(indices.begin(), end);
}

Status AttentionSparsityController::build_ranked_pool(const Selection& selection,
                                                      uint32_t block_size,
                                                      std::vector<int32_t>* pool,
                                                      size_t* pool_rows) {
    if (!pool) {
        return Status::error(ErrorCode::InvalidArgument, "null pool output");
    }
    if (block_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "block size must be > 0");
    }
    pool->clear();
    if (pool_rows) *pool_rows = 0;

    if (selection.dense) {
        // Dense prefill gathers nothing; an empty pool is the contract.
        return Status::success();
    }
    if (selection.token_indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "non-dense selection has no token indices");
    }
    const uint32_t n_tokens = selection.total_tokens;
    if (n_tokens == 0) {
        return Status::error(ErrorCode::InvalidArgument, "selection has no tokens");
    }
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    if (selection.signals.size() != n_blocks) {
        return Status::error(ErrorCode::InvalidArgument, "selection signals size mismatch");
    }

    // Map every selected token to its future cache row: the r-th selected
    // token (ascending positions) lands in KV row r.
    std::vector<int32_t> row_of_token(n_tokens, -1);
    for (size_t r = 0; r < selection.token_indices.size(); ++r) {
        const uint32_t t = selection.token_indices[r];
        if (t >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
        row_of_token[t] = static_cast<int32_t>(r);
    }

    // Rank blocks: mandatory first (ascending), then remaining selected
    // blocks by fused signal descending, stable on block index.
    std::vector<uint32_t> mandatory(selection.mandatory_blocks.begin(),
                                    selection.mandatory_blocks.end());
    std::sort(mandatory.begin(), mandatory.end());
    mandatory.erase(std::unique(mandatory.begin(), mandatory.end()), mandatory.end());
    for (uint32_t b : mandatory) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "mandatory block out of range");
        }
    }
    std::vector<uint32_t> rest(selection.block_indices.begin(),
                               selection.block_indices.end());
    std::sort(rest.begin(), rest.end());
    rest.erase(std::unique(rest.begin(), rest.end()), rest.end());
    for (uint32_t b : rest) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "selected block out of range");
        }
    }
    std::vector<uint32_t> ranked(mandatory);
    std::vector<uint32_t> non_mandatory;
    for (uint32_t b : rest) {
        if (!std::binary_search(mandatory.begin(), mandatory.end(), b)) {
            non_mandatory.push_back(b);
        }
    }
    std::sort(non_mandatory.begin(), non_mandatory.end(),
              [&](uint32_t a, uint32_t b) {
                  if (std::abs(selection.signals[a].fused -
                               selection.signals[b].fused) > 1e-12) {
                      return selection.signals[a].fused > selection.signals[b].fused;
                  }
                  return a < b;
              });
    ranked.insert(ranked.end(), non_mandatory.begin(), non_mandatory.end());

    // Emit pool rows block by block in rank order; tokens inside a block
    // keep ascending order. Blocks are bucketed first because the rank
    // order is not ascending by block index.
    std::map<uint32_t, std::vector<int32_t>> rows_of_block;
    for (uint32_t t : selection.token_indices) {
        rows_of_block[t / block_size].push_back(row_of_token[t]);
    }
    pool->reserve(selection.token_indices.size());
    for (uint32_t b : ranked) {
        auto it = rows_of_block.find(b);
        if (it == rows_of_block.end()) continue;
        pool->insert(pool->end(), it->second.begin(), it->second.end());
    }
    if (pool->size() != selection.token_indices.size()) {
        // A selected block with no tokens (fully trimmed by the selector)
        // must not silently shrink the pool contract.
        return Status::error(ErrorCode::InvalidArgument,
                             "selection blocks and token indices disagree");
    }
    if (pool_rows) *pool_rows = pool->size();
    return Status::success();
}

Status AttentionSparsityController::build_head_adaptive(const Selection& selection,
                                                        const HeadProfile& profile,
                                                        const HeadAdaptiveOptions& opts,
                                                        std::vector<int32_t>* pool,
                                                        std::vector<int32_t>* matrix,
                                                        std::string* reason) {
    if (!pool || !matrix) {
        return Status::error(ErrorCode::InvalidArgument, "null pool/matrix output");
    }
    if (opts.block_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "block size must be > 0");
    }
    if (opts.top_k_rows == 0) {
        return Status::error(ErrorCode::InvalidArgument, "head top_k_rows must be > 0");
    }
    if (profile.n_layer <= 0 || profile.n_head <= 0 ||
        profile.heads.size() != static_cast<size_t>(profile.n_layer) *
                                       static_cast<size_t>(profile.n_head)) {
        return Status::error(ErrorCode::InvalidArgument, "head profile geometry mismatch");
    }
    pool->clear();
    matrix->clear();
    if (reason) reason->clear();

    if (selection.dense) {
        // Dense prefill computes full attention; the matrix would be pure
        // overhead. Report and let the caller keep the dense path.
        if (reason) *reason = "dense";
        return Status::success();
    }

    // AR004 FR-4: the pool is profile-free; pattern routing and the
    // pyramid prior reuse the exact same ranking.
    auto pst = build_ranked_pool(selection, opts.block_size, pool);
    if (!pst.ok()) {
        return pst;
    }
    if (pool->empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "non-dense selection produced an empty pool");
    }

    // Matrix: retrieval heads dense (-1), everything else gathers
    // top_k_rows from the pool.
    matrix->reserve(profile.heads.size());
    uint32_t retrieval = 0;
    for (const HeadProfileEntry& entry : profile.heads) {
        if (entry.retrieval) {
            ++retrieval;
            matrix->push_back(-1);
        } else {
            matrix->push_back(static_cast<int32_t>(opts.top_k_rows));
        }
    }

    if (reason) {
        *reason = "head-adaptive: pool=" + std::to_string(pool->size()) +
                  " rows, retrieval=" + std::to_string(retrieval) + "/" +
                  std::to_string(profile.heads.size()) + " heads dense, k=" +
                  std::to_string(opts.top_k_rows);
    }
    return Status::success();
}

Status AttentionSparsityController::build_pattern_matrix(const HeadProfile& profile,
                                                         const PatternRoutingOptions& opts,
                                                         std::vector<int32_t>* matrix,
                                                         std::string* reason) {
    if (!matrix) {
        return Status::error(ErrorCode::InvalidArgument, "null matrix output");
    }
    if (opts.k_landmark == 0 || opts.k_local == 0 || opts.k_normal == 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "pattern k values must be > 0");
    }
    if (profile.n_layer <= 0 || profile.n_head <= 0 ||
        profile.heads.size() != static_cast<size_t>(profile.n_layer) *
                                       static_cast<size_t>(profile.n_head)) {
        return Status::error(ErrorCode::InvalidArgument, "head profile geometry mismatch");
    }
    matrix->clear();
    if (reason) reason->clear();

    matrix->reserve(profile.heads.size());
    uint32_t n_retrieval = 0;
    uint32_t n_landmark = 0;
    uint32_t n_local = 0;
    uint32_t n_normal = 0;
    for (const HeadProfileEntry& entry : profile.heads) {
        // Classification order is part of the contract: the retrieval flag
        // wins over every score, echo beats induction. A head at both
        // thresholds is a landmark (echo subsumes local copying).
        if (entry.retrieval) {
            ++n_retrieval;
            matrix->push_back(-1);
        } else if (entry.echo_score >= opts.echo_threshold) {
            ++n_landmark;
            matrix->push_back(static_cast<int32_t>(opts.k_landmark));
        } else if (entry.induction_score >= opts.induction_threshold) {
            ++n_local;
            matrix->push_back(static_cast<int32_t>(opts.k_local));
        } else {
            ++n_normal;
            matrix->push_back(static_cast<int32_t>(opts.k_normal));
        }
    }

    if (reason) {
        *reason = "pattern-routing: retrieval=" + std::to_string(n_retrieval) +
                  " landmark=" + std::to_string(n_landmark) +
                  " local=" + std::to_string(n_local) +
                  " normal=" + std::to_string(n_normal) +
                  " k(l/loc/norm)=" + std::to_string(opts.k_landmark) + "/" +
                  std::to_string(opts.k_local) + "/" + std::to_string(opts.k_normal);
    }
    return Status::success();
}

Status AttentionSparsityController::build_pyramid_matrix(uint32_t n_layer,
                                                         uint32_t n_head,
                                                         const PyramidOptions& opts,
                                                         std::vector<int32_t>* matrix) {
    if (!matrix) {
        return Status::error(ErrorCode::InvalidArgument, "null matrix output");
    }
    if (n_layer == 0 || n_head == 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "pyramid requires n_layer > 0 and n_head > 0");
    }
    if (opts.k_min == 0 || opts.k_max == 0) {
        return Status::error(ErrorCode::InvalidArgument, "pyramid k values must be > 0");
    }
    if (opts.k_min > opts.k_max) {
        return Status::error(ErrorCode::InvalidArgument,
                             "pyramid k_min must be <= k_max");
    }
    matrix->clear();
    matrix->reserve(static_cast<size_t>(n_layer) * n_head);
    for (uint32_t l = 0; l < n_layer; ++l) {
        // Distance from the middle layer, normalized to [0, 1]: the ends
        // are densest. A single layer is both first and last -> k_max.
        double t = 1.0;
        if (n_layer > 1) {
            t = 2.0 * std::fabs(static_cast<double>(l) -
                                static_cast<double>(n_layer - 1) / 2.0) /
                static_cast<double>(n_layer - 1);
        }
        const double k = static_cast<double>(opts.k_min) +
                         (static_cast<double>(opts.k_max) - opts.k_min) * t;
        const int32_t k_layer = static_cast<int32_t>(std::lround(k));
        for (uint32_t h = 0; h < n_head; ++h) {
            matrix->push_back(k_layer);
        }
    }
    return Status::success();
}

Status AttentionSparsityController::build_pool_blocks(
        const Selection& selection,
        const std::vector<int32_t>& attention_indices,
        uint32_t block_size,
        std::vector<int32_t>* starts,
        std::vector<int32_t>* sizes,
        std::vector<int32_t>* max_pos,
        std::string* reason) {
    if (!starts || !sizes || !max_pos) {
        return Status::error(ErrorCode::InvalidArgument, "null pool block output");
    }
    if (block_size != 32) {
        // The graph pools with ggml_pool_1d(AVG, 32, 32); any other block
        // size would break the pooled-row id arithmetic (start / 32).
        return Status::error(ErrorCode::InvalidArgument,
                             "pool blocks require block_size 32");
    }
    starts->clear();
    sizes->clear();
    max_pos->clear();
    if (reason) reason->clear();

    if (selection.dense) {
        // Dense prefill attends everything; pooled rows would duplicate.
        if (reason) *reason = "dense";
        return Status::success();
    }
    if (selection.token_indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "non-dense selection has no token indices");
    }
    const uint32_t n_tokens = selection.total_tokens;
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    if (selection.signals.size() != n_blocks) {
        return Status::error(ErrorCode::InvalidArgument, "selection signals size mismatch");
    }

    // Gathered positions (prompt indices, Selection semantics).
    std::vector<bool> gathered(n_tokens, false);
    for (int32_t p : attention_indices) {
        if (p < 0 || static_cast<uint32_t>(p) >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "attention index out of range");
        }
        gathered[p] = true;
    }

    // Selected tokens per block, ascending.
    std::vector<uint32_t> count_of(n_blocks, 0);
    for (uint32_t t : selection.token_indices) {
        if (t >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
        ++count_of[t / block_size];
    }

    // Walk blocks in ascending order; the row counter follows the compacted
    // row space. A full block whose rows start 32-aligned and that is not
    // (partially) gathered becomes a pool block; its max position is its
    // last token's prompt index (all 32 tokens are selected, so the block
    // spans prompt positions [b*32, b*32+32) contiguously).
    uint32_t row = 0;
    uint32_t pooled = 0;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (count_of[b] == 0) {
            continue;
        }
        const bool full = count_of[b] == block_size;
        if (full && row % block_size == 0) {
            bool is_gathered = false;
            for (uint32_t j = 0; j < block_size; ++j) {
                if (gathered[b * block_size + j]) {
                    is_gathered = true;
                    break;
                }
            }
            if (!is_gathered) {
                starts->push_back(static_cast<int32_t>(row));
                sizes->push_back(static_cast<int32_t>(block_size));
                max_pos->push_back(static_cast<int32_t>(b * block_size +
                                                        block_size - 1));
                ++pooled;
            }
        }
        row += count_of[b];
    }

    if (reason) {
        *reason = "pool-blocks: " + std::to_string(pooled) + " of " +
                  std::to_string(n_blocks) + " blocks";
    }
    return Status::success();
}

Status AttentionSparsityController::build_query_gated(
        const Selection& selection,
        const std::vector<double>& per_pos_scores,
        const QueryGatingOptions& opts,
        std::vector<std::vector<int32_t>>* group_rows,
        std::vector<int32_t>* q_bounds,
        std::string* reason) {
    if (!group_rows || !q_bounds) {
        return Status::error(ErrorCode::InvalidArgument, "null query gating output");
    }
    group_rows->clear();
    q_bounds->clear();
    if (reason) reason->clear();

    if (opts.query_chunk_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "query_chunk_size is 0");
    }
    if (opts.query_chunk_max == 0) {
        return Status::error(ErrorCode::InvalidArgument, "query_chunk_max is 0");
    }
    if (selection.dense) {
        // Dense prefill has no row space to gate; the caller keeps the
        // dense path.
        if (reason) *reason = "dense";
        return Status::success();
    }
    if (per_pos_scores.size() != selection.total_tokens) {
        return Status::error(ErrorCode::InvalidArgument,
                             "per-position score length " +
                                 std::to_string(per_pos_scores.size()) +
                                 " != total_tokens " +
                                 std::to_string(selection.total_tokens));
    }
    if (selection.token_indices.empty()) {
        if (reason) *reason = "no-rows";
        return Status::success();
    }

    // The gating block granularity is fixed at 32 rows, mirroring the pool
    // branch geometry. token_indices is ascending, so the row space walks
    // prompt order and each block occupies a contiguous row range.
    constexpr uint32_t bs = 32;
    const uint32_t n_tokens = selection.total_tokens;
    const uint32_t n_rows = static_cast<uint32_t>(selection.token_indices.size());
    const uint32_t n_blocks = (n_tokens + bs - 1) / bs;

    std::vector<uint32_t> blk_count(n_blocks, 0);
    std::vector<double> blk_sum(n_blocks, 0.0);
    for (size_t i = 0; i < selection.token_indices.size(); ++i) {
        const uint32_t pos = selection.token_indices[i];
        if (pos >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
        const uint32_t b = pos / bs;
        ++blk_count[b];
        blk_sum[b] += per_pos_scores[pos];
    }
    // Contiguous row range per selected block, ascending.
    std::vector<uint32_t> blk_first_row(n_blocks, 0);
    uint32_t row = 0;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        blk_first_row[b] = row;
        row += blk_count[b];
    }

    const uint32_t qcs = opts.query_chunk_size;
    uint32_t n_groups = (n_rows + qcs - 1) / qcs;
    bool capped = false;
    uint32_t eff_chunk = qcs;
    if (n_groups > opts.query_chunk_max) {
        // Too many groups would explode the graph's per-group loop; a
        // single global group keeps the request running (audited).
        n_groups = 1;
        eff_chunk = n_rows;
        capped = true;
    }

    q_bounds->push_back(0);
    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t lo = std::min(g * eff_chunk, n_rows);
        const uint32_t hi = std::min((g + 1) * eff_chunk, n_rows);
        q_bounds->push_back(static_cast<int32_t>(hi));

        // Mandatory blocks: the sink block (row 0 — guarantees every
        // query sees at least one row, the NaN guard) and the recent
        // block (the rows directly preceding the group).
        std::set<uint32_t> mandatory;
        const uint32_t sink_block = selection.token_indices.front() / bs;
        mandatory.insert(sink_block);
        if (lo > 0) {
            // Block containing row lo-1: the last block whose first row
            // is <= lo-1.
            uint32_t rb = sink_block;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (blk_count[b] == 0) continue;
                if (blk_first_row[b] <= lo - 1) {
                    rb = b;
                } else {
                    break;
                }
            }
            mandatory.insert(rb);
        }

        // Causally eligible blocks: at least one row < hi (rows are in
        // prompt order, so first_row <= hi-1 suffices). Later blocks are
        // fully masked for this group and would waste gather work.
        struct ScoredBlock {
            uint32_t block;
            double score;
        };
        std::vector<ScoredBlock> eligible;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            if (blk_count[b] == 0) continue;
            if (blk_first_row[b] > hi - 1) continue;
            if (mandatory.count(b)) continue;
            const double mean = blk_sum[b] / static_cast<double>(blk_count[b]);
            eligible.push_back({b, mean});
        }
        std::sort(eligible.begin(), eligible.end(),
                  [](const ScoredBlock& a, const ScoredBlock& b2) {
                      if (a.score != b2.score) return a.score > b2.score;
                      return a.block < b2.block;
                  });

        // Top-scoring eligible blocks under the per-group row budget.
        // Block-granular: a block that does not fit is skipped whole.
        std::set<uint32_t> chosen(mandatory);
        uint32_t used = 0;
        for (const ScoredBlock& sb : eligible) {
            if (used + blk_count[sb.block] > opts.top_k_per_group) continue;
            chosen.insert(sb.block);
            used += blk_count[sb.block];
        }

        // Expand blocks to ascending rows.
        std::vector<int32_t> rows;
        for (uint32_t b : chosen) {
            for (uint32_t r = blk_first_row[b]; r < blk_first_row[b] + blk_count[b]; ++r) {
                rows.push_back(static_cast<int32_t>(r));
            }
        }
        // Empty sets cannot happen (the sink block is always mandatory),
        // but the contract promises forced sink+recent; keep the guard.
        if (rows.empty()) {
            rows.push_back(0);
        }
        group_rows->push_back(std::move(rows));
    }

    if (reason) {
        *reason = "query-gating: groups=" + std::to_string(n_groups) +
                  (capped ? " (capped to global set)" : "") +
                  " rows=" + std::to_string(n_rows) +
                  " budget=" + std::to_string(opts.top_k_per_group);
    }
    return Status::success();
}

Status AttentionSparsityController::build_group_spans(
        const Selection& selection,
        const QueryGatingOptions& opts,
        std::vector<std::pair<uint32_t, uint32_t>>* spans,
        std::string* reason) {
    if (!spans || !reason) {
        return Status::error(ErrorCode::InvalidArgument, "null group span output");
    }
    spans->clear();
    reason->clear();

    if (opts.query_chunk_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "query_chunk_size is 0");
    }
    if (opts.query_chunk_max == 0) {
        return Status::error(ErrorCode::InvalidArgument, "query_chunk_max is 0");
    }
    if (selection.dense) {
        *reason = "dense";
        return Status::success();
    }
    if (selection.token_indices.empty()) {
        *reason = "no-rows";
        return Status::success();
    }
    // Out-of-range positions would break the row-to-position mapping below.
    for (uint32_t pos : selection.token_indices) {
        if (pos >= selection.total_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
    }

    // Mirror the v1 partition exactly (build_query_gated): row-space
    // chunks of query_chunk_size rows, degrading to a single global group
    // when the group count would exceed query_chunk_max.
    const uint32_t n_rows = static_cast<uint32_t>(selection.token_indices.size());
    uint32_t n_groups = (n_rows + opts.query_chunk_size - 1) / opts.query_chunk_size;
    uint32_t eff_chunk = opts.query_chunk_size;
    bool capped = false;
    if (n_groups > opts.query_chunk_max) {
        n_groups = 1;
        eff_chunk = n_rows;
        capped = true;
    }

    const std::vector<uint32_t>& toks = selection.token_indices;
    for (uint32_t g = 0; g < n_groups; ++g) {
        const uint32_t lo = std::min(g * eff_chunk, n_rows);
        const uint32_t hi = std::min((g + 1) * eff_chunk, n_rows);
        // Span = original prompt positions covered by the group's rows.
        // token_indices is ascending, so [lo, hi) rows map to the closed
        // token range [toks[lo], toks[hi-1]]; the half-open end is +1.
        spans->emplace_back(toks[lo], toks[hi - 1] + 1);
    }

    *reason = "group-spans: groups=" + std::to_string(n_groups) +
              (capped ? " (capped to global set)" : "") +
              " rows=" + std::to_string(n_rows);
    return Status::success();
}

} // namespace edge_prefill
