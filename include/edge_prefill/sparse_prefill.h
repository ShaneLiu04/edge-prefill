#pragma once

#include <cstdint>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/prefix_cache.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

class SparsePrefiller {
public:
    struct Options {
        // 0 = auto: align to n_ubatch (AR004 FR-1).
        uint32_t chunk_size = 0;
        bool enable_prefix_cache = true;
        bool enable_chunking = true;
        bool enable_ffn_skip = true;
        // AR014: keep the sequence and continue writing at the caller's
        // positions instead of resetting it. The caller passes only the delta
        // (new evidence + this turn's question) and positions that follow the
        // previous turn, so the KV prefix built by earlier turns is reused
        // verbatim. Off = the AR013 behaviour (every prefill starts at 0).
        bool append = false;
        // AR017: which KV sequence this prefill writes. Batched decoding keeps
        // one sequence per in-flight case, so a prefill may target seq != 0.
        int32_t seq_id = 0;
        uint64_t ffn_skip_mask = 0;
        bool collect_chunk_stats = true;
        // Honor selection.attention_indices with per-chunk gather
        // adjustment. False keeps the block-sparse execution untouched.
        bool enable_attn_sparse = true;
        // AR003 head-adaptive prefill: when enabled and both vectors are
        // non-empty, the backend receives the ranked row pool plus the
        // layer x head top-k matrix before prefill and the per-head gather
        // path replaces the scalar top-k path. On any backend rejection the
        // prefill falls back to the scalar path with a warning.
        bool enable_head_adaptive = false;
        std::vector<int32_t> head_pool;    // ranked cache rows
        std::vector<int32_t> head_matrix;  // n_layer * n_head, -1 dense / k sparse
        // AR004 attention-level compressed branches: with the scalar gather
        // active, selected-but-not-gathered full blocks are pooled (mean)
        // into one appended KV row each. Requires attn_top_k gather; off by
        // default; exclusive with the head-adaptive path.
        bool enable_pool_blocks = false;
        // AR004 FR-3 query-chunk gating: when enabled with non-empty
        // qg_group_rows/qg_bounds, every query group gathers its own KV row
        // set (per-chunk live filtering inside) and the graph loops over
        // groups. Replaces the scalar gather; exclusive with the
        // head-adaptive path; requires an un-prefixed row space.
        bool enable_query_gating = false;
        std::vector<std::vector<int32_t>> qg_group_rows;  // full-plan rows per group
        std::vector<int32_t> qg_bounds;                   // full-plan bounds (groups+1)
        // AR006 in-graph bound gating: one page selection per prefill
        // ubatch from real K min/max bounds plus a mandatory local window.
        // Takes precedence over enable_query_gating; requires an un-prefixed
        // row space. Backend rejection degrades to the row-list/scalar path.
        bool enable_qg_bound = false;
        uint32_t qg_sel_pages = 8;
        uint32_t qg_local_rows = 0;
        uint32_t qg_pool_pages = 0;
    };

    SparsePrefiller(LlamaBackend* backend, PrefixCache* cache);

    Status prefill(const std::vector<llama_token>& tokens,
                   const Selection& selection,
                   const Options& opts,
                   PrefillStats* stats);
    Status decode_next(llama_token token, llama_pos pos, std::vector<float>* logits);
    Status save_prefix_state(const std::vector<llama_token>& tokens,
                             const KvSignature& sig);

private:
    Status attn_prefill(const std::vector<llama_token>& tokens,
                        const std::vector<llama_pos>& positions,
                        const std::vector<uint32_t>& origins,
                        const std::vector<int32_t>& attn_indices,
                        const Options& opts,
                        PrefillStats* stats,
                        bool qg_active,
                        bool bound_active);

    LlamaBackend* backend_;
    PrefixCache* cache_;
};

} // namespace edge_prefill
