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
                        const std::vector<int32_t>& attn_indices,
                        const Options& opts,
                        PrefillStats* stats);

    LlamaBackend* backend_;
    PrefixCache* cache_;
};

} // namespace edge_prefill
