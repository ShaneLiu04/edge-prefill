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
        uint32_t chunk_size = 512;
        bool enable_prefix_cache = true;
        bool enable_chunking = true;
        bool enable_ffn_skip = true;
        uint64_t ffn_skip_mask = 0;
        bool collect_chunk_stats = true;
        // Honor selection.attention_indices with per-chunk gather
        // adjustment. False keeps the block-sparse execution untouched.
        bool enable_attn_sparse = true;
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
