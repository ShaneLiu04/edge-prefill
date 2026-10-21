#pragma once

#include <cstdint>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

struct Range {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint32_t size() const { return end - begin; }
};

struct ChunkStats {
    uint32_t index = 0;
    uint32_t tokens = 0;
    uint64_t elapsed_us = 0;
    int32_t kv_pos_max = -1;
    // Attention gather rows actually used by this chunk (0 when gathering
    // is disabled for the chunk). Audit counterpart of PrefillStats::attn_top_k.
    uint32_t attn_rows = 0;
};

class ChunkScheduler {
public:
    static Status plan(uint32_t n_tokens, uint32_t chunk_size, uint32_t n_ubatch,
                       std::vector<Range>* out);
};

} // namespace edge_prefill
