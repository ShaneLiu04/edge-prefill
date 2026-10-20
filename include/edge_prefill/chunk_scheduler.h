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
};

class ChunkScheduler {
public:
    static Status plan(uint32_t n_tokens, uint32_t chunk_size, uint32_t n_ubatch,
                       std::vector<Range>* out);
};

} // namespace edge_prefill
