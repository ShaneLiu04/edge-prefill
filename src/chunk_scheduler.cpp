#include "edge_prefill/chunk_scheduler.h"

#include <algorithm>

namespace edge_prefill {

Status ChunkScheduler::plan(uint32_t n_tokens, uint32_t chunk_size, uint32_t n_ubatch,
                            std::vector<Range>* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null chunk output");
    if (n_tokens == 0) return Status::error(ErrorCode::InvalidArgument, "n_tokens must be > 0");
    if (n_ubatch == 0) return Status::error(ErrorCode::InvalidArgument, "n_ubatch must be > 0");

    uint32_t effective = chunk_size == 0 ? std::min<uint32_t>(512, n_ubatch) : chunk_size;
    effective = std::min(effective, n_ubatch);
    effective = std::min(effective, n_tokens);
    if (effective == 0) effective = 1;

    out->clear();
    for (uint32_t begin = 0; begin < n_tokens; begin += effective) {
        const uint32_t end = std::min(n_tokens, begin + effective);
        out->push_back(Range{begin, end});
    }
    return Status::success();
}

} // namespace edge_prefill
