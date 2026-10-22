#include "edge_prefill/chunk_scheduler.h"

#include <algorithm>

#include "edge_prefill/log.h"

namespace edge_prefill {

Status ChunkScheduler::plan(uint32_t n_tokens, uint32_t chunk_size, uint32_t n_ubatch,
                            std::vector<Range>* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null chunk output");
    if (n_tokens == 0) return Status::error(ErrorCode::InvalidArgument, "n_tokens must be > 0");
    if (n_ubatch == 0) return Status::error(ErrorCode::InvalidArgument, "n_ubatch must be > 0");

    out->clear();

    // An explicit chunk at least as large as the input means "no chunking":
    // one range covering everything. llama_decode then splits internally at
    // n_ubatch (split_simple), which is only safe for paths whose gather
    // indices are not scoped to a scheduler chunk (the sparse attention
    // path pre-clamps its chunk to n_ubatch for exactly that reason).
    if (chunk_size != 0 && chunk_size >= n_tokens) {
        out->push_back(Range{0, n_tokens});
        return Status::success();
    }

    // 0 = auto: align to n_ubatch so every chunk is a single llama_decode
    // ubatch (AR004 FR-1; the old 512 fallback re-split large chunks and
    // cost 2.5x TTFT on the 4K benchmark).
    uint32_t effective = chunk_size == 0 ? n_ubatch : chunk_size;
    if (effective > n_ubatch) {
        log_warn("chunk_size " + std::to_string(effective) +
                 " exceeds n_ubatch " + std::to_string(n_ubatch) +
                 "; clamping to n_ubatch");
        effective = n_ubatch;
    }
    effective = std::min(effective, n_tokens);
    if (effective == 0) effective = 1;

    for (uint32_t begin = 0; begin < n_tokens; begin += effective) {
        const uint32_t end = std::min(n_tokens, begin + effective);
        out->push_back(Range{begin, end});
    }
    return Status::success();
}

} // namespace edge_prefill
