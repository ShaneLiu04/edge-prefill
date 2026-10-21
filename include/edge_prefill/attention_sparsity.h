#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/selector.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct AttentionSparsityOptions {
    // Must match the block size used for selection.
    uint32_t block_size = 32;
    // Number of blocks whose tokens the attention layers gather. 0 disables
    // attention-level sparsity (dense attention over the selected tokens).
    uint32_t top_k_blocks = 0;
    // Per-layer top-k (validated here, applied by the backend). When true,
    // layer_top_k must be non-empty; entries are validated against the model
    // layer count by the caller that programs the backend.
    bool per_layer = false;
    std::vector<uint32_t> layer_top_k;
};

// Derives the strictly ascending gather indices for the llama.cpp
// attention top-k patch from a block selection: keep the highest fused
// scoring blocks up to top_k_blocks, always including mandatory blocks
// (sink/recent/best-lexical anchors) so retrieval-critical context is
// never dropped. Indices are positions of tokens that will exist in the
// KV cache, i.e. the intersection with selection.token_indices.
class AttentionSparsityController {
public:
    static Status build(const Selection& selection,
                        const AttentionSparsityOptions& opts,
                        std::vector<int32_t>* attention_indices,
                        std::string* reason);
};

} // namespace edge_prefill
