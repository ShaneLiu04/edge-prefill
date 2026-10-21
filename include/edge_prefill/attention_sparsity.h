#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

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

    // Validates gather indices before execution: strictly ascending, within
    // [0, selection.total_tokens), members of selection.token_indices, and
    // containing the final prompt token (first-token logits need
    // self-attention). Returns InvalidArgument with the violated rule.
    static Status validate_indices(const std::vector<int32_t>& indices,
                                   const Selection& selection);

    // Per-chunk adjustment for chunked prefill: when a chunk executes, the KV
    // cache only holds positions <= chunk_end_pos, so gathering anything
    // beyond that would read unwritten cells. Fills allowed with the
    // ascending prefix of indices that is safe for this chunk. An empty
    // result is valid: the caller must disable gathering for that chunk
    // (dense attention over the written rows is causally safe).
    static void chunk_indices(const std::vector<int32_t>& indices,
                              llama_pos chunk_end_pos,
                              std::vector<int32_t>* allowed);
};

} // namespace edge_prefill
