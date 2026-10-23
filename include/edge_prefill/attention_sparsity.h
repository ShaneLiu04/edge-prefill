#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/head_profile.h"
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

struct HeadAdaptiveOptions {
    // Must match the block size used for selection.
    uint32_t block_size = 32;
    // Rows (tokens) each non-retrieval head gathers from the ranked pool.
    // Retrieval heads stay dense (-1 in the matrix). 0 is invalid here; the
    // caller decides whether head-adaptive prefill runs at all.
    uint32_t top_k_rows = 96;
};

// AR004 FR-4: MInference-style pattern routing. Each head is classified
// from its probe profile into one of four patterns, and the pattern maps
// to a per-head gather budget:
//   retrieval -> dense (-1; the profile flag wins over every score)
//   landmark  -> k_landmark  (echo-dominant; a small budget still keeps
//                             the mandatory pool prefix: sink/recent
//                             anchors, because the pool is ranked
//                             mandatory-first)
//   local     -> k_local     (induction-dominant; slightly wider)
//   normal    -> k_normal    (everything else; typically the configured
//                             global head budget)
struct PatternRoutingOptions {
    uint32_t k_landmark = 32;
    uint32_t k_local = 48;
    uint32_t k_normal = 96;
    double echo_threshold = 0.5;
    double induction_threshold = 0.5;
};

// AR004 FR-4: PyramidKV-style layer prior. Layer l interpolates from
// k_min at the middle layer (sparsest) to k_max at the first/last layer
// (densest); every head within a layer shares the same budget.
struct PyramidOptions {
    uint32_t k_min = 48;
    uint32_t k_max = 128;
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

    // AR003 head-adaptive prefill: builds the ranked row pool and the
    // layer x head top-k matrix for the per-head gather patch.
    //
    // Pool: cache rows (positions in the ascending active-token list) ordered
    // by importance — mandatory blocks first (sink/recent keep local
    // coherence), then the remaining selected blocks by fused signal
    // descending. The graph filters pool rows to the live cache width per
    // chunk, so the full pool is programmed once.
    //
    // Matrix: retrieval heads get -1 (dense); every other head gets
    // opts.top_k_rows. The binary classification yields exactly two distinct
    // matrix values per layer, satisfying the graph's <=4-group limit.
    static Status build_head_adaptive(const Selection& selection,
                                      const HeadProfile& profile,
                                      const HeadAdaptiveOptions& opts,
                                      std::vector<int32_t>* pool,
                                      std::vector<int32_t>* matrix,
                                      std::string* reason);

    // AR004 FR-4: profile-free ranked pool. Same ranking contract as
    // build_head_adaptive (mandatory blocks ascending, then remaining
    // selected blocks by fused signal descending, tokens ascending inside
    // a block). This is the pool source for pattern routing and the
    // pyramid prior, which replace only the matrix.
    static Status build_ranked_pool(const Selection& selection,
                                    uint32_t block_size,
                                    std::vector<int32_t>* pool,
                                    size_t* pool_rows = nullptr);

    // AR004 FR-4: builds a layer x head top-k matrix from pattern
    // classification of the profile. Classification order: retrieval
    // (profile flag) -> -1; echo_score >= echo_threshold -> k_landmark;
    // induction_score >= induction_threshold -> k_local; otherwise
    // k_normal. `reason` records the per-class head counts for audit.
    static Status build_pattern_matrix(const HeadProfile& profile,
                                       const PatternRoutingOptions& opts,
                                       std::vector<int32_t>* matrix,
                                       std::string* reason);

    // AR004 FR-4: builds a layer x head top-k matrix from the pyramid
    // layer prior, no profile required. k(l) = round(k_min +
    // (k_max - k_min) * 2*|l - (L-1)/2| / (L-1)); a single layer gets
    // k_max (it is both first and last). k_min == 0 or k_min > k_max ->
    // InvalidArgument.
    static Status build_pyramid_matrix(uint32_t n_layer, uint32_t n_head,
                                       const PyramidOptions& opts,
                                       std::vector<int32_t>* matrix);

    // AR004 FR-2: attention-level compressed branches. Selected blocks
    // that are fully present in the KV row space but not gathered become
    // pool blocks: the graph pools their 32 rows (mean) into one appended
    // KV row. A block is poolable only when all its tokens are selected
    // AND its first token lands on a 32-aligned row (pool_1d geometry:
    // starts must be multiples of 32 in the compacted row space, so a
    // partially selected block permanently breaks alignment for the
    // blocks after it). attention_indices are the gathered prompt
    // positions (Selection semantics); a block overlapping them is
    // already gathered and is not pooled. Dense selections and block_size
    // != 32 yield an empty set (dense) / InvalidArgument respectively.
    static Status build_pool_blocks(const Selection& selection,
                                    const std::vector<int32_t>& attention_indices,
                                    uint32_t block_size,
                                    std::vector<int32_t>* starts,
                                    std::vector<int32_t>* sizes,
                                    std::vector<int32_t>* max_pos,
                                    std::string* reason);
};

} // namespace edge_prefill
