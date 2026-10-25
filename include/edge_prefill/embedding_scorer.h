#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/status.h"

namespace edge_prefill {

// AR005: standalone block-encoding embedding engine (bge-small style GGUF).
// The GGUF's own pooling head wins (bge = CLS, generic embedders = MEAN);
// causal decoder GGUFs without a pooling head are rejected. Texts are
// encoded independently (no causal coupling), so any layer can consume
// the vectors: block selection (query vs block) and query-chunk gating
// (group text vs block) share the same engine.
//
// Sequences are encoded in parallel waves of n_parallel() texts; each text
// is truncated to the encoder window (512 tokens) with an audited warning.
// Pure planning layer: no llama.cpp patch involvement.
class EmbeddingScorer {
public:
    EmbeddingScorer() = default;
    ~EmbeddingScorer();
    EmbeddingScorer(const EmbeddingScorer&) = delete;
    EmbeddingScorer& operator=(const EmbeddingScorer&) = delete;

    // Loads an encoder-style embedding GGUF. The model's own pooling head
    // is used (MEAN or CLS); GGUFs that resolve to no pooling (causal
    // decoders) are rejected with Unsupported.
    // n_threads <= 0 uses the hardware default.
    Status load_model(const std::string& path, int32_t n_threads = 0);

    // Encodes texts independently; embeddings row order matches texts.
    // Empty texts, an empty string member, or a null output is
    // InvalidArgument; calling before load_model is InvalidArgument.
    Status encode(const std::vector<std::string>& texts,
                  std::vector<std::vector<double>>* embeddings);

    // G x B cosine matrix between group texts and original prompt block
    // texts. Column b is the original prompt block index (block b spans
    // positions [b*32, (b+1)*32)); the caller builds both text lists with
    // that blocking. All outputs are finite; a degenerate (all-zero)
    // embedding surfaces Internal instead of a fabricated score.
    Status score_group_block(const std::vector<std::string>& group_texts,
                             const std::vector<std::string>& block_texts,
                             std::vector<std::vector<double>>* matrix);

    // Per-block cosine scores between the query and the original prompt
    // block texts. Shared by selection (broadcast to per-token scores)
    // and by the 1D per-position fallback path.
    Status score_query_blocks(const std::string& query,
                              const std::vector<std::string>& block_texts,
                              std::vector<double>* block_scores);

    bool loaded() const { return ctx_ != nullptr; }
    size_t dim() const { return dim_; }
    uint32_t n_parallel() const { return n_parallel_; }

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    size_t dim_ = 0;
    uint32_t n_parallel_ = 8;

    // AR005: the same block texts are scored twice per case (selection's
    // query-block pass and the 2D group-block matrix) and re-used across
    // cases that share one context (multineedle). Encoding is
    // deterministic, so memoizing block vectors is exact reuse, not an
    // approximation. Keyed by a dual-seed FNV fingerprint of the text
    // list; a stale or foreign list misses and re-encodes.
    std::vector<std::vector<double>> block_cache_;
    std::array<uint64_t, 2> block_cache_key_ = {0, 0};

    static void texts_fingerprint(const std::vector<std::string>& texts,
                                  std::array<uint64_t, 2>* out);
    bool block_cache_match(const std::vector<std::string>& block_texts) const;
    void block_cache_store(const std::vector<std::string>& block_texts,
                           const std::vector<std::vector<double>>& rows,
                           size_t block_row_offset);
};

} // namespace edge_prefill
