#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/status.h"

namespace edge_prefill {

// Semantic relevance scorer used by BlockSelector to fuse draft-model
// signal into block selection. Implementations must return one finite
// score per prompt token; scores may be raw similarities (the selector
// normalizes across blocks).
class DraftScorer {
public:
    virtual ~DraftScorer() = default;
    virtual Status score(const std::vector<llama_token>& tokens,
                         const std::string& query,
                         std::vector<double>* per_token_scores) = 0;
};

struct DraftScorerOptions {
    // Aggregation granularity: all tokens in one block share the block
    // score. Must match the selector block size for correct fusion.
    uint32_t block_size = 32;
    // Forward chunk size for the draft prompt pass; bounds the number of
    // embedding outputs held per llama_decode call.
    uint32_t chunk_size = 256;
    // 0 = use hardware default thread count.
    int32_t n_threads = 0;
};

// Real GGUF-backed draft scorer. Loads a small draft model, runs one
// lightweight forward over the prompt plus one over the query, and scores
// each block by cosine similarity between the draft model's final hidden
// state at the block boundary and the query's final hidden state.
//
// Decoder GGUFs expose per-token hidden states through the embedding
// output (logits[i] != 0 positions), which is why the draft context is
// created with embeddings enabled and pooling NONE.
//
// The prompt tokens are main-model token ids; scoring requires the draft
// vocab to cover them (same model family). Otherwise score() returns
// Unsupported and the caller must fall back to lexical-only selection.
class LlamaDraftScorer : public DraftScorer {
public:
    LlamaDraftScorer() = default;
    ~LlamaDraftScorer() override;
    LlamaDraftScorer(const LlamaDraftScorer&) = delete;
    LlamaDraftScorer& operator=(const LlamaDraftScorer&) = delete;

    Status load_model(const std::string& path, const DraftScorerOptions& opts = {});
    Status score(const std::vector<llama_token>& tokens,
                 const std::string& query,
                 std::vector<double>* per_token_scores) override;

    bool loaded() const { return ctx_ != nullptr; }
    int32_t n_vocab() const;
    uint32_t n_ctx() const;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    DraftScorerOptions opts_;
};

} // namespace edge_prefill
