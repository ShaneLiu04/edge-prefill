#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/draft_scorer.h"
#include "edge_prefill/embedding_scorer.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

// AR005 FR-1: broadcasts block-level scores to per-token scores over the
// original prompt blocking — token i scores block (i / block_size). A
// trailing partial block consumes its own (whole) block score. The block
// score count must exactly match the blocking implied by n_tokens; a
// null output, zero block size, empty scores, zero tokens, or a
// mismatched count is InvalidArgument (no silent truncation).
Status broadcast_block_scores(const std::vector<double>& block_scores,
                              uint32_t block_size,
                              size_t n_tokens,
                              std::vector<double>* per_token_scores);

// AR005 FR-3: DraftScorer adapter over the standalone embedding engine,
// so BlockSelector consumes semantic scores with zero changes. Block text
// is the direct concatenation of token_pieces over each original block
// interval; the query and every block are encoded by EmbeddingScorer and
// scored by cosine similarity, then broadcast to per-token scores.
//
// The constructor only stores pointers (it cannot report a Status); every
// failure mode — null/unloaded scorer, null pieces, zero block size,
// empty tokens/query, null output — surfaces from score() as
// InvalidArgument. Encoding failures propagate unchanged.
class EmbeddingSelectionScorer : public DraftScorer {
public:
    EmbeddingSelectionScorer(EmbeddingScorer* scorer,
                             const std::vector<std::string>* token_pieces,
                             uint32_t block_size);

    Status score(const std::vector<llama_token>& tokens,
                 const std::string& query,
                 std::vector<double>* per_token_scores) override;

    const char* source() const override { return "embedding"; }

private:
    // Non-const: encoding goes through llama_decode, which mutates the
    // scorer's llama context (batching state); logical results stay
    // deterministic for fixed inputs.
    EmbeddingScorer* scorer_;
    const std::vector<std::string>* token_pieces_;
    uint32_t block_size_;
};

} // namespace edge_prefill
