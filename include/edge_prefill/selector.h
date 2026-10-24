#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/config.h"
#include "edge_prefill/draft_scorer.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct BlockSignal {
    double lexical = 0.0;
    double diversity = 0.0;
    double position = 0.0;
    double draft = 0.0;
    double fused = 0.0;
};

struct Selection {
    std::vector<uint32_t> block_indices;
    std::vector<uint32_t> token_indices;
    std::vector<BlockSignal> signals;
    std::vector<uint32_t> mandatory_blocks;
    // Attention-level top-k gather indices (strictly ascending token
    // positions), populated by AttentionSparsityController. Empty means
    // the dense attention path is used.
    std::vector<int32_t> attention_indices;
    // AR004 FR-3: per-position relevance scores (draft scores when a
    // DraftScorer is active, otherwise lexical idf hits), one entry per
    // prompt token. Empty for dense selections. Consumed by
    // AttentionSparsityController::build_query_gated; not serialized.
    std::vector<double> per_pos_scores;
    // AR005 FR-2: G x B_orig semantic matrix (embedding engine active
    // and query gating armed). Empty = the 1D v1 path. Rows follow the
    // build_group_spans group order; columns index original prompt
    // blocks (block b spans positions [b*32,(b+1)*32)).
    std::vector<std::vector<double>> per_group_block_scores;
    uint32_t attention_top_k = 0;
    uint32_t total_tokens = 0;
    uint32_t selected_tokens = 0;
    bool dense = false;
    // AR005: which engine produced per_pos_scores — "lexical" (idf
    // fallback or draft-disabled) or DraftScorer::source() ("draft",
    // "embedding"). Audited per request; never fabricated.
    std::string score_source = "lexical";
    std::string reason;

    std::string to_json() const;
};

struct SelectionOptions {
    uint32_t block_size = 32;
    double budget = 0.5;
    uint32_t min_dense_tokens = 512;
    uint32_t sink_blocks = 1;
    uint32_t recent_blocks = 2;
    uint32_t coverage_stride = 8;
    SelectorWeights weights;
};

class BlockSelector {
public:
    Status select(const std::vector<llama_token>& tokens,
                  const std::string& query_text,
                  const std::vector<std::string>& token_pieces,
                  const SelectionOptions& opts,
                  DraftScorer* draft,
                  Selection* out) const;
};

// AR005: fusion defaults when the embedding engine is the active score
// source — semantic evidence dominates while lexical/diversity/position
// keep tie-breaking capacity. Applied only when the user has not
// explicitly overridden the weights.
SelectorWeights semantic_priority_weights();

} // namespace edge_prefill
