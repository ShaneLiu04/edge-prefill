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
    // AR008 FR-1: positional phrase (bigram) match weight, min-max
    // normalised like the other signals. Zero when no query phrase occurs
    // in the block (and always zero when phrase_weight == 0).
    double phrase = 0.0;
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
    // AR007 FR-2/FR-3 audit: how many structural anchors were forced in
    // and how many candidates the MMR pass down-weighted. Both are zero
    // for dense selections.
    uint32_t anchor_blocks = 0;
    uint32_t mmr_penalized = 0;
    // AR008 FR-1 audit + FR-2 profile input: how many blocks matched a
    // discriminative query phrase, and how many discriminative query terms
    // the prompt/question carried in total.
    uint32_t phrase_blocks = 0;
    uint32_t specific_terms = 0;
    // AR009 FR-2 audit: discriminative sub-queries (requested facts) whose
    // evidence the admission actually covered.
    uint32_t coverage_queries = 0;
    // AR010 FR-3 audit: selected blocks that carry evidence, and why the fill
    // stopped ("budget" | "evidence-exhausted" | "no-signal" | "dense").
    uint32_t evidence_blocks = 0;
    std::string stop_reason = "budget";
    // AR011 FR-1/FR-2/FR-3 audit: how the admitted tokens split between the
    // structural anchors (sink/recent, never trimmed) and everything that is
    // evidence-driven, how many tokens the span pass trimmed away, and how many
    // requested facts the reservation could not cover.
    uint32_t structure_tokens = 0;
    uint32_t evidence_tokens = 0;
    uint32_t span_tokens = 0;
    uint32_t uncovered_facts = 0;
    // AR012 FR-4 audit: evidence sentences made whole, and blocks pulled in by
    // a sentence that crossed a block boundary.
    uint32_t sentences_completed = 0;
    uint32_t sentences_extended = 0;
    // AR013: positions actually fed to llama_decode, parallel to token_indices.
    // Empty means identity (token_indices), i.e. AR012 behaviour. With block
    // compaction the gaps between admitted blocks are collapsed so the KV span
    // tracks the admitted tokens instead of the document length.
    std::vector<uint32_t> positions;
    // AR013 audit: max(positions) + 1 (token_indices.back() + 1 when identity).
    uint32_t position_span = 0;
    // AR013 audit: "" (off/identity) | "block" | "prefix-cache" | "fallback".
    std::string compaction_reason;
    // AR008 FR-2 profile input: top-1 lexical score over the lexical sum.
    // A high value means the evidence is concentrated in one block, which
    // is exactly when selection-level sparsity is safe.
    double lexical_concentration = 0.0;
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
    // AR007 FR-2: structural anchors. anchor_ratio caps how much of the
    // token budget the anchors may consume; anchor_stride_scale multiplies
    // the coverage-sample count derived from the budget (>= 4 samples are
    // always placed so no region of the prompt is invisible).
    double anchor_ratio = 0.35;
    double anchor_stride_scale = 1.0;
    // AR007 FR-3: MMR de-duplication. Candidates whose token-set Jaccard
    // similarity to the already selected blocks exceeds mmr_threshold are
    // down-weighted by score * (1 - lambda * excess).
    double mmr_threshold = 0.60;
    double mmr_lambda = 0.50;
    // AR008 FR-1: weight of the positional phrase signal added to the fused
    // block score. 0 disables the feature and restores AR007 behaviour
    // bit-for-bit (no phrase set is built at all).
    double phrase_weight = 0.50;
    // AR009 FR-2: sub-query coverage weight (facility-location gain added to
    // the fill score). 0 disables multi-fact coverage and restores the AR008
    // ranking bit-for-bit; only multi-term queries are affected either way.
    double coverage_weight = 3.0;
    // AR010 FR-1: priors (position/diversity) may only order blocks that carry
    // evidence. 1 = saturating gate (default), 0 = legacy additive fusion.
    double prior_gate = 1.0;
    // AR010 FR-2: allow prior-only blocks to fill the remaining budget after
    // the evidence is exhausted. 0 makes the admission content-sized: the
    // selection stops as soon as no block carries evidence.
    double fill_priors = 1.0;
    // AR011 FR-1: absolute token budget for the structural anchors (sink +
    // recent). 0 keeps the AR010 accounting where every anchor competes for a
    // fraction of the whole budget. >0 turns the budget into
    // "structure (fixed) + evidence (the rest)".
    uint32_t structure_budget = 0;
    // AR011 FR-1: radius of the best-lexical anchor window in blocks.
    // 0 = AR010 (+-2 blocks).
    uint32_t anchor_span_radius = 0;
    // AR011 FR-2: reserve one evidence block per uncovered sub-query before the
    // greedy fill runs. 0 = AR009 facility-location only.
    int coverage_reserve = 0;
    // AR011 FR-3: trim evidence blocks to a token span around the evidence peak,
    // decoupling the admission granularity from the 32-token block granularity
    // used by the attention pool. 0 = whole blocks (AR010).
    int span_admission = 0;
    uint32_t span_min_tokens = 24;
    // AR012 FR-1: admit whole sentences instead of whole blocks. Evidence
    // tokens pull in their sentence; a sentence that spills into an unselected
    // block pulls that block in. 0 = AR011 behaviour (block/span only).
    int sentence_completion = 0;
    uint32_t span_max_tokens = 128;
    // AR012: 1 = also drop tokens outside the admitted sentences; 0 = union
    // only (add the missing sentence continuations, remove nothing).
    int sentence_trim = 0;
    // AR013 FR-2: 0 = identity positions (AR012 behaviour), 1 = collapse the
    // gaps between admitted blocks (intra-block offsets preserved). The KV span
    // then depends on the admitted token count, not on the prompt length.
    int position_compaction = 0;
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
