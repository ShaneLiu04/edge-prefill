#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct Metrics {
    double ttft_ms = 0.0;
    double e2e_ms = 0.0;
    double prefill_tokens_per_s = 0.0;
    uint64_t kv_bytes = 0;
    double quality = 0.0;
    double quality_retention = 0.0;
};

struct Sample {
    std::string task;
    std::string case_id;
    double ttft_ms = 0.0;
    // AR013: fastest timed repeat. Under external host load the median swings
    // by 5x between sessions; the minimum is the least contaminated estimator
    // of what the device can do, and is what the latency gate uses.
    double ttft_min_ms = 0.0;
    double e2e_ms = 0.0;
    double prefill_tokens_per_s = 0.0;
    uint64_t kv_bytes = 0;
    double quality = 0.0;
    bool dense = false;
    uint32_t prompt_tokens = 0;
    uint32_t selected_tokens = 0;
    std::string mode;
    double budget = 1.0;
    // Attention gather rows engaged for this sample (0 = disabled/fallback).
    uint32_t attn_top_k = 0;
    // Timed repeats behind the reported TTFT (median of `repeats` runs).
    uint32_t repeats = 1;
    // AR004 FR-5: raw answer perplexities behind the ppl quality
    // retention (0 for non-ppl tasks). quality = min(1, ppl_dense/ppl_sparse).
    double ppl_dense = 0.0;
    double ppl_sparse = 0.0;
    // AR004 FR-3 audit: query-chunk gating engaged and the planned group
    // count (0 when off / fell back).
    bool query_gating = false;
    uint32_t qg_groups = 0;
    // AR005 FR-5 audit: which engine produced the selection scores
    // ("lexical"/"draft"/"embedding"), the selection-stage scoring time
    // (semantic encoding when the embedding engine is active; added to
    // sparse TTFT for net speedup accounting), the draft forward time
    // when a draft model scored selection, and the query-gating score
    // path ("lexical-1d"/"embedding-2d"; empty when gating is off).
    std::string score_source;
    double semantic_score_ms = 0.0;
    double draft_score_ms = 0.0;
    std::string qg_source;
    // AR007 FR-2/FR-3 audit: admission-budget provenance and the structural
    // anchor / MMR de-duplication counters produced by the selector.
    std::string budget_source;
    uint32_t anchor_blocks = 0;
    uint32_t mmr_penalized = 0;
    // AR008 FR-1/FR-2 audit: blocks that matched a discriminative query
    // phrase, and which execution combination the portfolio router picked.
    uint32_t phrase_blocks = 0;
    uint32_t specific_terms = 0;
    // AR009 FR-2 audit: requested facts (rare query terms) whose evidence the
    // admission actually covered. 0 on single-fact prompts, where coverage is off.
    uint32_t coverage_queries = 0;
    // AR010 FR-3 audit.
    uint32_t evidence_blocks = 0;
    std::string stop_reason = "budget";
    // AR011 FR-1/FR-2/FR-3 audit: structural vs evidence token split, tokens
    // trimmed by the span pass, and requested facts the reservation could not
    // cover.
    uint32_t structure_tokens = 0;
    uint32_t evidence_tokens = 0;
    uint32_t span_tokens = 0;
    uint32_t uncovered_facts = 0;
    uint32_t sentences_completed = 0;
    uint32_t sentences_extended = 0;
    // AR013 FR-1 audit: context cells requested, the allocated context state
    // (llama.cpp KV buffer + logits), the written sequence state, the process
    // RSS, and the compacted position span. These make the memory claim
    // verifiable per case instead of being a property of the whole run.
    uint32_t ctx_cells = 0;
    uint64_t kv_alloc_bytes = 0;
    uint64_t kv_used_bytes = 0;
    uint64_t rss_mb = 0;
    uint32_t position_span = 0;
    std::string compaction_reason;
    // AR014 FR-1 audit: which KV session a turn ran on, how many tokens it
    // reused from earlier turns and how many it actually decoded. A session arm
    // turn with reused_tokens == 0 is the first turn of that conversation.
    std::string session_id;
    uint32_t reused_tokens = 0;
    uint32_t appended_tokens = 0;
    std::string session_reset_reason;
    // AR014 FR-2 audit: length and FNV-1a hash of the conversation log this turn
    // ended on. Both arms must report the same values; a mismatch means the
    // assembly (not the KV reuse) diverged.
    uint32_t session_log_tokens = 0;
    uint64_t session_log_hash = 0;
    // AR015 audit: a warm sample restored its prompt KV from a content-addressed
    // snapshot instead of running the prefill.
    uint32_t snapshot_hit = 0;
    uint64_t snapshot_bytes = 0;
    double snapshot_load_ms = 0.0;
    double snapshot_save_ms = 0.0;
    std::string snapshot_state;   // off / cold / warm / miss:<reason>
    std::string snapshot_key;     // first 16 hex chars of the snapshot name
    std::string portfolio_choice = "none";
    std::string portfolio_reason;
};

class Reporter {
public:
    Status write_sample(const Sample& sample);
    Status write_raw_csv(const std::string& path);
    Status write_report(const std::string& path,
                        const std::vector<Sample>& samples,
                        const EnvironmentInfo& env);
    Status write_environment(const std::string& path, const EnvironmentInfo& env);
    const std::vector<Sample>& samples() const { return samples_; }

private:
    std::vector<Sample> samples_;
};

std::string environment_json(const EnvironmentInfo& env);

} // namespace edge_prefill
