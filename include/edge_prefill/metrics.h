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
