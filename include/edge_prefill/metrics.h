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
