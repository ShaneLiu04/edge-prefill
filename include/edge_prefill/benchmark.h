#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/config.h"
#include "edge_prefill/metrics.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct BenchmarkConfig {
    std::string model;
    std::string data_dir;
    std::vector<Mode> modes;
    std::vector<double> budgets;
    std::vector<uint32_t> lengths;
    int32_t repeats = 5;
    uint32_t warmup = 1;
    uint32_t max_new_tokens = 32;
    std::string out_dir;
};

class BenchmarkRunner {
public:
    Status run(const BenchmarkConfig& cfg, std::vector<Sample>* out,
               std::vector<std::string>* warnings);
};

} // namespace edge_prefill
