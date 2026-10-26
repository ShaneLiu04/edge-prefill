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
    // Attention-level sparsity: 0 disables; > 0 derives gather indices per
    // non-dense configuration via AttentionSparsityController.
    uint32_t attn_top_k = 0;
    std::string attn_top_k_layers;
    // AR003 head-adaptive prefill: head_adaptive with a loadable profile
    // and head_top_k > 0 activates the per-head gather matrix instead of
    // the scalar top-k path.
    bool head_adaptive = false;
    uint32_t head_top_k = 0;
    std::string head_profile;
    // AR004 FR-4: pattern routing over the profile matrix, and the
    // pyramid layer prior when no profile is loadable. Both default off.
    bool head_pattern_routing = false;
    bool pyramid_prior = false;
    uint32_t pyramid_k_min = 48;
    uint32_t pyramid_k_max = 128;
    // AR004 FR-2: attention-level compressed branches over the scalar
    // gather. Default off.
    bool pool_branches = false;
    // AR004 FR-3: MoBA-style query-chunk gating (replaces the scalar
    // gather; exclusive with head-adaptive). Default off.
    bool query_gating = false;
    uint32_t query_chunk_size = 128;
    uint32_t query_chunk_max = 16;
    uint32_t query_top_k_per_group = 256;
    // AR006 in-graph query-aware bound gating (see Config).
    bool qg_bound = false;
    uint32_t qg_sel_pages = 8;
    uint32_t qg_local_rows = 0;
    uint32_t qg_pool_pages = 0;
    // AR005: embedding GGUF activates the semantic engine for selection
    // and upgrades query gating to the per-group 2D matrix. Explicit
    // configuration never degrades: a load failure fails the run.
    std::string embedding_model;
    // Thread count for the embedding engine (tiny encoder graphs are
    // capped internally); 0 = hardware default.
    int32_t embed_threads = 0;
    // Selector fusion weights and whether the user pinned them (pinned
    // weights suppress the semantic-priority default).
    SelectorWeights weights;
    bool weights_overridden = false;
    // AR007 selection quality knobs (see Config): structural anchors, MMR
    // de-duplication and the measured budget curve. Defaults keep the AR006
    // numerics when budget_curve is empty and anchors/MMR stay at their
    // documented defaults.
    uint32_t sink_blocks = 1;
    uint32_t recent_blocks = 2;
    uint32_t coverage_stride = 0;
    double anchor_ratio = 0.35;
    double anchor_stride_scale = 1.0;
    double mmr_threshold = 0.60;
    double mmr_lambda = 0.50;
    std::string budget_curve;
    // AR008 FR-1/FR-2: positional phrase admission weight and the measured
    // execution-combination table (opt-in; empty keeps the configured path).
    double phrase_weight = 0.50;
    std::string portfolio_table;
    // GPU offload: layers above this index move to the GPU backend (99 =
    // full offload). 0 keeps everything on CPU.
    int32_t n_gpu_layers = 0;
    std::string out_dir;
};

class BenchmarkRunner {
public:
    Status run(const BenchmarkConfig& cfg, std::vector<Sample>* out,
               std::vector<std::string>* warnings);
};

} // namespace edge_prefill
