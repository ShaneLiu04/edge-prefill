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
    double coverage_weight = 3.0;
    double prior_gate = 1.0;
    double fill_priors = 1.0;
    // AR011: structure-bounded admission (0 = AR010 accounting exactly).
    uint32_t structure_budget = 0;
    uint32_t anchor_span_radius = 0;
    int coverage_reserve = 0;
    int span_admission = 0;
    uint32_t span_min_tokens = 24;
    int sentence_completion = 0;
    uint32_t span_max_tokens = 128;
    int sentence_trim = 0;
    // AR013 FR-2: compact KV positions (0 = AR012 identity positions).
    int position_compaction = 0;
    // AR014 FR-1: reuse the KV prefix of a conversation across turns. Only
    // datasets whose index carries a `session` column are affected.
    int session_reuse = 0;
    // AR014: cap on the up-front session allocation (0 = no cap).
    uint32_t session_ctx_max = 4096;
    // AR015: content-addressed KV snapshots (empty = disabled).
    std::string snapshot_dir;
    int snapshot_use = 1;
    uint64_t snapshot_max_bytes = 512ull * 1024 * 1024;
    // AR013 FR-1: explicit context size. 0 = grow on demand to the position
    // span the case writes; a value that is too small fails the case instead of
    // silently truncating.
    uint32_t n_ctx = 0;
    // AR012 FR-2: when set,每 sample writes <case>_selected.txt (the admitted
    // prompt as text) and <case>_selection.json next to the raw CSV, so any
    // measured cell can be inspected without re-running the model.
    std::string dump_selection_dir;
    std::string portfolio_table;
    // GPU offload: layers above this index move to the GPU backend (99 =
    // full offload). 0 keeps everything on CPU.
    int32_t n_gpu_layers = 0;
    // AR016 FR-1: execution plan plumbing. 0 means auto (hardware_concurrency
    // for threads, backend default for ubatch). Requesting GPU layers and
    // observing zero offloaded layers is an error unless allow_cpu_fallback.
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    uint32_t n_ubatch = 0;
    bool flash_attn = false;
    bool allow_cpu_fallback = false;
    const char* plan_source = "default";
    // AR016 FR-2/FR-3: fitted cost model + latency target for the budget.
    double budget_target_ms = 0.0;
    double cost_a_ms = -1.0;
    double cost_b_ms_per_token = -1.0;
    double cost_r2 = 0.0;
    uint32_t cost_n = 0;
    // AR017: decode several sequences per forward pass (continuous batching).
    // 1 = the serial path (bit-identical to AR016). Values > 1 need a context
    // with n_seq_max >= decode_batch_size and enough KV cells for every
    // in-flight sequence; the runner sizes it up front and flushes the batch
    // whenever a case would not fit.
    uint32_t decode_batch_size = 1;
    // Total KV cells for the batch context (0 = decode_batch_size * 1536,
    // clamped to the model's training context).
    uint32_t decode_batch_ctx = 0;
    // Largest admitted span (positions) that may join a batch. Longer cases run
    // serially: in a multi-sequence context the unified KV cache reserves a
    // ubatch window per stream, and spans above ~1K were observed to collide
    // with that reservation. 0 = no limit beyond the per-sequence share.
    uint32_t decode_batch_max_span = 1024;
    // AR018: prefill/decode separation. The prefill runs on its own backend
    // (its own model instance, so it can sit on a different device) in a worker
    // thread; the finished sequence is handed to the decode backend as KV state
    // and decoded there, batched. 0 keeps the single-context path (AR017).
    bool pipeline_mode = false;
    // GPU layers for the prefill backend (-1 = same as n_gpu_layers). The
    // decode backend keeps n_gpu_layers.
    int32_t pipeline_prefill_gpu_layers = -1;
    // Threads for the prefill backend (0 = n_threads).
    int32_t pipeline_prefill_threads = 0;
    // How many cases may run ahead of the decode batch (0 = 2 * decode_batch_size).
    // This is what buys the overlap: the worker keeps prefilling while the
    // decoder works through the batch already in flight.
    uint32_t pipeline_ahead = 0;
    std::string out_dir;
};

class BenchmarkRunner {
public:
    Status run(const BenchmarkConfig& cfg, std::vector<Sample>* out,
               std::vector<std::string>* warnings);
};

} // namespace edge_prefill
