#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

enum class Mode { Latency, Balanced, Quality, Reuse, Dense };
enum class KvPrecision { F16, Q8_0, Q4_0 };

const char* mode_name(Mode mode);
const char* kv_precision_name(KvPrecision precision);

struct SelectorWeights {
    double lexical = 0.45;
    double diversity = 0.15;
    double position = 0.20;
    double draft = 0.20;
};

struct Config {
    std::string model_path;
    std::string draft_model_path;
    // AR005: bge-small style embedding GGUF. Non-empty activates the
    // semantic engine (selection adapter + query-gating 2D matrix); a
    // load failure with an explicit path is a hard error, never a silent
    // downgrade.
    std::string embedding_model;
    std::string prompt;
    std::string prompt_file;
    std::string data_dir;
    std::string results_dir = "benchmarks/results";
    std::string dump_plan_path;
    std::string dump_selection_path;

    uint32_t n_ctx = 0;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    uint32_t n_ubatch = 512;
    int32_t n_gpu_layers = 0;
    uint32_t max_new_tokens = 32;
    uint32_t block_size = 32;
    // 0 = auto: align to n_ubatch so each chunk is a single llama_decode
    // ubatch (AR004 FR-1). Explicit values are clamped to n_ubatch.
    uint32_t chunk_size = 0;
    uint32_t min_dense_tokens = 512;
    uint32_t sink_blocks = 1;
    uint32_t recent_blocks = 2;
    // AR007: 0 means auto (coverage sample count derived from the budget);
    // any value > 1 pins the maximum stride-bucket density.
    uint32_t coverage_stride = 0;
    // AR007 FR-2/FR-3 selection quality knobs.
    double anchor_ratio = 0.35;
    double anchor_stride_scale = 1.0;
    double mmr_threshold = 0.60;
    double mmr_lambda = 0.50;
    // AR008 FR-1: weight of the positional phrase (bigram) admission
    // signal. 0 disables the feature and restores AR007 selection exactly.
    double phrase_weight = 0.50;
    // AR009 FR-2: multi-fact coverage weight (0 = AR008 selection exactly).
    double coverage_weight = 3.0;
    // AR010 FR-1/FR-2: evidence-gated priors and content-sized admission.
    double prior_gate = 1.0;
    double fill_priors = 1.0;
    // AR011: structure-bounded admission (0 = AR010 accounting exactly).
    uint32_t structure_budget = 0;
    uint32_t anchor_span_radius = 0;
    int coverage_reserve = 0;
    int span_admission = 0;
    uint32_t span_min_tokens = 24;
    // AR012: sentence-complete admission (0 = AR011 behaviour).
    int sentence_completion = 0;
    uint32_t span_max_tokens = 128;
    int sentence_trim = 0;
    // AR013 FR-2: 0 = identity positions (AR012), 1 = collapse the gaps between
    // admitted blocks so the KV span tracks the admitted tokens.
    int position_compaction = 0;
    // AR014 FR-1: reuse the KV prefix of a conversation across turns (datasets
    // whose index carries a `session` column). 0 keeps every row independent.
    int session_reuse = 0;
    // Up-front cap on the one-shot session allocation. A conversation is sized
    // once (re-creating the context mid-conversation wipes the KV), so this
    // bounds the up-front KV cost; a turn that outgrows it rebuilds the log.
    uint32_t session_ctx_max = 4096;
    // AR015: content-addressed KV snapshots. Empty disables the feature.
    std::string snapshot_dir;
    // 1 = load a matching snapshot when present (warm arm); 0 = recompute and
    // only write snapshots (cold arm).
    int snapshot_use = 1;
    uint64_t snapshot_max_bytes = 512ull * 1024 * 1024;
    // AR016 FR-3: latency target for the token budget (0 = disabled). When set,
    // the runner solves for the largest budget whose predicted TTFT fits.
    double budget_target_ms = 0.0;
    // AR016 FR-2: fitted prefill cost model (TTFT = a + b * tokens). A negative
    // value means "not fitted"; the SLA path requires a valid model.
    double cost_a_ms = -1.0;
    double cost_b_ms_per_token = -1.0;
    double cost_r2 = 0.0;
    uint32_t cost_n = 0;
    // AR016 FR-1: permit a silent-looking downgrade from a requested GPU
    // placement. Off by default: requesting GPU layers and getting zero is an
    // error, because that is how the AR015 GPU record was mislabelled.
    int allow_cpu_fallback = 0;    std::string dump_selection_dir;
    // AR008 FR-2: measured execution-combination table. Empty keeps the
    // configured (or default) path — routing is opt-in, exactly like the
    // budget curve.
    std::string portfolio_table;
    // AR007 FR-3: measured budget-vs-quality curve. Empty disables the
    // length-adaptive budget (behavior falls back to the mode default).
    std::string budget_curve;
    uint32_t repeats = 5;
    uint32_t warmup = 1;

    double budget = 0.5;
    Mode mode = Mode::Balanced;
    SelectorWeights weights;

    KvPrecision kv_type = KvPrecision::F16;
    KvPrecision prefill_kv_type = KvPrecision::F16;
    KvPrecision decode_kv_type = KvPrecision::F16;

    uint64_t skip_ffn_mask = 0;
    // Attention-level top-k: number of highest-scoring blocks whose tokens
    // the attention layers gather during prefill. 0 disables the gather
    // path (dense attention over the selected tokens).
    uint32_t attn_top_k = 0;
    // Comma-separated per-layer top-k override; validated against the model
    // layer count after model load. Empty string disables per-layer control.
    std::string attn_top_k_layers;
    // AR003: head-level adaptive sparse prefill. Empty profile path or
    // head_adaptive=false keeps the AR002 scalar path byte-identical.
    std::string head_profile_path;
    bool head_adaptive = false;
    double head_retrieval_threshold = 0.10;
    // Rows each non-retrieval head gathers from the ranked pool when
    // head-adaptive prefill is active. 0 disables head-adaptive mode.
    uint32_t head_top_k = 0;
    // AR004 FR-4: MInference-style pattern routing. When true (with a
    // validated profile and head_top_k > 0), the uniform head matrix is
    // replaced by per-class budgets: retrieval -> dense, landmark ->
    // pyramid k_landmark, local -> k_local, normal -> head_top_k.
    // Default off keeps the AR003 matrix byte-identical.
    bool head_pattern_routing = false;
    // AR004 FR-4: PyramidKV-style layer prior when no profile is loaded.
    // Applies only when head_top_k > 0 and the selection is sparse; the
    // pool ranking is profile-free. Default off keeps AR003 behavior.
    bool pyramid_prior = false;
    uint32_t pyramid_k_min = 48;
    uint32_t pyramid_k_max = 128;
    // AR004 FR-2: attention-level compressed branches. With the scalar
    // attention gather active, selected-but-not-gathered full blocks are
    // pooled (mean) into one appended KV row each, so dropped context
    // stays addressable at reduced fidelity. Requires attn_top_k > 0;
    // exclusive with head-adaptive mode. Default off keeps AR003
    // behavior.
    bool pool_branches = false;
    // AR004 FR-3: MoBA-style query-chunk gating. Query rows are
    // partitioned into groups of query_chunk_size; every group gathers
    // its own KV row set (sink+recent mandatory, top-scoring blocks under
    // query_top_k_per_group rows). More than query_chunk_max groups
    // degrades to a single global set. Replaces the scalar gather;
    // exclusive with head-adaptive mode. Default off keeps AR003
    // behavior.
    bool query_gating = false;
    uint32_t query_chunk_size = 128;
    uint32_t query_chunk_max = 16;
    uint32_t query_top_k_per_group = 256;
    // AR006 in-graph query-aware bound gating (Quest-style). One page
    // selection per prefill ubatch from real K min/max upper bounds:
    // qg_sel_pages pages (32 rows each, 0 = local only) ranked by the
    // bound plus a mandatory local window of qg_local_rows rows (0 =
    // ubatch-sized). Takes precedence over query_gating. Default off keeps
    // the AR005 behavior byte-identical.
    bool qg_bound = false;
    uint32_t qg_sel_pages = 8;
    uint32_t qg_local_rows = 0;
    uint32_t qg_pool_pages = 0;
    bool risk_guard = true;
    double risk_sep_threshold = 0.5;
    double risk_cov_threshold = 0.35;
    bool enable_prefix_cache = true;
    uint64_t prefix_cache_bytes = 256ull * 1024 * 1024;
    bool enable_ffn_skip = true;
    bool enable_chunking = true;
    bool deterministic = true;
    bool use_chat_template = false;
    uint64_t seed = 42;

    bool budget_overridden = false;
    bool skip_ffn_overridden = false;
    bool kv_overridden = false;
    // AR005: user-set selector weights suppress the semantic-priority
    // default that applies when the embedding engine is active.
    bool weights_overridden = false;

    std::vector<std::string> warnings;
};

struct Plan {
    std::vector<std::string> passes;
    double budget = 1.0;
    uint32_t block_size = 32;
    // 0 = auto (align to n_ubatch); mirrors Config::chunk_size.
    uint32_t chunk_size = 0;
    uint64_t skip_ffn_mask = 0;
    KvPrecision kv_type = KvPrecision::F16;
    // Attention top-k in blocks (mirrors Config::attn_top_k); 0 in every
    // dense-fallback plan because dense prefill never gathers.
    uint32_t attention_top_k = 0;
    bool dense_fallback = false;
    // AR007 FR-3: where the admission budget came from —
    // "override" (user/conf), "model" (measured curve), "clamp" (memory
    // ceiling), "default" (mode default, no curve).
    std::string budget_source = "default";
    std::vector<std::string> reasons;

    std::string to_json() const;
};

struct PlanningCosts {
    double select_base_ms = 0.5;
    double select_ms_per_token = 0.0001;
    double dense_ms_per_token = 0.0;
};

struct PlannerInput {
    uint32_t n_tokens = 0;
    uint32_t prefix_hit = 0;
    Mode mode = Mode::Balanced;
    bool arch_supports_ffn_skip = false;
    int32_t n_layer = 0;
    uint32_t n_ubatch = 512;
    uint64_t cache_bytes = 0;
    PlanningCosts costs;
};

struct KvSignature {
    std::string model_key;
    uint32_t n_ctx = 0;
    KvPrecision type_k = KvPrecision::F16;
    KvPrecision type_v = KvPrecision::F16;

    bool operator==(const KvSignature& other) const {
        return model_key == other.model_key && n_ctx == other.n_ctx &&
               type_k == other.type_k && type_v == other.type_v;
    }
};

struct PhaseKVPlan {
    KvPrecision prefill_type = KvPrecision::F16;
    KvPrecision decode_type = KvPrecision::F16;
    bool allow_handoff = false;
};

struct EnvironmentInfo {
    std::string os;
    std::string cpu;
    uint32_t physical_cores = 0;
    uint32_t logical_cores = 0;
    std::string compiler;
    std::string cmake_version;
    std::string llama_commit;
    std::string model_sha256;
    std::string data_sha256;
    uint32_t threads = 0;
    std::string kv_type;
    std::string command;
    std::string timestamp;
};

Status parse_config_file(const std::string& path, Config* out);
Status parse_kv_precision(const std::string& text, KvPrecision* out);
Status parse_mode(const std::string& text, Mode* out);
// Parses a comma-separated list of non-negative integers ("4,8,16").
Status parse_u32_list(const std::string& text, std::vector<uint32_t>* out);

} // namespace edge_prefill
