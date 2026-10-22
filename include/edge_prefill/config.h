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
    uint32_t coverage_stride = 8;
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
