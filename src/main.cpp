#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/backend.h"
#include "edge_prefill/benchmark.h"
#include "edge_prefill/config.h"
#include "edge_prefill/draft_scorer.h"
#include "edge_prefill/embedding_scorer.h"
#include "edge_prefill/ffn_skip.h"
#include "edge_prefill/head_classifier.h"
#include "edge_prefill/json_writer.h"
#include "edge_prefill/log.h"
#include "edge_prefill/metrics.h"
#include "edge_prefill/planner.h"
#include "edge_prefill/prefix_cache.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/sparse_prefill.h"
#include "edge_prefill/semantic_gate.h"

namespace edge_prefill {
namespace {

struct Args {
    std::string command;
    std::string config;
    std::string model;
    std::string prompt;
    std::string prompt_file;
    std::string data_dir;
    std::string out_dir;
    std::string dump_plan;
    std::string dump_selection;
    std::string mode;
    std::string kv_type;
    std::string skip_ffn;
    std::string draft_model;
    std::string attn_top_k_layers;
    std::string head_profile;
    std::string output;
    std::string calibration_file;
    uint32_t probe_block_len = 32;
    uint32_t probe_repeats = 8;
    uint32_t probe_k = 8;
    uint32_t tokens = 0;
    uint32_t max_new_tokens = 32;
    uint32_t n_ctx = 0;
    uint32_t n_ubatch = 512;
    int32_t threads = 0;
    int32_t n_gpu_layers = 0;
    int32_t repeats = 5;
    uint32_t attn_top_k = 0;
    bool attn_top_k_set = false;
    bool head_adaptive = false;
    bool head_adaptive_set = false;
    bool risk_guard = true;
    bool risk_guard_set = false;
    double head_retrieval_threshold = 0.10;
    bool head_retrieval_threshold_set = false;
    uint32_t head_top_k = 0;
    bool head_top_k_set = false;
    bool head_pattern_routing = false;
    bool head_pattern_routing_set = false;
    bool pyramid_prior = false;
    bool pyramid_prior_set = false;
    uint32_t pyramid_k_min = 48;
    bool pyramid_k_min_set = false;
    uint32_t pyramid_k_max = 128;
    bool pyramid_k_max_set = false;
    bool pool_branches = false;
    bool pool_branches_set = false;
    bool query_gating = false;
    bool query_gating_set = false;
    uint32_t query_chunk_size = 128;
    bool query_chunk_size_set = false;
    uint32_t query_chunk_max = 16;
    bool query_chunk_max_set = false;
    uint32_t query_top_k_per_group = 256;
    bool query_top_k_per_group_set = false;
    double budget = 0.5;
    bool budget_set = false;
    std::string embedding_model;
    double selector_w_lexical = 0.0;
    bool selector_w_lexical_set = false;
    double selector_w_diversity = 0.0;
    bool selector_w_diversity_set = false;
    double selector_w_position = 0.0;
    bool selector_w_position_set = false;
    double selector_w_draft = 0.0;
    bool selector_w_draft_set = false;
    bool chat = false;
    bool json = false;
    bool verbose = false;
};

void usage() {
    std::cout <<
        "EdgePrefill — real GGUF prefill optimization\n\n"
        "Usage:\n"
        "  edge-prefill run    --model M (--prompt P | --prompt-file F) [options]\n"
        "  edge-prefill plan   --tokens N [--mode balanced] [--budget 0.5] [--json]\n"
        "  edge-prefill bench  --config C [--model M] [--out DIR]\n"
        "  edge-prefill eval   --config C [--model M] [--out DIR]\n"
        "  edge-prefill classify-heads --model M --output profile.json [options]\n"
        "  edge-prefill inspect --model M [--json]\n\n"
        "Common options:\n"
        "  --mode latency|balanced|quality|reuse|dense\n"
        "  --budget <0..1>   --kv-type f16|q8_0|q4_0   --skip-ffn <spec>\n"
        "  --draft-model M   --attn-top-k N  --attn-top-k-layers 4,8,16\n"
        "  --embedding-model P  (bge-small style GGUF; activates semantic gating)\n"
        "  --selector-w-lexical F  --selector-w-diversity F\n"
        "  --selector-w-position F  --selector-w-draft F\n"
        "  --head-profile P  --head-adaptive on|off  --head-retrieval-threshold F\n"
        "  --head-pattern-routing on|off  --pyramid-prior on|off\n"
        "  --pyramid-k-min N  --pyramid-k-max N  --pool-branches on|off\n"
        "  --query-gating on|off  --query-chunk-size N  --query-chunk-max N\n"
        "  --query-top-k N\n"
        "  --risk-guard on|off\n"
        "  --probe-block-len N  --probe-repeats N  --probe-k N  --output F\n"
        "  --calibration-file F  (probe segment from real-task text)\n"
        "  --threads N       --n-ctx N                 --n-ubatch N\n"
        "  --max-new-tokens N  --chat   --dump-plan P  --dump-selection P\n"
        "  --verbose\n";
}

bool parse_i64_strict(const std::string& text, int64_t* out) {
    try {
        size_t pos = 0;
        const int64_t value = std::stoll(text, &pos, 10);
        if (pos != text.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

Status parse_args(int argc, char** argv, Args* args) {
    if (argc < 2) return Status::error(ErrorCode::InvalidArgument, "missing command");
    args->command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto next = [&](std::string* out) -> Status {
            if (i + 1 >= argc) {
                return Status::error(ErrorCode::InvalidArgument, "missing value for " + flag);
            }
            *out = argv[++i];
            return Status::success();
        };
        if (flag == "--help" || flag == "-h") {
            usage();
            std::exit(0);
        } else if (flag == "--config") {
            auto st = next(&args->config);
            if (!st.ok()) return st;
        } else if (flag == "--model") {
            auto st = next(&args->model);
            if (!st.ok()) return st;
        } else if (flag == "--prompt") {
            auto st = next(&args->prompt);
            if (!st.ok()) return st;
        } else if (flag == "--prompt-file") {
            auto st = next(&args->prompt_file);
            if (!st.ok()) return st;
        } else if (flag == "--data-dir") {
            auto st = next(&args->data_dir);
            if (!st.ok()) return st;
        } else if (flag == "--out") {
            auto st = next(&args->out_dir);
            if (!st.ok()) return st;
        } else if (flag == "--dump-plan") {
            auto st = next(&args->dump_plan);
            if (!st.ok()) return st;
        } else if (flag == "--dump-selection") {
            auto st = next(&args->dump_selection);
            if (!st.ok()) return st;
        } else if (flag == "--mode") {
            auto st = next(&args->mode);
            if (!st.ok()) return st;
        } else if (flag == "--kv-type") {
            auto st = next(&args->kv_type);
            if (!st.ok()) return st;
        } else if (flag == "--skip-ffn") {
            auto st = next(&args->skip_ffn);
            if (!st.ok()) return st;
        } else if (flag == "--draft-model") {
            auto st = next(&args->draft_model);
            if (!st.ok()) return st;
        } else if (flag == "--embedding-model") {
            auto st = next(&args->embedding_model);
            if (!st.ok()) return st;
        } else if (flag == "--selector-w-lexical" || flag == "--selector-w-diversity" ||
                   flag == "--selector-w-position" || flag == "--selector-w-draft") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            double parsed = 0.0;
            try {
                size_t pos = 0;
                parsed = std::stod(value, &pos);
                if (pos != value.size() || parsed < 0.0) {
                    return Status::error(ErrorCode::InvalidArgument,
                                         "invalid " + flag + ": " + value);
                }
            } catch (...) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid " + flag + ": " + value);
            }
            if (flag == "--selector-w-lexical") {
                args->selector_w_lexical = parsed;
                args->selector_w_lexical_set = true;
            } else if (flag == "--selector-w-diversity") {
                args->selector_w_diversity = parsed;
                args->selector_w_diversity_set = true;
            } else if (flag == "--selector-w-position") {
                args->selector_w_position = parsed;
                args->selector_w_position_set = true;
            } else {
                args->selector_w_draft = parsed;
                args->selector_w_draft_set = true;
            }
        } else if (flag == "--attn-top-k-layers") {
            auto st = next(&args->attn_top_k_layers);
            if (!st.ok()) return st;
            // Reject malformed lists before any model is loaded.
            std::vector<uint32_t> layers;
            auto list_st = parse_u32_list(args->attn_top_k_layers, &layers);
            if (!list_st.ok()) return list_st;
        } else if (flag == "--attn-top-k") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 0 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument, "invalid --attn-top-k: " + value);
            }
            args->attn_top_k = static_cast<uint32_t>(parsed);
            args->attn_top_k_set = true;
        } else if (flag == "--head-profile") {
            auto st = next(&args->head_profile);
            if (!st.ok()) return st;
        } else if (flag == "--head-adaptive") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --head-adaptive: " + value);
            }
            args->head_adaptive = value == "on";
            args->head_adaptive_set = true;
        } else if (flag == "--risk-guard") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --risk-guard: " + value);
            }
            args->risk_guard = value == "on";
            args->risk_guard_set = true;
        } else if (flag == "--head-retrieval-threshold") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            try {
                size_t pos = 0;
                const double parsed = std::stod(value, &pos);
                if (pos != value.size() || parsed < 0.0) {
                    return Status::error(ErrorCode::InvalidArgument,
                                         "invalid --head-retrieval-threshold: " + value);
                }
                args->head_retrieval_threshold = parsed;
                args->head_retrieval_threshold_set = true;
            } catch (...) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --head-retrieval-threshold: " + value);
            }
        } else if (flag == "--head-top-k") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 0 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --head-top-k: " + value);
            }
            args->head_top_k = static_cast<uint32_t>(parsed);
            args->head_top_k_set = true;
        } else if (flag == "--head-pattern-routing") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --head-pattern-routing: " + value);
            }
            args->head_pattern_routing = value == "on";
            args->head_pattern_routing_set = true;
        } else if (flag == "--pyramid-prior") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --pyramid-prior: " + value);
            }
            args->pyramid_prior = value == "on";
            args->pyramid_prior_set = true;
        } else if (flag == "--pyramid-k-min") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --pyramid-k-min: " + value);
            }
            args->pyramid_k_min = static_cast<uint32_t>(parsed);
            args->pyramid_k_min_set = true;
        } else if (flag == "--pyramid-k-max") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --pyramid-k-max: " + value);
            }
            args->pyramid_k_max = static_cast<uint32_t>(parsed);
            args->pyramid_k_max_set = true;
        } else if (flag == "--pool-branches") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --pool-branches: " + value);
            }
            args->pool_branches = value == "on";
            args->pool_branches_set = true;
        } else if (flag == "--query-gating") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            if (value != "on" && value != "off") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --query-gating: " + value);
            }
            args->query_gating = value == "on";
            args->query_gating_set = true;
        } else if (flag == "--query-chunk-size") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --query-chunk-size: " + value);
            }
            args->query_chunk_size = static_cast<uint32_t>(parsed);
            args->query_chunk_size_set = true;
        } else if (flag == "--query-chunk-max") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 16) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --query-chunk-max: " + value);
            }
            args->query_chunk_max = static_cast<uint32_t>(parsed);
            args->query_chunk_max_set = true;
        } else if (flag == "--query-top-k") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 0 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --query-top-k: " + value);
            }
            args->query_top_k_per_group = static_cast<uint32_t>(parsed);
            args->query_top_k_per_group_set = true;
        } else if (flag == "--output") {
            auto st = next(&args->output);
            if (!st.ok()) return st;
        } else if (flag == "--calibration-file") {
            auto st = next(&args->calibration_file);
            if (!st.ok()) return st;
        } else if (flag == "--probe-block-len") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --probe-block-len: " + value);
            }
            args->probe_block_len = static_cast<uint32_t>(parsed);
        } else if (flag == "--probe-repeats") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 2 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --probe-repeats: " + value);
            }
            args->probe_repeats = static_cast<uint32_t>(parsed);
        } else if (flag == "--probe-k") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            int64_t parsed = 0;
            if (!parse_i64_strict(value, &parsed) || parsed < 1 || parsed > 0xffffffffll) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "invalid --probe-k: " + value);
            }
            args->probe_k = static_cast<uint32_t>(parsed);
        } else if (flag == "--tokens") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->tokens = static_cast<uint32_t>(std::stoul(value));
        } else if (flag == "--max-new-tokens") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->max_new_tokens = static_cast<uint32_t>(std::stoul(value));
        } else if (flag == "--n-ctx") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->n_ctx = static_cast<uint32_t>(std::stoul(value));
        } else if (flag == "--n-ubatch") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->n_ubatch = static_cast<uint32_t>(std::stoul(value));
        } else if (flag == "--threads") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->threads = static_cast<int32_t>(std::stol(value));
        } else if (flag == "--ngl" || flag == "--n-gpu-layers") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->n_gpu_layers = static_cast<int32_t>(std::stol(value));
        } else if (flag == "--repeats") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->repeats = static_cast<int32_t>(std::stol(value));
        } else if (flag == "--budget") {
            std::string value;
            auto st = next(&value);
            if (!st.ok()) return st;
            args->budget = std::stod(value);
            args->budget_set = true;
        } else if (flag == "--chat") {
            args->chat = true;
        } else if (flag == "--json") {
            args->json = true;
        } else if (flag == "--verbose") {
            args->verbose = true;
        } else {
            return Status::error(ErrorCode::InvalidArgument, "unknown argument: " + flag);
        }
    }
    return Status::success();
}

std::string read_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string cpu_name() {
    std::ifstream file("/proc/cpuinfo");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) return line.substr(colon + 2);
        }
    }
    return "unknown";
}

std::string sha256_of_file(const std::string& path) {
    const std::string command = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "unavailable";
    char buf[128] = {};
    const char* result = fgets(buf, sizeof(buf), pipe);
    pclose(pipe);
    if (!result) return "unavailable";
    std::string text(buf);
    const auto space = text.find(' ');
    return space == std::string::npos ? text : text.substr(0, space);
}

EnvironmentInfo make_environment(const Config& cfg, const std::string& command_line) {
    EnvironmentInfo env;
    utsname u {};
    if (uname(&u) == 0) env.os = std::string(u.sysname) + " " + u.release;
    env.cpu = cpu_name();
    env.logical_cores = static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    env.physical_cores = env.logical_cores;
    env.compiler =
#if defined(__clang__)
        "clang " __clang_version__;
#elif defined(__GNUC__)
        "gcc " __VERSION__;
#else
        "unknown";
#endif
    env.cmake_version = "see README";
    env.llama_commit = EDGE_PREFILL_LLAMA_COMMIT;
    env.model_sha256 = sha256_of_file(cfg.model_path);
    env.data_sha256 = sha256_of_file(cfg.data_dir + "/index.tsv");
    env.threads = cfg.n_threads > 0 ? static_cast<uint32_t>(cfg.n_threads)
                                    : static_cast<uint32_t>(sysconf(_SC_NPROCESSORS_ONLN));
    env.kv_type = kv_precision_name(cfg.kv_type);
    env.command = command_line;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    env.timestamp = std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(now).count());
    return env;
}

Status apply_common_config(const Args& args, Config* cfg) {
    if (!args.config.empty()) {
        auto st = parse_config_file(args.config, cfg);
        if (!st.ok()) return st;
    }
    if (!args.model.empty()) cfg->model_path = args.model;
    if (!args.prompt.empty()) cfg->prompt = args.prompt;
    if (!args.prompt_file.empty()) {
        cfg->prompt = read_text_file(args.prompt_file);
        if (cfg->prompt.empty()) {
            return Status::error(ErrorCode::Io, "cannot read prompt file");
        }
    }
    if (args.budget_set) {
        cfg->budget = args.budget;
        cfg->budget_overridden = true;
    }
    if (!args.mode.empty()) {
        auto st = parse_mode(args.mode, &cfg->mode);
        if (!st.ok()) return st;
    }
    if (!args.kv_type.empty()) {
        auto st = parse_kv_precision(args.kv_type, &cfg->kv_type);
        if (!st.ok()) return st;
        cfg->prefill_kv_type = cfg->kv_type;
        cfg->decode_kv_type = cfg->kv_type;
        cfg->kv_overridden = true;
    }
    if (!args.skip_ffn.empty()) {
        cfg->enable_ffn_skip = true;
        cfg->skip_ffn_overridden = true;
        cfg->skip_ffn_mask = 0; // parsed after model load (needs n_layer)
    }
    if (!args.draft_model.empty()) cfg->draft_model_path = args.draft_model;
    if (!args.embedding_model.empty()) cfg->embedding_model = args.embedding_model;
    if (args.selector_w_lexical_set) {
        cfg->weights.lexical = args.selector_w_lexical;
        cfg->weights_overridden = true;
    }
    if (args.selector_w_diversity_set) {
        cfg->weights.diversity = args.selector_w_diversity;
        cfg->weights_overridden = true;
    }
    if (args.selector_w_position_set) {
        cfg->weights.position = args.selector_w_position;
        cfg->weights_overridden = true;
    }
    if (args.selector_w_draft_set) {
        cfg->weights.draft = args.selector_w_draft;
        cfg->weights_overridden = true;
    }
    if (args.attn_top_k_set) cfg->attn_top_k = args.attn_top_k;
    if (!args.attn_top_k_layers.empty()) cfg->attn_top_k_layers = args.attn_top_k_layers;
    if (!args.head_profile.empty()) cfg->head_profile_path = args.head_profile;
    if (args.head_adaptive_set) cfg->head_adaptive = args.head_adaptive;
    if (args.head_top_k_set) cfg->head_top_k = args.head_top_k;
    if (args.head_pattern_routing_set) cfg->head_pattern_routing = args.head_pattern_routing;
    if (args.pyramid_prior_set) cfg->pyramid_prior = args.pyramid_prior;
    if (args.pyramid_k_min_set) cfg->pyramid_k_min = args.pyramid_k_min;
    if (args.pyramid_k_max_set) cfg->pyramid_k_max = args.pyramid_k_max;
    if (args.pool_branches_set) cfg->pool_branches = args.pool_branches;
    if (args.query_gating_set) cfg->query_gating = args.query_gating;
    if (args.query_chunk_size_set) cfg->query_chunk_size = args.query_chunk_size;
    if (args.query_chunk_max_set) cfg->query_chunk_max = args.query_chunk_max;
    if (args.query_top_k_per_group_set) {
        cfg->query_top_k_per_group = args.query_top_k_per_group;
    }
    if (cfg->pyramid_k_min > cfg->pyramid_k_max) {
        return Status::error(ErrorCode::InvalidArgument,
                             "pyramid-k-min must be <= pyramid-k-max");
    }
    if (args.risk_guard_set) cfg->risk_guard = args.risk_guard;
    if (args.head_retrieval_threshold_set) {
        cfg->head_retrieval_threshold = args.head_retrieval_threshold;
    }
    if (args.max_new_tokens != 32) cfg->max_new_tokens = args.max_new_tokens;
    if (args.n_ctx != 0) cfg->n_ctx = args.n_ctx;
    if (args.n_ubatch != 512) cfg->n_ubatch = args.n_ubatch;
    if (args.threads != 0) cfg->n_threads = args.threads;
    if (args.chat) cfg->use_chat_template = true;
    return Status::success();
}

std::string chat_prompt(const llama_model* model, const std::string& user) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) return user;
    llama_chat_message message{"user", user.c_str()};
    std::vector<char> buf(user.size() * 2 + 256);
    const int32_t len = llama_chat_apply_template(tmpl, &message, 1, true, buf.data(),
                                                  static_cast<int32_t>(buf.size()));
    if (len <= 0 || static_cast<size_t>(len) > buf.size()) return user;
    return std::string(buf.data(), static_cast<size_t>(len));
}

Status run_command(const Args& args) {
    Config cfg;
    auto st = apply_common_config(args, &cfg);
    if (!st.ok()) return st;
    if (cfg.model_path.empty()) return Status::error(ErrorCode::InvalidArgument, "missing --model");
    if (cfg.prompt.empty()) return Status::error(ErrorCode::InvalidArgument, "missing --prompt");

    st = LlamaBackend::init();
    if (!st.ok()) return st;
    LlamaBackend backend;
    backend.set_n_gpu_layers(cfg.n_gpu_layers);
    st = backend.load_model(cfg.model_path);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    if (!args.skip_ffn.empty()) {
        auto mask_st = FfnSkipController::parse(args.skip_ffn, backend.n_layer(),
                                                &cfg.skip_ffn_mask, nullptr);
        if (!mask_st.ok()) {
            LlamaBackend::shutdown();
            return mask_st;
        }
    }
    if (!cfg.attn_top_k_layers.empty()) {
        std::vector<uint32_t> layers;
        auto layer_st = parse_u32_list(cfg.attn_top_k_layers, &layers);
        if (!layer_st.ok()) {
            LlamaBackend::shutdown();
            return layer_st;
        }
        if (layers.size() != static_cast<size_t>(backend.n_layer())) {
            LlamaBackend::shutdown();
            return Status::error(ErrorCode::InvalidArgument,
                                 "attn-top-k-layers size != n_layer");
        }
    }
    LlamaDraftScorer draft;
    if (!cfg.draft_model_path.empty()) {
        DraftScorerOptions dopts;
        dopts.block_size = cfg.block_size;
        dopts.n_threads = cfg.n_threads;
        auto draft_st = draft.load_model(cfg.draft_model_path, dopts);
        if (!draft_st.ok()) {
            LlamaBackend::shutdown();
            return draft_st;
        }
    }

    // AR005: explicit embedding configuration never degrades — a failed
    // load is a hard error so the semantic path is either real or absent.
    EmbeddingScorer embed;
    if (!cfg.embedding_model.empty()) {
        auto embed_st = embed.load_model(cfg.embedding_model, cfg.n_threads);
        if (!embed_st.ok()) {
            LlamaBackend::shutdown();
            return embed_st;
        }
    }

    std::string prompt = cfg.use_chat_template ? chat_prompt(backend.model(), cfg.prompt)
                                               : cfg.prompt;
    std::vector<llama_token> tokens;
    st = backend.tokenize(prompt, &tokens, true, true);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    if (tokens.empty()) {
        LlamaBackend::shutdown();
        return Status::error(ErrorCode::Tokenize, "empty prompt tokens");
    }
    std::vector<std::string> pieces(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) backend.token_to_piece(tokens[i], &pieces[i]);

    Config run_cfg = cfg;
    if (run_cfg.n_ctx == 0) {
        run_cfg.n_ctx = static_cast<uint32_t>(tokens.size() + cfg.max_new_tokens + 64);
    }
    ContextOptions opts;
    opts.n_ctx = run_cfg.n_ctx;
    opts.n_ubatch = run_cfg.n_ubatch;
    opts.n_threads = run_cfg.n_threads;
    opts.n_threads_batch = run_cfg.n_threads_batch;
    opts.type_k = run_cfg.kv_type;
    opts.type_v = run_cfg.kv_type;
    // GPU offload is a model-level parameter applied at load time (see
    // run_command's set_n_gpu_layers call), not a context option.
    // The gather patch only hooks the non-flash attention layout; disable
    // flash whenever attention sparsity can engage so the CLI never
    // silently degrades to dense attention.
    if (run_cfg.attn_top_k > 0 || run_cfg.head_adaptive) {
        opts.flash_attn = false;
    }
    st = backend.create_context(opts);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }

    PlannerInput pin;
    pin.n_tokens = static_cast<uint32_t>(tokens.size());
    pin.mode = run_cfg.mode;
    pin.arch_supports_ffn_skip = true;
    pin.n_layer = backend.n_layer();
    pin.n_ubatch = run_cfg.n_ubatch;
    PlanningCosts costs;
    costs.dense_ms_per_token = 0.0;
    pin.costs = costs;
    Plan plan;
    st = Planner::make_plan(pin, run_cfg, &plan);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    if (!args.dump_plan.empty()) {
        std::ofstream out(args.dump_plan);
        out << plan.to_json() << '\n';
    }

    SelectionOptions sopts;
    sopts.block_size = run_cfg.block_size;
    sopts.budget = plan.dense_fallback ? 1.0 : plan.budget;
    sopts.min_dense_tokens = run_cfg.min_dense_tokens;
    sopts.sink_blocks = run_cfg.sink_blocks;
    sopts.recent_blocks = run_cfg.recent_blocks;
    sopts.coverage_stride = run_cfg.coverage_stride;
    sopts.weights = run_cfg.weights;
    // AR005: semantic evidence dominates the fusion when the embedding
    // engine is active, unless the user pinned weights or pure latency
    // mode makes the extra encoding cost the wrong trade.
    if (embed.loaded() && run_cfg.mode != Mode::Latency && !run_cfg.weights_overridden) {
        sopts.weights = semantic_priority_weights();
    }
    Selection selection;
    BlockSelector selector;
    // The embedding adapter outranks the draft scorer: the standalone
    // encoder is the AR005 primary signal, the draft model the fallback.
    EmbeddingSelectionScorer embed_adapter(embed.loaded() ? &embed : nullptr,
                                           &pieces, run_cfg.block_size);
    DraftScorer* selection_scorer =
            embed.loaded() ? static_cast<DraftScorer*>(&embed_adapter)
                           : (draft.loaded() ? static_cast<DraftScorer*>(&draft)
                                             : nullptr);
    // AR005 FR-5: selection-stage scoring time (semantic encoding when the
    // embedding engine is active, draft forward otherwise) is accounted
    // separately from the prefill TTFT so speedups stay net-honest.
    const auto sel_begin = std::chrono::steady_clock::now();
    st = selector.select(tokens, run_cfg.prompt, pieces, sopts, selection_scorer,
                         &selection);
    const double select_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - sel_begin)
                                     .count();
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    if (plan.attention_top_k > 0 && !selection.dense) {
        AttentionSparsityOptions aopts;
        aopts.block_size = run_cfg.block_size;
        aopts.top_k_blocks = plan.attention_top_k;
        if (!run_cfg.attn_top_k_layers.empty()) {
            aopts.per_layer = true;
            auto layer_st = parse_u32_list(run_cfg.attn_top_k_layers, &aopts.layer_top_k);
            if (!layer_st.ok()) {
                LlamaBackend::shutdown();
                return layer_st;
            }
        }
        std::vector<int32_t> attn_indices;
        std::string attn_reason;
        auto attn_st = AttentionSparsityController::build(selection, aopts,
                                                          &attn_indices, &attn_reason);
        if (!attn_st.ok()) {
            LlamaBackend::shutdown();
            return attn_st;
        }
        selection.attention_indices = std::move(attn_indices);
        selection.attention_top_k = static_cast<uint32_t>(selection.attention_indices.size());
        log_info("attention sparsity: " + attn_reason);
    }
    if (!args.dump_selection.empty()) {
        std::ofstream out(args.dump_selection);
        out << selection.to_json() << '\n';
    }

    std::unique_ptr<PrefixCache> cache;
    if (run_cfg.enable_prefix_cache) {
        PrefixCache::create(run_cfg.prefix_cache_bytes, backend.fingerprint().to_key(), &cache);
    }
    SparsePrefiller prefilter(&backend, cache.get());
    SparsePrefiller::Options popts;
    // 0 = auto (align to n_ubatch) for both sparse and dense-fallback
    // plans (AR004 FR-1).
    popts.chunk_size = plan.chunk_size;
    popts.enable_prefix_cache = run_cfg.enable_prefix_cache;
    popts.enable_chunking = run_cfg.enable_chunking;
    popts.enable_ffn_skip = run_cfg.enable_ffn_skip;
    popts.ffn_skip_mask = plan.dense_fallback ? 0 : plan.skip_ffn_mask;
    // AR004 compressed branches ride on the scalar gather; the head path
    // is exclusive in the graph (v1), so only enable one.
    popts.enable_pool_blocks = run_cfg.pool_branches && !run_cfg.head_adaptive;

    // AR003/AR004 FR-4: per-head adaptive gather. The ranked pool is
    // profile-free; the matrix comes from the profile (uniform or pattern
    // routed) or, without a profile, from the pyramid layer prior.
    if (run_cfg.head_adaptive && run_cfg.head_top_k > 0 && !selection.dense) {
        HeadProfile head_profile;
        bool profile_ok = false;
        if (!run_cfg.head_profile_path.empty()) {
            auto hst = load_head_profile(run_cfg.head_profile_path, &head_profile);
            if (!hst.ok()) {
                log_warn("head profile load failed (" + hst.message +
                         "); using fallback path");
            } else {
                hst = validate_head_profile(head_profile, backend.n_layer(),
                                            backend.n_head());
                if (!hst.ok()) {
                    log_warn("head profile rejected (" + hst.message +
                             "); using fallback path");
                } else {
                    profile_ok = true;
                }
            }
        }

        if (profile_ok) {
            HeadAdaptiveOptions hopts;
            hopts.block_size = run_cfg.block_size;
            hopts.top_k_rows = run_cfg.head_top_k;
            std::vector<int32_t> pool;
            std::vector<int32_t> matrix;
            std::string head_reason;
            bool built = AttentionSparsityController::build_head_adaptive(
                             selection, head_profile, hopts, &pool, &matrix,
                             &head_reason).ok() &&
                         !pool.empty() && !matrix.empty();
            // AR004 FR-4: pattern routing replaces only the matrix; the
            // pool ranking is shared.
            if (built && run_cfg.head_pattern_routing) {
                PatternRoutingOptions popts_routing;
                popts_routing.k_normal = run_cfg.head_top_k;
                matrix.clear();
                built = AttentionSparsityController::build_pattern_matrix(
                            head_profile, popts_routing, &matrix,
                            &head_reason).ok() &&
                        !matrix.empty();
            }
            if (built) {
                popts.enable_head_adaptive = true;
                popts.head_pool = std::move(pool);
                popts.head_matrix = std::move(matrix);
                log_info("head adaptive: " + head_reason);
            } else {
                log_warn("head-adaptive build failed (" + head_reason +
                         "); using scalar path");
            }
        } else if (run_cfg.pyramid_prior) {
            // AR004 FR-4: no profile — pyramid layer prior over the
            // profile-free ranked pool.
            std::vector<int32_t> pool;
            std::vector<int32_t> matrix;
            PyramidOptions pyopts;
            pyopts.k_min = run_cfg.pyramid_k_min;
            pyopts.k_max = run_cfg.pyramid_k_max;
            if (AttentionSparsityController::build_ranked_pool(
                        selection, run_cfg.block_size, &pool).ok() &&
                !pool.empty() &&
                AttentionSparsityController::build_pyramid_matrix(
                        static_cast<uint32_t>(backend.n_layer()),
                        static_cast<uint32_t>(backend.n_head()),
                        pyopts, &matrix).ok() &&
                !matrix.empty()) {
                popts.enable_head_adaptive = true;
                const size_t pool_size = pool.size();
                popts.head_pool = std::move(pool);
                popts.head_matrix = std::move(matrix);
                log_info("pyramid prior: pool=" + std::to_string(pool_size) +
                         " k(min/max)=" + std::to_string(pyopts.k_min) + "/" +
                         std::to_string(pyopts.k_max));
            } else {
                log_warn("pyramid-prior build failed; using scalar path");
            }
        } else {
            log_warn("head-adaptive requires --head-profile (or --pyramid-prior "
                     "without one); using scalar path");
        }
    }

    // AR004 FR-3 / AR005 FR-2: query-chunk gating. The head matrix path
    // wins (graph-level exclusivity, v1); gating replaces the scalar
    // gather and pool rides only on the scalar gather, so both are off
    // while gating is armed. With the embedding engine active the plan
    // upgrades to the per-group 2D matrix; every degradation step is
    // audited, never silent.
    std::string qg_source;
    if (run_cfg.query_gating && !popts.enable_head_adaptive && !selection.dense) {
        if (!embed.loaded()) {
            log_debug("qg: embedding-disabled (no embedding model configured)");
        }
        if (selection.per_pos_scores.size() != selection.total_tokens) {
            log_debug("qg: lexical-1d-unavailable (per-position scores missing)");
        }
    }
    if (run_cfg.query_gating && !popts.enable_head_adaptive &&
        !selection.dense &&
        selection.per_pos_scores.size() == selection.total_tokens) {
        QueryGatingOptions qg_opts;
        qg_opts.query_chunk_size = run_cfg.query_chunk_size;
        qg_opts.query_chunk_max = run_cfg.query_chunk_max;
        qg_opts.top_k_per_group = run_cfg.query_top_k_per_group;
        if (embed.loaded()) {
            // AR005 FR-2: group texts are the raw contiguous slices of
            // the group token spans (context included), block texts the
            // original 32-token blocks; failure degrades to the 1D path
            // with an audit instead of failing the request.
            std::vector<std::pair<uint32_t, uint32_t>> spans;
            std::string span_reason;
            if (AttentionSparsityController::build_group_spans(
                        selection, qg_opts, &spans, &span_reason).ok() &&
                !spans.empty()) {
                std::vector<std::string> group_texts;
                group_texts.reserve(spans.size());
                for (const auto& span : spans) {
                    std::string text;
                    for (uint32_t i = span.first; i < span.second; ++i) {
                        text += pieces[i];
                    }
                    group_texts.push_back(std::move(text));
                }
                const uint32_t bs = run_cfg.block_size;
                const uint32_t n_prompt_blocks =
                        (selection.total_tokens + bs - 1) / bs;
                std::vector<std::string> block_texts(n_prompt_blocks);
                for (uint32_t b = 0; b < n_prompt_blocks; ++b) {
                    const uint32_t begin = b * bs;
                    const uint32_t end =
                            std::min(selection.total_tokens, begin + bs);
                    for (uint32_t i = begin; i < end; ++i) {
                        block_texts[b] += pieces[i];
                    }
                }
                std::vector<std::vector<double>> matrix;
                auto score_st =
                        embed.score_group_block(group_texts, block_texts, &matrix);
                if (score_st.ok() && matrix.size() == spans.size()) {
                    selection.per_group_block_scores = std::move(matrix);
                } else {
                    log_warn("semantic qg matrix unavailable (" + score_st.message +
                             "); falling back to 1D scores");
                }
            } else {
                log_warn("group spans unavailable (" + span_reason +
                         "); falling back to 1D scores");
            }
        }
        const uint32_t n_prompt_blocks =
                selection.per_group_block_scores.empty()
                        ? 0
                        : (selection.total_tokens + run_cfg.block_size - 1) /
                                  run_cfg.block_size;
        std::string qg_reason;
        auto qst = AttentionSparsityController::build_query_gated(
                selection, selection.per_pos_scores,
                selection.per_group_block_scores, n_prompt_blocks, qg_opts,
                &popts.qg_group_rows, &popts.qg_bounds, &qg_reason);
        if (!qst.ok()) {
            log_warn("query gating plan rejected (" + qst.message +
                     "); using scalar path");
        } else if (popts.qg_group_rows.empty()) {
            log_debug("query gating: none (" + qg_reason + ")");
        } else {
            popts.enable_query_gating = true;
            popts.enable_pool_blocks = false;
            qg_source = selection.per_group_block_scores.empty()
                                ? "lexical-1d"
                                : "embedding-2d";
            log_info(qg_reason);
        }
    }
    PrefillStats stats;
    const auto begin = std::chrono::steady_clock::now();
    st = prefilter.prefill(tokens, selection, popts, &stats);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    const auto first = std::chrono::steady_clock::now();
    const double ttft_ms = std::chrono::duration<double, std::milli>(first - begin).count();

    const float* first_logits = llama_get_logits_ith(backend.context(), -1);
    std::string generated;
    if (first_logits) {
        std::vector<float> logits(first_logits, first_logits + backend.n_vocab());
        llama_token next = static_cast<llama_token>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        for (uint32_t i = 0; i < run_cfg.max_new_tokens; ++i) {
            std::string piece;
            backend.token_to_piece(next, &piece);
            generated += piece;
            if (next == llama_vocab_eos(backend.vocab())) break;
            std::vector<float> step;
            st = prefilter.decode_next(next, static_cast<llama_pos>(tokens.size() + i), &step);
            if (!st.ok()) break;
            next = static_cast<llama_token>(
                std::max_element(step.begin(), step.end()) - step.begin());
        }
    }

    std::cout << generated << '\n';
    std::cerr << "[edge-prefill] prompt=" << stats.prompt_tokens
              << " selected=" << stats.selected_tokens << " chunks=" << stats.chunks
              << " ttft_ms=" << ttft_ms << " kv_bytes=" << stats.kv_bytes
              << " attn_top_k=" << stats.attn_top_k
              << " attn_fallback=" << (stats.attn_fallback ? 1 : 0)
              << " mode=" << mode_name(run_cfg.mode) << " budget=" << sopts.budget
              << " score_source=" << selection.score_source
              << " semantic_score_ms=" << (embed.loaded() ? select_ms : 0.0)
              << " draft_score_ms="
              << (!embed.loaded() && draft.loaded() ? select_ms : 0.0)
              << " qg_source=" << qg_source << '\n';
    LlamaBackend::shutdown();
    return Status::success();
}

Status plan_command(const Args& args) {
    Config cfg;
    if (!args.mode.empty()) {
        auto st = parse_mode(args.mode, &cfg.mode);
        if (!st.ok()) return st;
    }
    if (args.budget_set) {
        cfg.budget = args.budget;
        cfg.budget_overridden = true;
    }
    if (args.attn_top_k_set) cfg.attn_top_k = args.attn_top_k;
    if (!args.attn_top_k_layers.empty()) cfg.attn_top_k_layers = args.attn_top_k_layers;
    PlannerInput in;
    in.n_tokens = args.tokens;
    in.mode = cfg.mode;
    in.n_ubatch = args.n_ubatch;
    in.arch_supports_ffn_skip = true;
    in.n_layer = 28;
    Plan plan;
    auto st = Planner::make_plan(in, cfg, &plan);
    if (!st.ok()) return st;
    std::cout << plan.to_json() << '\n';
    return Status::success();
}

Status bench_command(const Args& args) {
    if (args.config.empty() && args.model.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "missing --config or --model");
    }
    Config cfg;
    if (!args.config.empty()) {
        auto st = parse_config_file(args.config, &cfg);
        if (!st.ok()) return st;
    }
    if (!args.model.empty()) cfg.model_path = args.model;
    if (!args.data_dir.empty()) cfg.data_dir = args.data_dir;
    if (!args.out_dir.empty()) cfg.results_dir = args.out_dir;
    if (cfg.model_path.empty()) return Status::error(ErrorCode::InvalidArgument, "missing model");
    if (cfg.data_dir.empty()) return Status::error(ErrorCode::InvalidArgument, "missing data_dir");
    if (args.repeats > 0) cfg.repeats = static_cast<uint32_t>(args.repeats);

    BenchmarkConfig bench;
    bench.model = cfg.model_path;
    bench.data_dir = cfg.data_dir;
    bench.modes = {Mode::Dense, cfg.mode};
    bench.budgets = {1.0, cfg.budget};
    bench.lengths = {0};
    bench.repeats = static_cast<int32_t>(cfg.repeats);
    bench.warmup = cfg.warmup;
    bench.max_new_tokens = cfg.max_new_tokens;
    bench.attn_top_k = cfg.attn_top_k;
    bench.attn_top_k_layers = cfg.attn_top_k_layers;
    bench.head_adaptive = cfg.head_adaptive;
    bench.head_top_k = cfg.head_top_k;
    bench.head_profile = cfg.head_profile_path;
    bench.head_pattern_routing = cfg.head_pattern_routing;
    bench.pyramid_prior = cfg.pyramid_prior;
    bench.pyramid_k_min = cfg.pyramid_k_min;
    bench.pyramid_k_max = cfg.pyramid_k_max;
    bench.pool_branches = cfg.pool_branches;
    bench.query_gating = cfg.query_gating;
    bench.query_chunk_size = cfg.query_chunk_size;
    bench.query_chunk_max = cfg.query_chunk_max;
    bench.query_top_k_per_group = cfg.query_top_k_per_group;
    bench.embedding_model = cfg.embedding_model;
    bench.weights = cfg.weights;
    bench.weights_overridden = cfg.weights_overridden;
    bench.n_gpu_layers = cfg.n_gpu_layers;
    bench.out_dir = cfg.results_dir;

    BenchmarkRunner runner;
    std::vector<Sample> samples;
    std::vector<std::string> warnings;
    auto st = runner.run(bench, &samples, &warnings);
    if (!st.ok()) return st;
    if (!warnings.empty()) {
        for (const auto& w : warnings) log_warn(w);
    }
    Reporter reporter;
    for (const auto& sample : samples) {
        auto add = reporter.write_sample(sample);
        if (!add.ok()) return add;
    }
    const std::string base = cfg.results_dir + "/" + std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    st = reporter.write_raw_csv(base + "_raw.csv");
    if (!st.ok()) return st;
    const auto env = make_environment(cfg, "edge-prefill bench");
    st = reporter.write_report(base + "_report.md", samples, env);
    if (!st.ok()) return st;
    st = reporter.write_environment(base + "_env.json", env);
    if (!st.ok()) return st;
    std::cout << "samples=" << samples.size() << " out=" << base << "\n";
    return Status::success();
}

Status classify_heads_command(const Args& args) {
    if (args.model.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "classify-heads requires --model");
    }
    if (args.output.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "classify-heads requires --output");
    }
    auto st = LlamaBackend::init();
    if (!st.ok()) return st;
    LlamaBackend backend;
    backend.set_n_gpu_layers(args.n_gpu_layers);
    st = backend.load_model(args.model);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    ContextOptions opts;
    // n_ctx 0 lets llama.cpp pick its default; the probe-length check inside
    // classify() still rejects oversized probes with a param suggestion.
    opts.n_ctx = args.n_ctx;
    opts.n_ubatch = args.n_ubatch;
    opts.n_threads = args.threads;
    opts.n_threads_batch = args.threads;
    // The per-head ablation probe gathers rows via the non-flash path.
    opts.flash_attn = false;
    st = backend.create_context(opts);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }

    HeadProbeParams params;
    params.block_len = args.probe_block_len;
    params.repeats = args.probe_repeats;
    params.probe_k = args.probe_k;

    // AR005 FR-4: calibrate the probe segment from real-task text instead
    // of the fixed-seed random vocabulary sample.
    if (!args.calibration_file.empty()) {
        std::ifstream in(args.calibration_file, std::ios::binary);
        if (!in) {
            LlamaBackend::shutdown();
            return Status::error(ErrorCode::Io,
                                 "cannot read --calibration-file: " +
                                     args.calibration_file);
        }
        params.source = ProbeSource::CalibrationText;
        params.calibration_text.assign(std::istreambuf_iterator<char>(in),
                                       std::istreambuf_iterator<char>());
    }

    const auto fp = backend.fingerprint();
    HeadClassifier classifier(args.head_retrieval_threshold);
    HeadProfile profile;
    st = classifier.classify(&backend, params, fp.head_hash, &profile);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    st = save_head_profile(args.output, profile);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }

    uint32_t retrieval = 0;
    for (const HeadProfileEntry& entry : profile.heads) {
        if (entry.retrieval) ++retrieval;
    }
    std::cout << "heads=" << profile.heads.size() << " retrieval=" << retrieval
              << " n_layer=" << profile.n_layer << " n_head=" << profile.n_head
              << " out=" << args.output << "\n";
    LlamaBackend::shutdown();
    return Status::success();
}

Status inspect_command(const Args& args) {
    if (args.model.empty()) return Status::error(ErrorCode::InvalidArgument, "missing --model");
    auto st = LlamaBackend::init();
    if (!st.ok()) return st;
    LlamaBackend backend;
    backend.set_n_gpu_layers(args.n_gpu_layers);
    st = backend.load_model(args.model);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    const auto fp = backend.fingerprint();
    std::cout << "arch=" << fp.arch << " n_layer=" << fp.n_layer
              << " n_ctx_train=" << fp.n_ctx_train << " n_vocab=" << backend.n_vocab()
              << " file_size=" << fp.file_size << " head_hash=" << fp.head_hash << "\n";
    LlamaBackend::shutdown();
    return Status::success();
}

} // namespace
} // namespace edge_prefill

int main(int argc, char** argv) {
    using namespace edge_prefill;
    Args args;
    auto st = parse_args(argc, argv, &args);
    if (!st.ok()) {
        std::cerr << "error: " << st.message << '\n';
        usage();
        return static_cast<int>(st.code);
    }
    if (args.verbose) set_log_level(LogLevel::Debug);
    if (args.command == "run") st = run_command(args);
    else if (args.command == "plan") st = plan_command(args);
    else if (args.command == "bench" || args.command == "eval") st = bench_command(args);
    else if (args.command == "classify-heads") st = classify_heads_command(args);
    else if (args.command == "inspect") st = inspect_command(args);
    else if (args.command == "help" || args.command == "--help") {
        usage();
        return 0;
    } else {
        st = Status::error(ErrorCode::InvalidArgument, "unknown command: " + args.command);
    }
    if (!st.ok()) {
        std::cerr << "error[" << error_code_name(st.code) << "]: " << st.message << '\n';
        return static_cast<int>(st.code);
    }
    return 0;
}
