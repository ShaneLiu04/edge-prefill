#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

#include "edge_prefill/backend.h"
#include "edge_prefill/benchmark.h"
#include "edge_prefill/config.h"
#include "edge_prefill/ffn_skip.h"
#include "edge_prefill/json_writer.h"
#include "edge_prefill/log.h"
#include "edge_prefill/metrics.h"
#include "edge_prefill/planner.h"
#include "edge_prefill/prefix_cache.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/sparse_prefill.h"

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
    uint32_t tokens = 0;
    uint32_t max_new_tokens = 32;
    uint32_t n_ctx = 0;
    uint32_t n_ubatch = 512;
    int32_t threads = 0;
    int32_t repeats = 5;
    double budget = 0.5;
    bool budget_set = false;
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
        "  edge-prefill inspect --model M [--json]\n\n"
        "Common options:\n"
        "  --mode latency|balanced|quality|reuse|dense\n"
        "  --budget <0..1>   --kv-type f16|q8_0|q4_0   --skip-ffn <spec>\n"
        "  --threads N       --n-ctx N                 --n-ubatch N\n"
        "  --max-new-tokens N  --chat   --dump-plan P  --dump-selection P\n"
        "  --verbose\n";
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
    opts.n_gpu_layers = run_cfg.n_gpu_layers;
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
    Selection selection;
    BlockSelector selector;
    st = selector.select(tokens, run_cfg.prompt, pieces, sopts, nullptr, &selection);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
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
    popts.chunk_size = plan.dense_fallback ? 512 : plan.chunk_size;
    popts.enable_prefix_cache = run_cfg.enable_prefix_cache;
    popts.enable_chunking = run_cfg.enable_chunking;
    popts.enable_ffn_skip = run_cfg.enable_ffn_skip;
    popts.ffn_skip_mask = plan.dense_fallback ? 0 : plan.skip_ffn_mask;
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
              << " mode=" << mode_name(run_cfg.mode) << " budget=" << sopts.budget << '\n';
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

Status inspect_command(const Args& args) {
    if (args.model.empty()) return Status::error(ErrorCode::InvalidArgument, "missing --model");
    auto st = LlamaBackend::init();
    if (!st.ok()) return st;
    LlamaBackend backend;
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
