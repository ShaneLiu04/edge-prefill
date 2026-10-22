#include "edge_prefill/metrics.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "edge_prefill/json_writer.h"

namespace edge_prefill {
namespace {

std::string mode_of(const Sample& sample) { return sample.mode; }

void ensure_parent(const std::string& path) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
}

} // namespace

Status Reporter::write_sample(const Sample& sample) {
    if (sample.task.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "sample task is empty");
    }
    samples_.push_back(sample);
    return Status::success();
}

Status Reporter::write_raw_csv(const std::string& path) {
    if (path.empty()) return Status::error(ErrorCode::Io, "empty csv path");
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) return Status::error(ErrorCode::Io, "cannot write csv: " + path);
    out << "task,case,ttft_ms,e2e_ms,prefill_tps,kv_bytes,quality,dense,prompt_tokens,"
           "selected_tokens,mode,budget,attn_top_k,repeats,ppl_dense,ppl_sparse\n";
    out << std::setprecision(10);
    for (const auto& s : samples_) {
        out << s.task << ',' << s.case_id << ',' << s.ttft_ms << ',' << s.e2e_ms << ','
            << s.prefill_tokens_per_s << ',' << s.kv_bytes << ',' << s.quality << ','
            << (s.dense ? 1 : 0) << ',' << s.prompt_tokens << ',' << s.selected_tokens << ','
            << mode_of(s) << ',' << s.budget << ',' << s.attn_top_k << ',' << s.repeats
            << ',' << s.ppl_dense << ',' << s.ppl_sparse << '\n';
    }
    return Status::success();
}

Status Reporter::write_report(const std::string& path,
                              const std::vector<Sample>& samples,
                              const EnvironmentInfo& env) {
    if (path.empty()) return Status::error(ErrorCode::Io, "empty report path");
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) return Status::error(ErrorCode::Io, "cannot write report: " + path);
    out << "# EdgePrefill Benchmark Report\n\n";
    out << "- Model: `" << env.model_sha256 << "`\n";
    out << "- llama.cpp: `" << env.llama_commit << "`\n";
    out << "- CPU: " << env.cpu << "\n";
    out << "- Threads: " << env.threads << "\n";
    out << "- KV: " << env.kv_type << "\n";
    out << "- Command: `" << env.command << "`\n\n";
    out << "| task | case | TTFT ms | E2E ms | prefill tok/s | KV bytes | quality | dense | "
           "prompt | selected | mode | budget | attn_top_k | repeats |\n";
    out << "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |\n";
    for (const auto& s : samples) {
        out << "| " << s.task << " | " << s.case_id << " | " << s.ttft_ms << " | " << s.e2e_ms
            << " | " << s.prefill_tokens_per_s << " | " << s.kv_bytes << " | " << s.quality
            << " | " << (s.dense ? "yes" : "no") << " | " << s.prompt_tokens << " | "
            << s.selected_tokens << " | " << s.mode << " | " << s.budget << " | "
            << s.attn_top_k << " | " << s.repeats << " |\n";
    }
    if (samples.empty()) {
        out << "\n_No samples._\n";
    }
    return Status::success();
}

std::string environment_json(const EnvironmentInfo& env) {
    std::string out;
    JsonWriter json(&out);
    json.begin_object();
    json.key("os");
    json.value(env.os);
    json.key("cpu");
    json.value(env.cpu);
    json.key("physical_cores");
    json.value(static_cast<uint64_t>(env.physical_cores));
    json.key("logical_cores");
    json.value(static_cast<uint64_t>(env.logical_cores));
    json.key("compiler");
    json.value(env.compiler);
    json.key("cmake");
    json.value(env.cmake_version);
    json.key("llama_commit");
    json.value(env.llama_commit);
    json.key("model_sha256");
    json.value(env.model_sha256);
    json.key("data_sha256");
    json.value(env.data_sha256);
    json.key("threads");
    json.value(static_cast<uint64_t>(env.threads));
    json.key("kv_type");
    json.value(env.kv_type);
    json.key("command");
    json.value(env.command);
    json.key("timestamp");
    json.value(env.timestamp);
    json.end_object();
    return out;
}

Status Reporter::write_environment(const std::string& path, const EnvironmentInfo& env) {
    if (path.empty()) return Status::error(ErrorCode::Io, "empty environment path");
    ensure_parent(path);
    std::ofstream out(path);
    if (!out) return Status::error(ErrorCode::Io, "cannot write environment: " + path);
    out << environment_json(env) << '\n';
    return Status::success();
}

} // namespace edge_prefill
