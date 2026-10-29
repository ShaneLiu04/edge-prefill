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
           "selected_tokens,mode,budget,attn_top_k,repeats,ppl_dense,ppl_sparse,"
           "query_gating,qg_groups,score_source,semantic_score_ms,draft_score_ms,"
           "qg_source,budget_source,anchor_blocks,mmr_penalized,phrase_blocks,"
           "specific_terms,coverage_queries,evidence_blocks,stop_reason,"
           "portfolio_choice,portfolio_reason,structure_tokens,evidence_tokens,"
           "span_tokens,uncovered_facts,sentences_completed,sentences_extended,"
           "ctx_cells,kv_alloc_bytes,kv_used_bytes,rss_mb,position_span,"
           "compaction_reason,ttft_min_ms\n";
    out << std::setprecision(10);
    for (const auto& s : samples_) {
        out << s.task << ',' << s.case_id << ',' << s.ttft_ms << ',' << s.e2e_ms << ','
            << s.prefill_tokens_per_s << ',' << s.kv_bytes << ',' << s.quality << ','
            << (s.dense ? 1 : 0) << ',' << s.prompt_tokens << ',' << s.selected_tokens << ','
            << mode_of(s) << ',' << s.budget << ',' << s.attn_top_k << ',' << s.repeats
            << ',' << s.ppl_dense << ',' << s.ppl_sparse << ','
            << (s.query_gating ? 1 : 0) << ',' << s.qg_groups << ','
            << (s.score_source.empty() ? "lexical" : s.score_source) << ','
            << s.semantic_score_ms << ',' << s.draft_score_ms << ','
            << s.qg_source << ',' << s.budget_source << ',' << s.anchor_blocks
            << ',' << s.mmr_penalized << ',' << s.phrase_blocks << ',' << s.specific_terms << ','
            << s.coverage_queries << ',' << s.evidence_blocks << ','
            << (s.stop_reason.empty() ? "budget" : s.stop_reason) << ','
            << (s.portfolio_choice.empty() ? "none" : s.portfolio_choice) << ','
            << s.portfolio_reason << ','
            << s.structure_tokens << ',' << s.evidence_tokens << ','
            << s.span_tokens << ',' << s.uncovered_facts << ','
            << s.sentences_completed << ',' << s.sentences_extended << ','
            << s.ctx_cells << ',' << s.kv_alloc_bytes << ',' << s.kv_used_bytes << ','
            << s.rss_mb << ',' << s.position_span << ',' << s.compaction_reason << ',' << s.ttft_min_ms
            << '\n';
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
           "prompt | selected | mode | budget | attn_top_k | repeats | score_source | "
           "semantic_ms | qg_source |\n";
    out << "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |\n";
    for (const auto& s : samples) {
        out << "| " << s.task << " | " << s.case_id << " | " << s.ttft_ms << " | " << s.e2e_ms
            << " | " << s.prefill_tokens_per_s << " | " << s.kv_bytes << " | " << s.quality
            << " | " << (s.dense ? "yes" : "no") << " | " << s.prompt_tokens << " | "
            << s.selected_tokens << " | " << s.mode << " | " << s.budget << " | "
            << s.attn_top_k << " | " << s.repeats << " | "
            << (s.score_source.empty() ? "lexical" : s.score_source) << " | "
            << s.semantic_score_ms << " | " << s.qg_source << " |\n";
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
