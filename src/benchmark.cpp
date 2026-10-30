#include "edge_prefill/benchmark.h"

#include "edge_prefill/budget_curve.h"
#include "edge_prefill/cost_model.h"
#include "edge_prefill/exec_plan.h"
#include "edge_prefill/portfolio.h"
#include "edge_prefill/hash_util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/backend.h"
#include "edge_prefill/embedding_scorer.h"
#include "edge_prefill/head_profile.h"
#include "edge_prefill/log.h"
#include "edge_prefill/planner.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/kv_snapshot.h"
#include "edge_prefill/session.h"
#include "edge_prefill/semantic_gate.h"
#include "edge_prefill/sparse_prefill.h"

namespace edge_prefill {

namespace {

// AR013 FR-1: the first context is deliberately small — every case grows it to
// the position span it actually writes (LlamaBackend::ensure_context). Reserving
// the model's training context up front cost 896 MiB for a 146-token prompt.
constexpr uint32_t kSeedContextCells = 256;

} // namespace
namespace {

struct Case {
    std::string id;
    std::string task;
    std::string context_path;
    std::string question;
    std::string answer;
    // AR014: rows sharing a non-empty session run on one KV sequence, in file
    // order. An empty value keeps the row independent (AR013 behaviour).
    std::string session;
};

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    for (char c : line) {
        if (c == '\t') {
            fields.push_back(current);
            current.clear();
        } else if (c != '\r') {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string normalize_answer(std::string text) {
    std::string out;
    for (unsigned char c : text) {
        if (std::isalnum(c) || c == ' ') out.push_back(static_cast<char>(std::tolower(c)));
        else out.push_back(' ');
    }
    std::string collapsed;
    bool space = false;
    for (char c : out) {
        if (c == ' ') {
            if (!space && !collapsed.empty()) collapsed.push_back(' ');
            space = true;
        } else {
            collapsed.push_back(c);
            space = false;
        }
    }
    if (!collapsed.empty() && collapsed.back() == ' ') collapsed.pop_back();
    return collapsed;
}

double f1_score(const std::string& prediction, const std::string& answer) {
    const auto pred = normalize_answer(prediction);
    const auto gold = normalize_answer(answer);
    if (gold.empty()) return 0.0;
    if (pred.find(gold) != std::string::npos) return 1.0;
    std::istringstream ps(pred);
    std::istringstream gs(gold);
    std::unordered_map<std::string, int> pred_counts;
    std::unordered_map<std::string, int> gold_counts;
    std::string token;
    while (ps >> token) ++pred_counts[token];
    while (gs >> token) ++gold_counts[token];
    int overlap = 0;
    for (const auto& [word, count] : pred_counts) {
        auto it = gold_counts.find(word);
        if (it != gold_counts.end()) overlap += std::min(count, it->second);
    }
    if (overlap == 0) return 0.0;
    const int pred_total = static_cast<int>(pred_counts.size());
    const int gold_total = static_cast<int>(gold_counts.size());
    const double precision = static_cast<double>(overlap) / pred_total;
    const double recall = static_cast<double>(overlap) / gold_total;
    return 2.0 * precision * recall / (precision + recall);
}

std::string build_prompt(const llama_model* model, bool chat, const std::string& context,
                         const std::string& question) {
    if (chat) {
        const char* tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl) {
            llama_chat_message messages[2];
            messages[0].role = "system";
            messages[0].content = "You answer questions using only the provided context.";
            messages[1].role = "user";
            const std::string user = "Context:\n" + context + "\n\nQuestion: " + question +
                                     "\nAnswer with the shortest exact span.";
            messages[1].content = user.c_str();
            std::vector<char> buf(2 * (user.size() + 128));
            const int32_t len = llama_chat_apply_template(tmpl, messages, 2, true, buf.data(),
                                                          static_cast<int32_t>(buf.size()));
            if (len > 0 && static_cast<size_t>(len) <= buf.size()) {
                return std::string(buf.data(), static_cast<size_t>(len));
            }
        }
    }
    return "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:";
}

// AR014: one user turn rendered with the model's chat template, so a session
// turn is primed exactly like the single-turn benchmark prompt (assistant
// opener included) instead of ending on a bare "Answer:".
std::string build_chat_turn(const llama_model* model, const std::string& evidence,
                            const std::string& question) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    const std::string user = "Context excerpt:\n" + evidence + "\n\nQuestion: " + question +
                             "\nAnswer with the shortest exact span.";
    if (!tmpl) return "Context excerpt:\n" + evidence + "\n\nQuestion: " + question + "\nAnswer:";
    llama_chat_message message;
    message.role = "user";
    message.content = user.c_str();
    std::vector<char> buf(user.size() + 256);
    const int32_t len = llama_chat_apply_template(tmpl, &message, 1, true, buf.data(),
                                                  static_cast<int32_t>(buf.size()));
    if (len > 0 && static_cast<size_t>(len) <= buf.size()) {
        return std::string(buf.data(), static_cast<size_t>(len));
    }
    return "Context excerpt:\n" + evidence + "\n\nQuestion: " + question + "\nAnswer:";
}

std::string truncate_context(const std::vector<llama_token>& tokens,
                             const std::vector<std::string>& pieces, uint32_t max_tokens) {
    if (tokens.size() <= max_tokens) {
        std::string out;
        for (const auto& piece : pieces) out += piece;
        return out;
    }
    std::string out;
    for (uint32_t i = 0; i < max_tokens; ++i) out += pieces[i];
    return out;
}

llama_token greedy(const std::vector<float>& logits) {
    return static_cast<llama_token>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
}

double log_softmax_at(const std::vector<float>& logits, llama_token token) {
    if (logits.empty()) return 0.0;
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float value : logits) sum += std::exp(static_cast<double>(value - max_logit));
    const double log_z = std::log(sum) + max_logit;
    const int32_t idx = static_cast<int32_t>(token);
    if (idx < 0 || static_cast<size_t>(idx) >= logits.size()) return 0.0;
    return static_cast<double>(logits[static_cast<size_t>(idx)]) - log_z;
}

// AR014: the session path needs the same admission options as the regular
// path. Keeping one builder means a future option added to the selector cannot
// silently apply to only half of the benchmark.
// AR015: everything that decides *which* tokens a case admits, in one string.
// A snapshot is only valid inside one policy, so this string is part of its
// key; adding a knob here is what keeps old snapshots from being reused.
std::string admission_policy_key(const Config& cfg, const Plan& plan) {
    std::string out;
    auto add = [&out](const char* name, double v) {
        out += name;
        out += '=';
        out += std::to_string(v);
        out += ';';
    };
    add("mode", static_cast<double>(static_cast<int>(cfg.mode)));
    add("budget", plan.budget);
    add("dense_fallback", plan.dense_fallback ? 1.0 : 0.0);
    add("block", cfg.block_size);
    add("topk", cfg.attn_top_k);
    add("sinkb", cfg.sink_blocks);
    add("recentb", cfg.recent_blocks);
    add("covstride", cfg.coverage_stride);
    add("anchor", cfg.anchor_ratio);
    add("anchorscale", cfg.anchor_stride_scale);
    add("mmrt", cfg.mmr_threshold);
    add("mmrl", cfg.mmr_lambda);
    add("phrase", cfg.phrase_weight);
    add("coverage", cfg.coverage_weight);
    add("priorgate", cfg.prior_gate);
    add("fillpriors", cfg.fill_priors);
    add("structure", cfg.structure_budget);
    add("spanradius", cfg.anchor_span_radius);
    add("covreserve", cfg.coverage_reserve);
    add("spanadm", cfg.span_admission);
    add("spanmin", cfg.span_min_tokens);
    add("sentcomp", cfg.sentence_completion);
    add("senttrim", cfg.sentence_trim);
    add("spanmax", cfg.span_max_tokens);
    add("compaction", cfg.position_compaction);
    add("wlex", cfg.weights.lexical);
    add("wdiv", cfg.weights.diversity);
    out += "budget_curve=";
    out += cfg.budget_curve;
    out += ';';
    return out;
}

SelectionOptions selection_options_for(const Config& run_cfg, const Plan& plan) {
    SelectionOptions sopts;
    sopts.block_size = run_cfg.block_size;
    sopts.budget = plan.dense_fallback ? 1.0 : plan.budget;
    sopts.min_dense_tokens = run_cfg.min_dense_tokens;
    sopts.sink_blocks = run_cfg.sink_blocks;
    sopts.recent_blocks = run_cfg.recent_blocks;
    sopts.coverage_stride = run_cfg.coverage_stride;
    sopts.anchor_ratio = run_cfg.anchor_ratio;
    sopts.anchor_stride_scale = run_cfg.anchor_stride_scale;
    sopts.mmr_threshold = run_cfg.mmr_threshold;
    sopts.mmr_lambda = run_cfg.mmr_lambda;
    sopts.phrase_weight = run_cfg.phrase_weight;
    sopts.coverage_weight = run_cfg.coverage_weight;
    sopts.prior_gate = run_cfg.prior_gate;
    sopts.fill_priors = run_cfg.fill_priors;
    sopts.structure_budget = run_cfg.structure_budget;
    sopts.anchor_span_radius = run_cfg.anchor_span_radius;
    sopts.coverage_reserve = run_cfg.coverage_reserve;
    sopts.span_admission = run_cfg.span_admission;
    sopts.span_min_tokens = run_cfg.span_min_tokens;
    sopts.sentence_completion = run_cfg.sentence_completion;
    sopts.span_max_tokens = run_cfg.span_max_tokens;
    sopts.sentence_trim = run_cfg.sentence_trim;
    sopts.position_compaction = run_cfg.position_compaction;
    sopts.weights = run_cfg.weights;
    return sopts;
}

// AR004 FR-5: mean answer log-probability after a prefill. The caller
// supplies the first-token logits of the prefilled context; each further
// answer token consumes one decode step. Returns false when the answer
// cannot be scored end to end (decode failure) so callers can skip the
// sample instead of publishing a fabricated quality number.
bool score_answer_mean_logprob(LlamaBackend& backend, const float* first_logits,
                               size_t prompt_tokens,
                               const std::vector<llama_token>& answer_tokens,
                               double* mean_lp) {
    if (!first_logits || answer_tokens.empty() || !mean_lp) return false;
    std::vector<float> logits(first_logits, first_logits + backend.n_vocab());
    double logprob_sum = 0.0;
    for (size_t i = 0; i < answer_tokens.size(); ++i) {
        logprob_sum += log_softmax_at(logits, answer_tokens[i]);
        if (i + 1 == answer_tokens.size()) break;
        std::vector<float> step_logits;
        if (!backend.decode(answer_tokens[i],
                            static_cast<llama_pos>(prompt_tokens + i), &step_logits)
                 .ok()) {
            return false;
        }
        logits = std::move(step_logits);
    }
    *mean_lp = logprob_sum / static_cast<double>(answer_tokens.size());
    return true;
}

} // namespace

Status BenchmarkRunner::run(const BenchmarkConfig& cfg, std::vector<Sample>* out,
                            std::vector<std::string>* warnings) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null sample output");
    if (cfg.model.empty()) return Status::error(ErrorCode::InvalidArgument, "empty model");
    if (cfg.data_dir.empty()) return Status::error(ErrorCode::InvalidArgument, "empty data_dir");
    if (cfg.modes.empty()) return Status::error(ErrorCode::InvalidArgument, "empty modes");
    if (cfg.repeats < 0) return Status::error(ErrorCode::InvalidArgument, "negative repeats");
    if (cfg.repeats == 0) {
        if (warnings) warnings->push_back("repeats-defaulted");
    }

    const std::string index_path = (std::filesystem::path(cfg.data_dir) / "index.tsv").string();
    std::ifstream index(index_path);
    if (!index) {
        return Status::error(ErrorCode::Io, "missing dataset index: " + index_path +
                                                " (run benchmarks/scripts/gen_niah.py)");
    }
    std::vector<Case> cases;
    std::string line;
    while (std::getline(index, line)) {
        if (line.empty()) continue;
        const auto fields = split_tsv(line);
        if (fields.size() < 5) continue;
        if (fields[0] == "id" && fields[1] == "task") continue;
        Case row{fields[0], fields[1], fields[2], fields[3], fields[4], ""};
        if (fields.size() >= 6) row.session = fields[5];
        cases.push_back(std::move(row));
    }
    if (cases.empty()) {
        return Status::error(ErrorCode::Io, "dataset index has no cases: " + index_path);
    }

    Config base;
    base.model_path = cfg.model;
    base.n_ctx = 0;
    base.max_new_tokens = cfg.max_new_tokens;
    base.repeats = cfg.repeats == 0 ? 5 : static_cast<uint32_t>(cfg.repeats);
    base.warmup = cfg.warmup;
    base.use_chat_template = true;
    base.kv_type = KvPrecision::F16;
    base.results_dir = cfg.out_dir;
    base.attn_top_k = cfg.attn_top_k;
    base.attn_top_k_layers = cfg.attn_top_k_layers;
    base.weights = cfg.weights;
    base.sink_blocks = cfg.sink_blocks;
    base.recent_blocks = cfg.recent_blocks;
    base.coverage_stride = cfg.coverage_stride;
    base.anchor_ratio = cfg.anchor_ratio;
    base.anchor_stride_scale = cfg.anchor_stride_scale;
    base.mmr_threshold = cfg.mmr_threshold;
    base.mmr_lambda = cfg.mmr_lambda;
    base.budget_curve = cfg.budget_curve;
    base.phrase_weight = cfg.phrase_weight;
    base.coverage_weight = cfg.coverage_weight;
    base.prior_gate = cfg.prior_gate;
    base.fill_priors = cfg.fill_priors;
    base.structure_budget = cfg.structure_budget;
    base.anchor_span_radius = cfg.anchor_span_radius;
    base.coverage_reserve = cfg.coverage_reserve;
    base.span_admission = cfg.span_admission;
    base.span_min_tokens = cfg.span_min_tokens;
    base.sentence_completion = cfg.sentence_completion;
    base.span_max_tokens = cfg.span_max_tokens;
    base.sentence_trim = cfg.sentence_trim;
    base.position_compaction = cfg.position_compaction;
    base.session_reuse = cfg.session_reuse;
    base.session_ctx_max = cfg.session_ctx_max;
    base.snapshot_dir = cfg.snapshot_dir;
    base.snapshot_use = cfg.snapshot_use;
    base.snapshot_max_bytes = cfg.snapshot_max_bytes;
    base.dump_selection_dir = cfg.dump_selection_dir;
    base.portfolio_table = cfg.portfolio_table;
    if (!base.budget_curve.empty()) {
        BudgetCurve curve;
        const auto curve_st = load_budget_curve(base.budget_curve, &curve);
        if (!curve_st.ok()) {
            if (warnings) warnings->push_back("budget-curve-unusable:" + curve_st.message);
            base.budget_curve.clear();
        } else if (!curve.matches_model(sha256_of_file(base.model_path))) {
            if (warnings) {
                warnings->push_back("budget-curve-model-mismatch:" + base.budget_curve);
            }
            log_warn("budget curve model fingerprint mismatch; falling back to defaults");
            base.budget_curve.clear();
        }
    }

    // AR016: the model fingerprint is a 1 GB SHA256 (seconds of CPU). It used to
    // be recomputed inside the per-cell portfolio check, which stole CPU from
    // timed prefills of neighbouring cells and inflated the variance. Compute it
    // at most once per run.
    std::string model_sha256;
    bool model_sha256_ready = false;
    auto model_hash = [&]() -> const std::string& {
        if (!model_sha256_ready) {
            model_sha256 = sha256_of_file(base.model_path);
            model_sha256_ready = true;
        }
        return model_sha256;
    };

    LlamaBackend backend;
    auto st = LlamaBackend::init();
    if (!st.ok()) return st;
    backend.set_n_gpu_layers(cfg.n_gpu_layers);
    st = backend.load_model(base.model_path);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    // AR016 FR-1: verify the requested placement instead of assuming it. A
    // request for GPU layers that offloads nothing is an error unless the
    // caller explicitly opted into the CPU fallback.
    PlacementReport placement;
    placement.requested_layers = cfg.n_gpu_layers;
    placement.offloaded_layers = backend.placement().offloaded_layers;
    placement.total_layers = backend.placement().total_layers;
    placement.gpu_device_present = backend.placement().gpu_device_present;
    placement.build_supports_gpu = backend.placement().build_supports_gpu;
    placement.device_name = backend.placement().device_name;
    st = placement.resolve(cfg.allow_cpu_fallback);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    if (placement.state == DeviceState::CpuFallback) {
        log_warn("requested GPU layers but nothing was offloaded; running on CPU (device_state=cpu(fallback))");
    }
    const std::string device_state = device_state_name(placement.state);
    const std::string device_name = backend.placement().device_name;
    ContextOptions ctx_opts;
    // AR013 FR-1: an explicit n_ctx (CLI/conf) is honoured instead of always
    // letting llama.cpp pick the training context. 0 keeps the lazy behaviour:
    // the context starts at the default and grows on demand through
    // LlamaBackend::ensure_context to the span the case actually writes.
    ctx_opts.n_ctx = cfg.n_ctx != 0 ? cfg.n_ctx : kSeedContextCells;
    // AR016 FR-1: the plan's execution knobs are honoured. 0 keeps the old
    // behaviour (auto threads, 512 ubatch, gather-friendly non-flash graph).
    ctx_opts.n_ubatch = cfg.n_ubatch != 0 ? cfg.n_ubatch : 512;
    ctx_opts.n_threads = cfg.n_threads;
    ctx_opts.n_threads_batch = cfg.n_threads_batch;
    ctx_opts.n_batch = std::max<uint32_t>(2048, ctx_opts.n_ubatch);
    ctx_opts.type_k = KvPrecision::F16;
    ctx_opts.type_v = KvPrecision::F16;
    // The attention gather paths (scalar top-k and per-head matrix) require
    // the non-flash layout; keep every mode on the same graph family so
    // dense-vs-sparse timings stay internally comparable.
    ctx_opts.flash_attn = cfg.flash_attn;
    st = backend.create_context(ctx_opts);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }

    // AR003: load and validate the offline head profile once. A missing or
    // mismatched profile downgrades the run to the scalar path with a
    // warning instead of failing.
    HeadProfile head_profile;
    bool head_ok = false;
    if (cfg.head_adaptive) {
        if (cfg.head_profile.empty()) {
            if (warnings) warnings->push_back("head-adaptive-no-profile");
        } else {
            auto pst = load_head_profile(cfg.head_profile, &head_profile);
            if (!pst.ok()) {
                if (warnings) warnings->push_back("head-profile-load:" + pst.message);
            } else {
                auto vst = validate_head_profile(head_profile, backend.n_layer(),
                                                 backend.n_head());
                if (!vst.ok()) {
                    if (warnings) warnings->push_back("head-profile-invalid:" + vst.message);
                } else {
                    head_ok = true;
                }
            }
        }
    }

    std::unique_ptr<PrefixCache> cache;
    if (base.enable_prefix_cache) {
        PrefixCache::create(base.prefix_cache_bytes, backend.fingerprint().to_key(), &cache);
    }
    SparsePrefiller prefilter(&backend, cache.get());
    // AR015: one snapshot directory per run; empty disables the feature
    // entirely so every existing configuration stays bit-identical.
    KvSnapshotStore* snapshot_store = nullptr;
    if (!base.snapshot_dir.empty()) {
        auto sst = KvSnapshotStore::create(base.snapshot_dir, base.snapshot_max_bytes,
                                           &snapshot_store);
        if (!sst.ok()) {
            log_warn("snapshot store disabled: " + sst.message);
            snapshot_store = nullptr;
        }
    }
    // AR014: one KV sequence per conversation.
    SessionStore sessions;

    // AR005: explicit embedding configuration never degrades — a failed
    // load fails the whole run so the semantic signal is either real or
    // the run never starts.
    EmbeddingScorer embed;
    if (!cfg.embedding_model.empty()) {
        auto embed_st = embed.load_model(cfg.embedding_model, cfg.embed_threads);
        if (!embed_st.ok()) {
            LlamaBackend::shutdown();
            return embed_st;
        }
    }

    const auto wall_begin = std::chrono::steady_clock::now();
    // AR004 FR-5: dense ppl reference per (case, length). Computed once and
    // reused across modes/budgets so every ppl sample's quality is a
    // retention ratio against the same dense number.
    std::unordered_map<std::string, double> ppl_dense_ref;
    for (const auto& c : cases) {
        const std::string context_path =
            (std::filesystem::path(cfg.data_dir) / c.context_path).string();
        std::string context = read_file(context_path);
        if (context.empty()) {
            if (warnings) warnings->push_back("missing-context:" + c.id);
            continue;
        }
        std::vector<llama_token> ctx_tokens;
        st = backend.tokenize(context, &ctx_tokens, false, false);
        if (!st.ok()) continue;
        std::vector<std::string> ctx_pieces(ctx_tokens.size());
        for (size_t i = 0; i < ctx_tokens.size(); ++i) {
            backend.token_to_piece(ctx_tokens[i], &ctx_pieces[i]);
        }

        for (uint32_t length : cfg.lengths) {
            if (length == 0) length = static_cast<uint32_t>(ctx_tokens.size());
            context = truncate_context(ctx_tokens, ctx_pieces, length);
            if (!c.session.empty()) {
                // AR014: a conversation turn. Both arms walk the same log; the
                // only difference is whether the prefix already in the KV cache
                // is recomputed (`session_reuse=0`) or reused (`=1`).
                SessionState* session = nullptr;
                auto bst = sessions.begin(c.session, c.context_path, &session);
                if (!bst.ok()) {
                    if (warnings) warnings->push_back("session-mismatch:" + c.id);
                    continue;
                }
                std::vector<llama_token> doc_tokens;
                if (!backend.tokenize(context, &doc_tokens, false, false).ok() ||
                    doc_tokens.empty()) {
                    if (warnings) warnings->push_back("session-doc-tokenize:" + c.id);
                    continue;
                }
                std::vector<std::string> doc_pieces(doc_tokens.size());
                for (size_t i = 0; i < doc_tokens.size(); ++i) {
                    backend.token_to_piece(doc_tokens[i], &doc_pieces[i]);
                }

                for (Mode mode : cfg.modes) {
                    for (double budget : cfg.budgets) {
                        if (mode == Mode::Dense) continue;
                        if (budget >= 0.95) continue;
                        Config run_cfg = base;
                        run_cfg.mode = mode;
                        if (budget > 0.0) run_cfg.budget = budget;
                        run_cfg.budget_overridden = budget > 0.0;
                        PlannerInput pin;
                        pin.n_tokens = static_cast<uint32_t>(doc_tokens.size());
                        pin.mode = mode;
                        pin.arch_supports_ffn_skip = true;
                        pin.n_layer = backend.n_layer();
                        pin.n_ubatch = backend.context_options().n_ubatch;
                        Plan plan;
                        if (!Planner::make_plan(pin, run_cfg, &plan).ok()) continue;

                        SelectionOptions sopts = selection_options_for(run_cfg, plan);
                        Selection selection;
                        BlockSelector selector;
                        auto sst = selector.select(doc_tokens, c.question, doc_pieces,
                                                   sopts, nullptr, &selection);
                        if (!sst.ok()) continue;

                        // delta = this turn's admitted evidence that is not in
                        // the log yet, in document order.
                        std::vector<uint32_t> new_admitted;
                        std::vector<llama_token> delta;
                        size_t cursor = 0;
                        for (uint32_t idx : selection.token_indices) {
                            while (cursor < session->admitted.size() &&
                                   session->admitted[cursor] < idx) {
                                ++cursor;
                            }
                            if (cursor < session->admitted.size() &&
                                session->admitted[cursor] == idx) {
                                continue;
                            }
                            new_admitted.push_back(idx);
                            delta.push_back(doc_tokens[idx]);
                        }
                        // A turn is a normal chat turn: the first one carries the
                        // system prompt and the whole admitted evidence (exactly the
                        // AR013 prompt), later ones append the previous assistant
                        // tail plus this turn's new excerpt and question.
                        std::string evidence_text;
                        for (uint32_t idx : new_admitted) evidence_text += doc_pieces[idx];
                        std::string turn_text;
                        bool add_special = false;
                        if (session->turns == 0) {
                            std::string all_evidence;
                            for (uint32_t idx : selection.token_indices) {
                                all_evidence += doc_pieces[idx];
                            }
                            turn_text = build_prompt(backend.model(), base.use_chat_template,
                                                     all_evidence, c.question);
                            add_special = true;
                        } else {
                            turn_text = session->pending_tail +
                                        build_chat_turn(backend.model(), evidence_text,
                                                        c.question);
                        }
                        std::vector<llama_token> turn_tokens;
                        if (!backend.tokenize(turn_text, &turn_tokens, add_special, true).ok() ||
                            turn_tokens.empty()) {
                            if (warnings) warnings->push_back("session-turn-tokenize:" + c.id);
                            continue;
                        }
                        delta.insert(delta.end(), turn_tokens.begin(), turn_tokens.end());

                        const uint32_t reused = static_cast<uint32_t>(session->log.size());
                        std::vector<llama_token> log_after = session->log;
                        log_after.insert(log_after.end(), delta.begin(), delta.end());

                        // Re-creating the context wipes the KV cache, so a
                        // session must not grow mid-conversation. Size it once,
                        // for the whole conversation, from the admission bound
                        // (turns x per-turn bound); if that bound is ever
                        // exceeded the turn rebuilds the log instead of
                        // decoding a delta into an empty cache.
                        const uint32_t need = static_cast<uint32_t>(log_after.size()) +
                                              base.max_new_tokens + 8;
                        uint32_t wanted = need;
                        if (session->turns == 0) {
                            uint32_t session_turns = 0;
                            for (const auto& other : cases) {
                                if (other.session == c.session) ++session_turns;
                            }
                            // Estimate the whole conversation from this turn's
                            // delta (its admitted evidence + question) with 50%
                            // slack. Overshooting wastes KV, undershooting only
                            // costs one rebuild, so cap it.
                            const uint32_t per_turn =
                                    static_cast<uint32_t>(delta.size() * 3 / 2) +
                                    base.max_new_tokens + 32;
                            wanted = std::max(wanted, session_turns * per_turn + 256);
                        }
                        if (base.session_ctx_max) {
                            wanted = std::min(wanted, std::max(base.session_ctx_max, need));
                        }
                        const uint32_t creations_before = backend.context_creations();
                        const auto grow = backend.ensure_context(wanted);
                        if (!grow.ok()) {
                            if (warnings) warnings->push_back("session-grow:" + c.id);
                            continue;
                        }
                        bool reuse = run_cfg.session_reuse != 0 &&
                                     session->next_pos > 0 && !delta.empty();
                        if (reuse && backend.context_creations() != creations_before) {
                            reuse = false;
                            log_warn("session " + c.session +
                                     ": context re-created, rebuilding the log");
                        }

                        // AR015 x AR014: a resumed conversation can restore the
                        // KV this exact log produced in a previous process. The
                        // key covers the whole log, so a hit proves the prefix
                        // matches and the turn's delta does not need decoding.
                        SnapshotKey turn_key;
                        SnapshotPayload turn_payload;
                        SnapshotLoadResult turn_load;
                        SnapshotSaveResult turn_save;
                        bool turn_from_snapshot = false;
                        if (snapshot_store != nullptr) {
                            turn_key.model_key = backend.fingerprint().to_key();
                            turn_key.policy_key = admission_policy_key(base, plan);
                            turn_key.kv_type = kv_precision_name(base.kv_type);
                            turn_key.n_tokens = static_cast<uint32_t>(log_after.size());
                            turn_key.n_ctx = llama_n_ctx(backend.context());
                            turn_key.token_hash = hash_tokens(log_after);
                            std::vector<llama_pos> turn_positions(log_after.size());
                            std::iota(turn_positions.begin(), turn_positions.end(), 0);
                            turn_key.position_hash = hash_positions(turn_positions);
                            if (base.snapshot_use != 0) {
                                turn_load = snapshot_store->load(turn_key, log_after,
                                                                 &turn_payload);
                                if (turn_load.hit) {
                                    turn_from_snapshot =
                                            backend.load_state(turn_payload.state).ok();
                                    if (!turn_from_snapshot) turn_load.state = "miss:load";
                                }
                            }
                        }

                        std::vector<llama_token> batch_tokens = reuse ? delta : log_after;
                        Selection dense;
                        dense.dense = true;
                        dense.total_tokens = static_cast<uint32_t>(batch_tokens.size());
                        dense.token_indices.resize(batch_tokens.size());
                        std::iota(dense.token_indices.begin(), dense.token_indices.end(), 0u);
                        if (reuse) {
                            dense.positions.resize(delta.size());
                            for (size_t i = 0; i < delta.size(); ++i) {
                                dense.positions[i] = session->next_pos +
                                                     static_cast<uint32_t>(i);
                            }
                            dense.position_span = session->next_pos +
                                                  static_cast<uint32_t>(delta.size());
                        } else {
                            dense.position_span = static_cast<uint32_t>(log_after.size());
                        }

                        SparsePrefiller::Options popts;
                        popts.chunk_size = plan.chunk_size;
                        popts.enable_prefix_cache = false;
                        popts.enable_chunking = true;
                        popts.enable_ffn_skip = false;
                        popts.enable_attn_sparse = false;
                        popts.append = reuse;

                        PrefillStats stats;
                        const auto turn_begin = std::chrono::steady_clock::now();
                        if (turn_from_snapshot) {
                            auto* restored_mem = llama_get_memory(backend.context());
                            stats.kv_pos_max = llama_memory_seq_pos_max(restored_mem, 0);
                            stats.kv_pos_min = llama_memory_seq_pos_min(restored_mem, 0);
                            stats.prompt_tokens = static_cast<uint32_t>(log_after.size());
                            stats.selected_tokens = static_cast<uint32_t>(log_after.size());
                            stats.kv_bytes = turn_payload.state.size();
                        } else {
                            st = prefilter.prefill(batch_tokens, dense, popts, &stats);
                        }
                        const auto turn_first = std::chrono::steady_clock::now();
                        if (!st.ok()) {
                            SessionStore::reset(session, st.message);
                            if (warnings) warnings->push_back("session-prefill:" + c.id);
                            continue;
                        }
                        // A resumed turn did its work (read + verify + restore)
                        // during the probe above, so that is its TTFT; charging
                        // only the otherwise-empty timed region would report 0.
                        const double ttft_ms =
                                turn_from_snapshot
                                        ? turn_load.load_ms
                                        : std::chrono::duration<double, std::milli>(
                                                  turn_first - turn_begin)
                                                  .count();

                        // AR015: persist the turn's prompt KV for the next
                        // process. Taken before generation so the stored state
                        // holds exactly `log_after` at positions 0..n-1.
                        if (snapshot_store != nullptr && !turn_from_snapshot) {
                            const float* turn_raw = llama_get_logits_ith(backend.context(), -1);
                            if (turn_raw != nullptr) {
                                SnapshotPayload out_payload;
                                out_payload.logits.assign(turn_raw, turn_raw + backend.n_vocab());
                                out_payload.n_vocab =
                                        static_cast<uint32_t>(out_payload.logits.size());
                                if (backend.save_state(&out_payload.state).ok()) {
                                    turn_save =
                                            snapshot_store->save(turn_key, log_after, out_payload);
                                }
                            }
                        }

                        std::vector<float> turn_logits_owned;
                        if (turn_from_snapshot) {
                            turn_logits_owned = turn_payload.logits;
                        } else if (const float* raw = llama_get_logits_ith(backend.context(), -1)) {
                            turn_logits_owned.assign(raw, raw + backend.n_vocab());
                        }
                        const float* logits =
                                turn_logits_owned.empty() ? nullptr : turn_logits_owned.data();
                        std::string generated;
                        // AR014: every id we actually decode becomes part of the
                        // conversation KV, so the log must contain it — otherwise
                        // the next turn would append at positions that still hold
                        // the generated tokens and the two arms would silently
                        // diverge (measured: same log hash, different answers).
                        std::vector<llama_token> decoded_ids;
                        if (logits) {
                            std::vector<float> head(logits, logits + backend.n_vocab());
                            llama_token next = greedy(head);
                            for (uint32_t i = 0; i < base.max_new_tokens; ++i) {
                                std::string piece;
                                backend.token_to_piece(next, &piece);
                                generated += piece;
                                if (next == llama_vocab_eos(backend.vocab())) break;
                                std::vector<float> step;
                                if (!backend.decode(
                                            next,
                                            static_cast<llama_pos>(stats.kv_pos_max) + 1 +
                                                    static_cast<llama_pos>(i),
                                            &step)
                                             .ok()) {
                                    break;
                                }
                                decoded_ids.push_back(next);
                                next = greedy(step);
                            }
                        }
                        const auto turn_end = std::chrono::steady_clock::now();
                        const double e2e_ms = std::chrono::duration<double, std::milli>(
                                turn_end - turn_begin).count();

                        Sample sample;
                        sample.task = c.task;
                        sample.case_id = c.id + "@" + std::to_string(length);
                        sample.ttft_ms = ttft_ms;
                        sample.ttft_min_ms = ttft_ms;
                        sample.e2e_ms = e2e_ms;
                        sample.prefill_tokens_per_s =
                                stats.total_us > 0
                                        ? static_cast<double>(stats.selected_tokens) * 1e6 /
                                                  stats.total_us
                                        : 0.0;
                        sample.kv_bytes = stats.kv_bytes;
                        sample.quality = f1_score(generated, c.answer);
                        sample.dense = false;
                        sample.prompt_tokens = stats.prompt_tokens;
                        sample.selected_tokens = stats.selected_tokens;
                        sample.mode = mode_name(mode);
                        sample.budget = sopts.budget;
                        sample.attn_top_k = 0;
                        sample.repeats = 1;
                        sample.score_source = selection.score_source;
                        sample.budget_source = plan.budget_source;
                        sample.anchor_blocks = selection.anchor_blocks;
                        sample.phrase_blocks = selection.phrase_blocks;
                        sample.specific_terms = selection.specific_terms;
                        sample.coverage_queries = selection.coverage_queries;
                        sample.evidence_blocks = selection.evidence_blocks;
                        sample.stop_reason = selection.stop_reason;
                        sample.structure_tokens = selection.structure_tokens;
                        sample.evidence_tokens = selection.evidence_tokens;
                        sample.span_tokens = selection.span_tokens;
                        sample.sentences_completed = selection.sentences_completed;
                        sample.sentences_extended = selection.sentences_extended;
                        sample.uncovered_facts = selection.uncovered_facts;
                        sample.ctx_cells = backend.n_ctx();
                        sample.kv_alloc_bytes = backend.kv_alloc_bytes();
                        sample.kv_used_bytes = sample.kv_bytes;
                        sample.rss_mb = backend.rss_mb();
                        sample.position_span = dense.position_span;
                        sample.compaction_reason.clear();
                        sample.session_id = c.session;
                        sample.reused_tokens = reuse ? reused : 0;
                        sample.appended_tokens = static_cast<uint32_t>(batch_tokens.size());
                        sample.session_reset_reason = session->reset_reason;
                        uint64_t log_hash = 1469598103934665603ull;
                        std::vector<llama_token> log_full = session->log;
                        for (llama_token tok : log_full) {
                            log_hash ^= static_cast<uint64_t>(static_cast<uint32_t>(tok));
                            log_hash *= 1099511628211ull;
                        }
                        sample.session_log_tokens = session->next_pos;
                        sample.session_log_hash = log_hash;
                        // AR015 audit on the session path: a resumed turn is
                        // served from a snapshot instead of the delta decode.
                        sample.snapshot_hit = turn_from_snapshot ? 1 : 0;
                        sample.snapshot_state =
                                snapshot_store == nullptr
                                        ? "off"
                                        : (turn_from_snapshot
                                                   ? turn_load.state
                                                   : (base.snapshot_use != 0 ? turn_load.state
                                                                             : "cold"));
                        sample.snapshot_bytes =
                                turn_from_snapshot ? turn_load.bytes : turn_save.bytes;
                        sample.snapshot_load_ms = turn_from_snapshot ? turn_load.load_ms : 0.0;
                        sample.snapshot_save_ms = turn_save.ok ? turn_save.save_ms : 0.0;
                        sample.snapshot_key =
                                snapshot_store != nullptr ? turn_key.hex().substr(0, 16)
                                                          : std::string();
                        log_info("turn=" + sample.case_id + " session=" + c.session +
                                 " reused=" + std::to_string(sample.reused_tokens) +
                                 " appended=" + std::to_string(sample.appended_tokens) +
                                 " quality=" + std::to_string(sample.quality) +
                                 " generated=" + generated);
                        out->push_back(sample);

                        // Commit only after the turn succeeded, so a failure
                        // never leaves a half-written log behind.
                        log_after.insert(log_after.end(), decoded_ids.begin(),
                                         decoded_ids.end());
                        session->log = std::move(log_after);
                        session->admitted.insert(session->admitted.end(),
                                                 new_admitted.begin(),
                                                 new_admitted.end());
                        std::sort(session->admitted.begin(), session->admitted.end());
                        session->next_pos = static_cast<uint32_t>(session->log.size());
                        session->pending_tail = "<|im_end|>\n";
                        ++session->turns;
                    }
                }
                continue;
            }
            const std::string prompt = build_prompt(backend.model(), base.use_chat_template,
                                                    context, c.question);
            std::vector<llama_token> tokens;
            st = backend.tokenize(prompt, &tokens, true, true);
            if (!st.ok()) continue;
            // AR013: the hard ceiling is the model's training context (or an
            // explicit n_ctx), not the size of the context we happen to have
            // allocated — that one is grown on demand to the compacted span.
            // With compaction on, the raw prompt may exceed the ceiling; the
            // span check happens after selection (FR-3).
            const uint32_t hard_limit =
                    cfg.n_ctx != 0
                            ? cfg.n_ctx
                            : static_cast<uint32_t>(std::max(0, backend.n_ctx_train()));
            if (base.position_compaction == 0 && hard_limit > 0 &&
                tokens.size() + base.max_new_tokens + 8 > hard_limit) {
                if (warnings) {
                    warnings->push_back("context-too-long:" + c.id + "@" +
                                        std::to_string(length));
                }
                continue;
            }
            std::vector<std::string> pieces(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                backend.token_to_piece(tokens[i], &pieces[i]);
            }

            // AR004 FR-5: for ppl cases, establish the dense reference
            // (mean answer log-prob over a fully dense prefill) before the
            // sparse configs run. NaN marks an unusable reference (bad
            // answer text, prefill or scoring failure); the sample is
            // skipped with an explicit warning, never scored with 0.
            double ppl_lp_dense = std::numeric_limits<double>::quiet_NaN();
            if (c.task == "ppl") {
                const std::string ref_key = c.id + "@" + std::to_string(length);
                const auto cached = ppl_dense_ref.find(ref_key);
                if (cached != ppl_dense_ref.end()) {
                    ppl_lp_dense = cached->second;
                } else {
                    std::vector<llama_token> answer_tokens;
                    if (!backend.tokenize(c.answer, &answer_tokens, false, false).ok() ||
                        answer_tokens.empty()) {
                        if (warnings) warnings->push_back("ppl-answer-invalid:" + c.id);
                    } else {
                        Selection dense_sel;
                        dense_sel.dense = true;
                        dense_sel.total_tokens = static_cast<uint32_t>(tokens.size());
                        dense_sel.token_indices.resize(tokens.size());
                        std::iota(dense_sel.token_indices.begin(),
                                  dense_sel.token_indices.end(), 0u);
                        SparsePrefiller::Options ref_opts;
                        ref_opts.enable_prefix_cache = false;
                        ref_opts.enable_ffn_skip = false;
                        PrefillStats ref_stats;
                        const auto grow = backend.ensure_context(
                                static_cast<uint32_t>(tokens.size()) +
                                base.max_new_tokens + 8);
                        if (!grow.ok()) {
                            if (warnings) {
                                warnings->push_back("ppl-dense-ref-context:" + grow.message);
                            }
                            ppl_dense_ref[ref_key] = ppl_lp_dense;
                            continue;
                        }
                        if (!prefilter.prefill(tokens, dense_sel, ref_opts, &ref_stats).ok()) {
                            if (warnings) warnings->push_back("ppl-dense-ref-failed:" + c.id);
                        } else {
                            const float* ref_logits = llama_get_logits_ith(backend.context(), -1);
                            double lp = 0.0;
                            if (!score_answer_mean_logprob(backend, ref_logits,
                                                           static_cast<size_t>(ref_stats.kv_pos_max) + 1,
                                                           answer_tokens, &lp)) {
                                if (warnings) {
                                    warnings->push_back("ppl-dense-ref-failed:" + c.id);
                                }
                            } else {
                                ppl_lp_dense = lp;
                            }
                        }
                    }
                    ppl_dense_ref[ref_key] = ppl_lp_dense;
                }
                if (std::isnan(ppl_lp_dense)) continue;
            }

            for (Mode mode : cfg.modes) {
                for (double budget : cfg.budgets) {
                    if (mode == Mode::Dense && budget < 0.95) continue;
                    if (mode != Mode::Dense && budget >= 0.95) continue;
                    Config run_cfg = base;
                    run_cfg.mode = mode;
                    // A 0 grid point is the AR007 "let the curve decide"
                    // sentinel: keep the conf budget untouched and let the
                    // planner read the curve instead of overriding it.
                    if (budget > 0.0) run_cfg.budget = budget;
                    // The loop budget is authoritative for a bench grid:
                    // without this the planner silently falls back to the
                    // mode default (0.5 for balanced, then the 4096-token
                    // clamp), which masked every conf/CLI budget request
                    // since AR002. Dense stays 1.0 - the planner checks
                    // mode==Dense before the override.
                    run_cfg.budget_overridden = budget > 0.0;
                    PlannerInput pin;
                    pin.n_tokens = static_cast<uint32_t>(tokens.size());
                    pin.mode = mode;
                    pin.arch_supports_ffn_skip = true;
                    pin.n_layer = backend.n_layer();
                    pin.n_ubatch = backend.context_options().n_ubatch;
                    Plan plan;
                    st = Planner::make_plan(pin, run_cfg, &plan);
                    if (!st.ok()) continue;

                    SelectionOptions sopts;
                    sopts.block_size = run_cfg.block_size;
                    sopts.budget = plan.dense_fallback ? 1.0 : plan.budget;
                    sopts.min_dense_tokens = run_cfg.min_dense_tokens;
                    sopts.sink_blocks = run_cfg.sink_blocks;
                    sopts.recent_blocks = run_cfg.recent_blocks;
                    sopts.coverage_stride = run_cfg.coverage_stride;
                    sopts.anchor_ratio = run_cfg.anchor_ratio;
                    sopts.anchor_stride_scale = run_cfg.anchor_stride_scale;
                    sopts.mmr_threshold = run_cfg.mmr_threshold;
                    sopts.mmr_lambda = run_cfg.mmr_lambda;
                    sopts.phrase_weight = run_cfg.phrase_weight;
                    sopts.coverage_weight = run_cfg.coverage_weight;
                    sopts.prior_gate = run_cfg.prior_gate;
                    sopts.fill_priors = run_cfg.fill_priors;
                    sopts.structure_budget = run_cfg.structure_budget;
                    sopts.anchor_span_radius = run_cfg.anchor_span_radius;
                    sopts.coverage_reserve = run_cfg.coverage_reserve;
                    sopts.span_admission = run_cfg.span_admission;
                    sopts.span_min_tokens = run_cfg.span_min_tokens;
                    sopts.sentence_completion = run_cfg.sentence_completion;
                    sopts.span_max_tokens = run_cfg.span_max_tokens;
                    sopts.sentence_trim = run_cfg.sentence_trim;
                    sopts.position_compaction = run_cfg.position_compaction;
                    sopts.weights = run_cfg.weights;
                    // AR005: semantic-priority fusion defaults when the
                    // embedding engine is active and the user has not
                    // pinned weights (pure latency mode keeps the AR004
                    // defaults — the encoding cost is the point there).
                    if (embed.loaded() && mode != Mode::Latency &&
                        !cfg.weights_overridden) {
                        sopts.weights = semantic_priority_weights();
                    }
                    Selection selection;
                    BlockSelector selector;
                    EmbeddingSelectionScorer embed_adapter(
                            embed.loaded() ? &embed : nullptr, &pieces,
                            run_cfg.block_size);
                    // AR005 FR-5: selection-stage scoring time, accounted
                    // separately from prefill TTFT (net-cost honesty).
                    const auto sel_begin = std::chrono::steady_clock::now();
                    st = selector.select(tokens, c.question, pieces, sopts,
                                         embed.loaded() ? &embed_adapter : nullptr,
                                         &selection);
                    if (!st.ok()) continue;
                    // AR016 FR-3: latency-targeted budget. The measured cost
                    // model maps admitted tokens to TTFT; when the first
                    // selection overshoots the target, shrink the budget and
                    // re-select (selection is deterministic pure CPU work, so
                    // this cannot change the numerics of a given budget).
                    double sla_predicted_ms = 0.0;
                    double sla_plan_ms = 0.0;
                    std::string sla_budget_source;
                    // Prefill is sublinear, so a fitted intercept may be
                    // slightly negative; the model is usable as long as it has a
                    // positive slope and enough samples behind it.
                    if (cfg.budget_target_ms > 0.0 && !selection.dense &&
                        cfg.cost_b_ms_per_token > 0.0 && cfg.cost_n >= 5) {
                        const auto sla_begin = std::chrono::steady_clock::now();
                        CostModel model;
                        model.intercept_ms = cfg.cost_a_ms;
                        model.ms_per_token = cfg.cost_b_ms_per_token;
                        model.r2 = cfg.cost_r2;
                        model.n_samples = cfg.cost_n;
                        model.valid = true;
                        bool clamped = false;
                        for (int iter = 0; iter < 8; ++iter) {
                            const uint32_t admitted =
                                static_cast<uint32_t>(selection.token_indices.size());
                            sla_predicted_ms = predict_ttft_ms(model, admitted);
                            if (sla_predicted_ms <= cfg.budget_target_ms) break;
                            const uint32_t fit = max_tokens_within(
                                model, cfg.budget_target_ms, 8u, admitted);
                            if (fit == 0 || fit >= admitted) break;
                            const double next =
                                sopts.budget * static_cast<double>(fit) /
                                static_cast<double>(admitted);
                            // Keep shrinking while the conservative prediction
                            // still overshoots; 0.1% granularity is one
                            // admission token at these budgets.
                            if (!(next < sopts.budget * 0.999)) break;
                            if (sopts.budget <= 0.011) {
                                clamped = true;
                                break;
                            }
                            sopts.budget = std::max(0.01, next);
                            Selection retry;
                            st = selector.select(tokens, c.question, pieces, sopts,
                                                 embed.loaded() ? &embed_adapter : nullptr,
                                                 &retry);
                            if (!st.ok()) break;
                            selection = std::move(retry);
                            if (selection.dense) break;
                        }
                        // The loop may break on a plateau without re-evaluating
                        // the last selection; report the prediction of the
                        // selection we are actually going to run.
                        sla_predicted_ms = predict_ttft_ms(
                            model, static_cast<uint32_t>(selection.token_indices.size()));
                        sla_plan_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - sla_begin)
                                          .count();
                        sla_budget_source = clamped ? "cost-model-clamped" : "cost-model";
                    }
                    if (!run_cfg.dump_selection_dir.empty()) {
                        const std::string case_tag = c.id;
                        std::ofstream jout(run_cfg.dump_selection_dir + "/" +
                                           case_tag + "_selection.json");
                        if (jout) jout << selection.to_json() << '\n';
                        std::ofstream tout(run_cfg.dump_selection_dir + "/" +
                                           case_tag + "_selected.txt");
                        if (tout) {
                            tout << "# prompt_tokens=" << tokens.size()
                                 << " selected_tokens=" << selection.token_indices.size()
                                 << '\n';
                            uint32_t prev = 0;
                            for (size_t k = 0; k < selection.token_indices.size(); ++k) {
                                const uint32_t pos = selection.token_indices[k];
                                if (k == 0) {
                                    tout << "[pos " << pos << "] ";
                                } else if (pos != prev + 1) {
                                    tout << "\n[... " << (pos - prev - 1)
                                         << " dropped tokens]\n[pos " << pos << "] ";
                                }
                                tout << pieces[pos];
                                prev = pos;
                            }
                            tout << '\n';
                        }
                    }

                    // AR008 FR-2: measured combination routing. The selector
                    // run above doubles as the profiler: its phrase hits,
                    // discriminative-term count and lexical concentration are
                    // exactly the features the table was built from, and a
                    // selection pass costs milliseconds. The gate path does
                    // not need the selection, so a routed switch is free.
                    std::string portfolio_choice = selection.dense ? "dense" : "selection";
                    std::string portfolio_reason = "disabled";
                    // The qg_* knobs live on the BenchmarkConfig (they are
                    // not part of the Config used for planning), so read them
                    // from `cfg` and not from the per-cell `run_cfg`.
                    bool routed_qg_bound = cfg.qg_bound;
                    uint32_t routed_sel_pages = cfg.qg_sel_pages;
                    uint32_t routed_local_rows = cfg.qg_local_rows;
                    if (!run_cfg.portfolio_table.empty() && !selection.dense) {
                        PortfolioProfile profile;
                        profile.n_tokens = selection.total_tokens;
                        profile.specific_terms = selection.specific_terms;
                        profile.phrase_blocks = selection.phrase_blocks;
                        profile.lexical_concentration = selection.lexical_concentration;
                        PortfolioTable table;
                        auto pt = load_portfolio_cached(run_cfg.portfolio_table, &table);
                        if (!pt.ok()) {
                            portfolio_reason = "fallback:table-unusable";
                            log_warn("portfolio table unusable (" + pt.message +
                                     "); keeping selection");
                        } else if (!table.matches_model(model_hash())) {
                            portfolio_reason = "fallback:model-mismatch";
                            log_warn("portfolio table model fingerprint mismatch; keeping selection");
                        } else {
                            PortfolioRow row;
                            if (table.recommend(profile, &row)) {
                                portfolio_choice = row.combo;
                                portfolio_reason = "table:" + row.key;
                                if (cfg.qg_bound) {
                                    // Explicit user intent wins over the table.
                                    portfolio_choice = "qg_bound";
                                    portfolio_reason = "user-override";
                                } else if (row.combo == "gate-p16") {
                                    routed_qg_bound = true;
                                    routed_sel_pages = 16;
                                    routed_local_rows = 0;
                                } else if (row.combo == "gate-p24-local32") {
                                    routed_qg_bound = true;
                                    routed_sel_pages = 24;
                                    routed_local_rows = 32;
                                } else if (row.combo == "dense") {
                                    // A routed dense run still goes through the
                                    // selector for audit, but the prefill is dense.
                                    routed_qg_bound = false;
                                }
                            } else {
                                portfolio_reason = "fallback:no-match";
                            }
                        }
                    }

                    // Derive attention gather indices for non-dense plans;
                    // an empty result (covers-all or disabled) keeps the
                    // block-sparse path untouched.
                    if (plan.attention_top_k > 0 && !selection.dense) {
                        AttentionSparsityOptions aopts;
                        aopts.block_size = run_cfg.block_size;
                        aopts.top_k_blocks = plan.attention_top_k;
                        if (!run_cfg.attn_top_k_layers.empty()) {
                            aopts.per_layer = true;
                            std::vector<uint32_t> layers;
                            if (parse_u32_list(run_cfg.attn_top_k_layers, &layers).ok()) {
                                aopts.layer_top_k = std::move(layers);
                            }
                        }
                        std::vector<int32_t> attn_indices;
                        std::string attn_reason;
                        if (AttentionSparsityController::build(selection, aopts, &attn_indices,
                                                               &attn_reason)
                                .ok()) {
                            selection.attention_indices = std::move(attn_indices);
                            selection.attention_top_k =
                                static_cast<uint32_t>(selection.attention_indices.size());
                        }
                    }

                    // Interpretability hook: EDGE_PREFILL_DUMP_SELECTION=DIR
                    // writes the full selection JSON (per-block signals,
                    // kept blocks, gather indices) per case for offline
                    // analysis and plotting. Off by default.
                    if (const char* dump_dir = std::getenv("EDGE_PREFILL_DUMP_SELECTION")) {
                        std::error_code ec;
                        std::filesystem::create_directories(dump_dir, ec);
                        const std::string path = std::string(dump_dir) + "/" + c.id + "@" +
                                                 std::to_string(length) + ".json";
                        std::ofstream out(path);
                        out << selection.to_json() << '\n';
                    }

                    if (log_level() >= LogLevel::Debug && !selection.dense) {
                        size_t needle_token = tokens.size();
                        std::string acc;
                        const std::string needle = c.answer;
                        for (size_t i = 0; i < pieces.size() && !needle.empty(); ++i) {
                            acc += pieces[i];
                            if (acc.find(needle) != std::string::npos) {
                                needle_token = i;
                                break;
                            }
                        }
                        const bool in_selected =
                            std::binary_search(selection.token_indices.begin(),
                                               selection.token_indices.end(), needle_token);
                        std::string blocks;
                        for (uint32_t b : selection.mandatory_blocks) {
                            if (!blocks.empty()) blocks += ",";
                            blocks += std::to_string(b);
                        }
                        std::vector<std::pair<double, uint32_t>> top_lex;
                        for (uint32_t b = 0; b < selection.signals.size(); ++b) {
                            top_lex.emplace_back(selection.signals[b].lexical, b);
                        }
                        std::sort(top_lex.begin(), top_lex.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                        std::string lexical_blocks;
                        for (size_t i = 0; i < top_lex.size() && i < 6; ++i) {
                            if (!lexical_blocks.empty()) lexical_blocks += ",";
                            lexical_blocks += std::to_string(top_lex[i].second) + ":" +
                                              std::to_string(top_lex[i].first);
                        }
                        log_debug("needle_token=" + std::to_string(needle_token) +
                                  " block=" + std::to_string(needle_token / sopts.block_size) +
                                  " in_selected=" + (in_selected ? "yes" : "no") +
                                  " mandatory=[" + blocks + "] top_lex=[" + lexical_blocks + "]");
                    }

                    SparsePrefiller::Options popts;
                    // 0 = auto (align to n_ubatch) for both sparse and
                    // dense-fallback plans (AR004 FR-1).
                    popts.chunk_size = plan.chunk_size;
                    popts.enable_prefix_cache = false;
                    popts.enable_chunking = true;
                    popts.enable_ffn_skip = true;
                    popts.ffn_skip_mask = plan.dense_fallback ? 0 : plan.skip_ffn_mask;
                    // AR004 compressed branches ride on the scalar gather;
                    // the head path disables them when it activates below
                    // (graph-level exclusivity, v1).
                    popts.enable_pool_blocks = cfg.pool_branches;

                    // AR003/AR004 FR-4: per-head adaptive gather. Pattern
                    // routing replaces only the matrix; without a profile
                    // the pyramid prior supplies one over the same pool.
                    if (cfg.head_adaptive && head_ok && cfg.head_top_k > 0 &&
                        !selection.dense) {
                        HeadAdaptiveOptions hopts;
                        hopts.block_size = run_cfg.block_size;
                        hopts.top_k_rows = cfg.head_top_k;
                        std::vector<int32_t> pool;
                        std::vector<int32_t> matrix;
                        std::string head_reason;
                        bool built = AttentionSparsityController::build_head_adaptive(
                                         selection, head_profile, hopts, &pool, &matrix,
                                         &head_reason).ok() &&
                                     !pool.empty() && !matrix.empty();
                        if (built && cfg.head_pattern_routing) {
                            PatternRoutingOptions routing;
                            routing.k_normal = cfg.head_top_k;
                            matrix.clear();
                            built = AttentionSparsityController::build_pattern_matrix(
                                        head_profile, routing, &matrix,
                                        &head_reason).ok() &&
                                    !matrix.empty();
                        }
                        if (built) {
                            popts.enable_head_adaptive = true;
                            popts.head_pool = std::move(pool);
                            popts.head_matrix = std::move(matrix);
                            // The pooled compressed branch is exclusive with
                            // the per-head matrix path in the graph (v1).
                            popts.enable_pool_blocks = false;
                            log_debug(head_reason);
                        } else if (!head_reason.empty()) {
                            log_warn("head-adaptive build failed: " + head_reason);
                        }
                    } else if (cfg.head_adaptive && !head_ok && cfg.pyramid_prior &&
                               cfg.head_top_k > 0 && !selection.dense) {
                        std::vector<int32_t> pool;
                        std::vector<int32_t> matrix;
                        PyramidOptions pyopts;
                        pyopts.k_min = cfg.pyramid_k_min;
                        pyopts.k_max = cfg.pyramid_k_max;
                        if (AttentionSparsityController::build_ranked_pool(
                                    selection, run_cfg.block_size, &pool).ok() &&
                            !pool.empty() &&
                            AttentionSparsityController::build_pyramid_matrix(
                                    static_cast<uint32_t>(backend.n_layer()),
                                    static_cast<uint32_t>(backend.n_head()),
                                    pyopts, &matrix).ok() &&
                            !matrix.empty()) {
                            popts.enable_head_adaptive = true;
                            popts.head_pool = std::move(pool);
                            popts.head_matrix = std::move(matrix);
                            // Exclusive with the pooled branch (v1).
                            popts.enable_pool_blocks = false;
                            log_debug("pyramid prior: k(min/max)=" +
                                      std::to_string(pyopts.k_min) + "/" +
                                      std::to_string(pyopts.k_max));
                        } else {
                            log_warn("pyramid-prior build failed; scalar path");
                        }
                    }

                    // AR004 FR-3 / AR005 FR-2: query-chunk gating replaces
                    // the scalar gather; head matrix wins (exclusivity),
                    // pool rides only on the scalar gather. With the
                    // embedding engine active the plan upgrades to the
                    // per-group 2D matrix; degradation steps are audited.
                    // AR006: bound gating takes precedence and needs no
                    // plan-level row sets at all.
                    std::string qg_source;
                    if (routed_qg_bound && !popts.enable_head_adaptive &&
                        !selection.dense) {
                        popts.enable_qg_bound = true;
                        popts.qg_sel_pages = routed_sel_pages;
                        popts.qg_local_rows = routed_local_rows;
                        popts.qg_pool_pages = cfg.qg_pool_pages;
                        popts.enable_query_gating = false;
                        popts.enable_pool_blocks = false;
                        qg_source = "attention-bound";
                        log_debug("bound gating plan: sel_pages=" +
                                  std::to_string(popts.qg_sel_pages) +
                                  " local_rows=" +
                                  std::to_string(cfg.qg_local_rows));
                    } else if (cfg.query_gating && !popts.enable_head_adaptive &&
                        !selection.dense &&
                        selection.per_pos_scores.size() == selection.total_tokens) {
                        QueryGatingOptions qg_opts;
                        qg_opts.query_chunk_size = cfg.query_chunk_size;
                        qg_opts.query_chunk_max = cfg.query_chunk_max;
                        qg_opts.top_k_per_group = cfg.query_top_k_per_group;
                        if (embed.loaded()) {
                            // AR005 FR-2: group texts are the raw
                            // contiguous slices of the group token spans;
                            // block texts the original 32-token blocks.
                            std::vector<std::pair<uint32_t, uint32_t>> spans;
                            std::string span_reason;
                            if (AttentionSparsityController::build_group_spans(
                                        selection, qg_opts, &spans, &span_reason)
                                    .ok() &&
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
                                    const uint32_t end = std::min(
                                            selection.total_tokens, begin + bs);
                                    for (uint32_t i = begin; i < end; ++i) {
                                        block_texts[b] += pieces[i];
                                    }
                                }
                                std::vector<std::vector<double>> matrix;
                                auto score_st = embed.score_group_block(
                                        group_texts, block_texts, &matrix);
                                if (score_st.ok() && matrix.size() == spans.size()) {
                                    selection.per_group_block_scores =
                                            std::move(matrix);
                                } else {
                                    log_warn("semantic qg matrix unavailable (" +
                                             score_st.message +
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
                                        : (selection.total_tokens +
                                           run_cfg.block_size - 1) /
                                                  run_cfg.block_size;
                        std::string qg_reason;
                        auto qst = AttentionSparsityController::build_query_gated(
                                selection, selection.per_pos_scores,
                                selection.per_group_block_scores, n_prompt_blocks,
                                qg_opts, &popts.qg_group_rows, &popts.qg_bounds,
                                &qg_reason);
                        if (!qst.ok()) {
                            log_warn("query gating plan rejected: " + qst.message);
                        } else if (popts.qg_group_rows.empty()) {
                            log_debug("query gating: none (" + qg_reason + ")");
                        } else {
                            popts.enable_query_gating = true;
                            popts.enable_pool_blocks = false;
                            qg_source = selection.per_group_block_scores.empty()
                                                ? "lexical-1d"
                                                : "embedding-2d";
                            log_debug(qg_reason);
                        }
                    }

                    // AR005 FR-5: the selection-stage clock stops after the
                    // 2D group-block matrix build (when armed): the matrix is
                    // selection work, and charging it to prefill would
                    // fabricate speedup. semantic_score_ms therefore covers
                    // select() + any embedding-side matrix construction.
                    const double select_ms =
                            std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - sel_begin)
                                    .count();

                    // AR013 FR-1/FR-2: size the context to what this case will
                    // actually write. With compacted positions the span follows
                    // the admitted tokens, so a 4K/64K document no longer forces
                    // a full-length KV cache. Growth is monotonic and rare, and
                    // an explicit n_ctx that is too small is a hard error rather
                    // than a silent truncation.
                    const uint32_t last_pos =
                            selection.positions.empty()
                                    ? (selection.token_indices.empty()
                                               ? 0
                                               : selection.token_indices.back())
                                    : selection.positions.back();
                    const uint32_t required_ctx = last_pos + base.max_new_tokens + 8;
                    if (cfg.n_ctx != 0 && required_ctx > cfg.n_ctx) {
                        if (warnings) {
                            warnings->push_back("n-ctx-too-small:" + c.id + "@" +
                                                std::to_string(length));
                        }
                        log_error("n_ctx " + std::to_string(cfg.n_ctx) +
                                  " is smaller than the required " +
                                  std::to_string(required_ctx) +
                                  " for case " + c.id + "@" + std::to_string(length));
                        continue;
                    }
                    {
                        auto gst = backend.ensure_context(required_ctx);
                        if (!gst.ok()) {
                            if (warnings) {
                                warnings->push_back("context-grow-failed:" + c.id + "@" +
                                                    std::to_string(length) + ":" +
                                                    gst.message);
                            }
                            log_warn("context grow failed for " + c.id + ": " +
                                     gst.message);
                            continue;
                        }
                    }

                    // Untimed warmup runs stabilize caches and page in model
                    // weights so the timed repeats measure steady state.
                    bool prefill_ok = true;
                    for (uint32_t w = 0; w < base.warmup; ++w) {
                        PrefillStats warm_stats;
                        if (!prefilter.prefill(tokens, selection, popts, &warm_stats).ok()) {
                            prefill_ok = false;
                            break;
                        }
                    }
                    if (!prefill_ok) {
                        if (warnings) warnings->push_back("prefill-failed:" + c.id);
                        continue;
                    }

                    // AR015: content-addressed snapshot. The key is derived
                    // from exactly the inputs the prefill sees (model, policy,
                    // tokens, positions, context size, KV precision), so a hit
                    // provably refers to the same sequence. The cold arm
                    // recomputes and saves; the warm arm loads and skips the
                    // prefill entirely.
                    SnapshotKey snapshot_key;
                    // The KV holds only the admitted tokens, so the snapshot is
                    // keyed by those (not by the whole prompt): two prompts that
                    // admit the same evidence share one snapshot.
                    std::vector<llama_token> admitted_tokens;
                    if (snapshot_store != nullptr) {
                        admitted_tokens.reserve(selection.token_indices.size());
                        for (uint32_t index : selection.token_indices) {
                            if (index < tokens.size()) admitted_tokens.push_back(tokens[index]);
                        }
                    }
                    const bool snapshot_enabled = snapshot_store != nullptr;
                    if (snapshot_enabled) {
                        snapshot_key.model_key = backend.fingerprint().to_key();
                        snapshot_key.policy_key = admission_policy_key(base, plan);
                        snapshot_key.kv_type = kv_precision_name(base.kv_type);
                        snapshot_key.n_tokens = static_cast<uint32_t>(admitted_tokens.size());
                        snapshot_key.n_ctx = llama_n_ctx(backend.context());
                        snapshot_key.token_hash = hash_tokens(admitted_tokens);
                        // AR013 may leave positions implicit (identity) when
                        // compaction is off; hash what the decode will use.
                        std::vector<llama_pos> effective_positions;
                        if (selection.positions.empty()) {
                            effective_positions.reserve(selection.token_indices.size());
                            for (uint32_t index : selection.token_indices) {
                                effective_positions.push_back(static_cast<llama_pos>(index));
                            }
                        } else {
                            effective_positions.reserve(selection.positions.size());
                            for (uint32_t position : selection.positions) {
                                effective_positions.push_back(static_cast<llama_pos>(position));
                            }
                        }
                        snapshot_key.position_hash = hash_positions(effective_positions);
                    }
                    SnapshotPayload snapshot_payload;
                    SnapshotLoadResult snapshot_load;
                    SnapshotSaveResult snapshot_save;
                    bool from_snapshot = false;
                    if (snapshot_enabled && base.snapshot_use != 0) {
                        // Probe once to decide the arm (hit or not). The timed
                        // loop below redoes the whole load so TTFT includes the
                        // read + verify, not just the memcpy.
                        snapshot_load = snapshot_store->load(snapshot_key, admitted_tokens,
                                                             &snapshot_payload);
                        if (snapshot_load.hit) {
                            const auto lst = backend.load_state(snapshot_payload.state);
                            from_snapshot = lst.ok();
                            if (!from_snapshot) snapshot_load.state = "miss:load";
                        }
                    }

                    const uint32_t reps = std::max<uint32_t>(1, base.repeats);
                    std::vector<double> ttft_samples;
                    PrefillStats stats;
                    const auto last_begin = std::chrono::steady_clock::now();
                    for (uint32_t r = 0; r < reps; ++r) {
                        const auto begin = std::chrono::steady_clock::now();
                        if (from_snapshot) {
                            // A warm request reads, verifies and restores; all
                            // three are inside the timer so the number is the
                            // latency a caller would actually see.
                            const SnapshotLoadResult again = snapshot_store->load(
                                    snapshot_key, admitted_tokens, &snapshot_payload);
                            if (!again.hit) {
                                st = Status::error(ErrorCode::Io,
                                                   "snapshot vanished between probe and timed load");
                            } else {
                                st = backend.load_state(snapshot_payload.state);
                                snapshot_load = again;
                            }
                        } else {
                            st = prefilter.prefill(tokens, selection, popts, &stats);
                        }
                        const auto first = std::chrono::steady_clock::now();
                        if (!st.ok()) {
                            prefill_ok = false;
                            break;
                        }
                        ttft_samples.push_back(
                            std::chrono::duration<double, std::milli>(first - begin).count());
                    }
                    if (!prefill_ok) {
                        if (warnings) warnings->push_back("prefill-failed:" + c.id);
                        continue;
                    }
                    // Median TTFT: robust to scheduler noise on either side.
                    std::sort(ttft_samples.begin(), ttft_samples.end());
                    const double ttft_ms = ttft_samples[ttft_samples.size() / 2];
                    const double ttft_min_ms = ttft_samples.front();

                    if (from_snapshot) {
                        // Rebuild the bookkeeping the rest of the loop reads
                        // from the restored state instead of from prefill.
                        auto* loaded_mem = llama_get_memory(backend.context());
                        stats.kv_pos_max = llama_memory_seq_pos_max(loaded_mem, 0);
                        stats.kv_pos_min = llama_memory_seq_pos_min(loaded_mem, 0);
                        stats.prompt_tokens = static_cast<uint32_t>(tokens.size());
                        stats.selected_tokens = snapshot_key.n_tokens;
                        stats.prompt_tokens = static_cast<uint32_t>(tokens.size());
                        stats.total_us = static_cast<uint64_t>(ttft_min_ms * 1000.0);
                        stats.ttft_us = stats.total_us;
                        stats.kv_bytes = snapshot_payload.state.size();
                    }

                    std::vector<float> first_logits_owned;
                    if (from_snapshot) {
                        first_logits_owned = snapshot_payload.logits;
                    } else {
                        const float* raw = llama_get_logits_ith(backend.context(), -1);
                        if (raw) {
                            first_logits_owned.assign(raw, raw + backend.n_vocab());
                        }
                    }
                    const float* first_logits =
                            first_logits_owned.empty() ? nullptr : first_logits_owned.data();

                    // AR015: the cold arm persists the prompt KV *before* any
                    // generation token is written, so the stored state is the
                    // prompt alone and the stored logits are the first-token
                    // distribution the cold path is about to use.
                    if (snapshot_enabled && !from_snapshot && first_logits != nullptr) {
                        SnapshotPayload out_payload;
                        if (backend.save_state(&out_payload.state).ok()) {
                            out_payload.logits.assign(first_logits,
                                                      first_logits + backend.n_vocab());
                            out_payload.n_vocab =
                                    static_cast<uint32_t>(out_payload.logits.size());
                            snapshot_save = snapshot_store->save(snapshot_key, admitted_tokens,
                                                                 out_payload);
                        }
                    }

                    std::string generated;
                    double quality = 0.0;
                    double sample_ppl_dense = 0.0;
                    double sample_ppl_sparse = 0.0;
                    if (c.task == "ppl") {
                        // AR004 FR-5: quality is the retention ratio
                        // min(1, ppl_dense/ppl_sparse) =
                        // min(1, exp(lp_sparse - lp_dense)). The raw
                        // geometric-mean probability is ~1e-2..1e-4 and
                        // unusable as a quality number; the raw perplexities
                        // land in the CSV audit columns.
                        std::vector<llama_token> answer_tokens;
                        double lp_sparse = 0.0;
                        if (!backend.tokenize(c.answer, &answer_tokens, false, false).ok() ||
                            answer_tokens.empty() ||
                            !score_answer_mean_logprob(backend, first_logits,
                                                   static_cast<size_t>(stats.kv_pos_max) + 1,
                                                       answer_tokens, &lp_sparse)) {
                            if (warnings) warnings->push_back("ppl-score-failed:" + c.id);
                            continue;
                        }
                        quality = std::min(1.0, std::exp(lp_sparse - ppl_lp_dense));
                        sample_ppl_dense = std::exp(-ppl_lp_dense);
                        sample_ppl_sparse = std::exp(-lp_sparse);
                    } else if (first_logits) {
                        std::vector<float> logits(first_logits,
                                                  first_logits + backend.n_vocab());
                        llama_token next = greedy(logits);
                        for (uint32_t i = 0; i < base.max_new_tokens; ++i) {
                            std::string piece;
                            backend.token_to_piece(next, &piece);
                            generated += piece;
                            if (next == llama_vocab_eos(backend.vocab())) break;
                            std::vector<float> step_logits;
                            st = backend.decode(
                                    next,
                                    static_cast<llama_pos>(stats.kv_pos_max) + 1 +
                                            static_cast<llama_pos>(i),
                                    &step_logits);
                            if (!st.ok()) break;
                            next = greedy(step_logits);
                        }
                        quality = f1_score(generated, c.answer);
                    }
                    const auto end = std::chrono::steady_clock::now();
                    // E2E covers the last timed repeat end to end: its prefill
                    // plus the greedy generation that follows.
                    const double e2e_ms =
                        std::chrono::duration<double, std::milli>(end - last_begin).count();

                    Sample sample;
                    sample.task = c.task;
                    sample.case_id = c.id + "@" + std::to_string(length);
                    sample.ttft_ms = ttft_ms;
                    sample.ttft_min_ms = ttft_min_ms;
                    sample.e2e_ms = e2e_ms;
                    sample.prefill_tokens_per_s =
                        stats.total_us > 0
                            ? static_cast<double>(stats.selected_tokens) * 1e6 / stats.total_us
                            : 0.0;
                    sample.kv_bytes = stats.kv_bytes;
                    sample.quality = quality;
                    sample.dense = selection.dense;
                    sample.prompt_tokens = stats.prompt_tokens;
                    sample.selected_tokens = stats.selected_tokens;
                    sample.mode = mode_name(mode);
                    sample.budget = sopts.budget;
                    sample.attn_top_k = static_cast<uint32_t>(std::max<int32_t>(0, stats.attn_top_k));
                    sample.repeats = reps;
                    sample.ppl_dense = sample_ppl_dense;
                    sample.ppl_sparse = sample_ppl_sparse;
                    sample.query_gating = stats.query_gating;
                    sample.qg_groups = stats.qg_groups;
                    // AR005 FR-5 audit columns.
                    sample.score_source = selection.score_source;
                    sample.semantic_score_ms = embed.loaded() ? select_ms : 0.0;
                    sample.draft_score_ms = 0.0;
                    sample.qg_source = qg_source;
                    sample.budget_source =
                        sla_budget_source.empty() ? plan.budget_source : sla_budget_source;
                    sample.predicted_ttft_ms = sla_predicted_ms;
                    sample.plan_select_ms = sla_plan_ms;
                    sample.anchor_blocks = selection.anchor_blocks;
                    sample.phrase_blocks = selection.phrase_blocks;
                    sample.specific_terms = selection.specific_terms;
                    sample.coverage_queries = selection.coverage_queries;
                    sample.evidence_blocks = selection.evidence_blocks;
                    sample.stop_reason = selection.stop_reason;
                    sample.structure_tokens = selection.structure_tokens;
                    sample.evidence_tokens = selection.evidence_tokens;
                    sample.span_tokens = selection.span_tokens;
                    sample.sentences_completed = selection.sentences_completed;
                    sample.sentences_extended = selection.sentences_extended;
                    sample.uncovered_facts = selection.uncovered_facts;
                    sample.portfolio_choice = portfolio_choice;
                    sample.portfolio_reason = portfolio_reason;
                    sample.mmr_penalized = selection.mmr_penalized;
                    // AR013 FR-1: per-case resource audit. The allocation is
                    // read after the prefill so a grown context is reflected.
                    sample.ctx_cells = backend.n_ctx();
                    sample.kv_alloc_bytes = backend.kv_alloc_bytes();
                    sample.kv_used_bytes = sample.kv_bytes;
                    sample.rss_mb = backend.rss_mb();
                    sample.position_span = selection.position_span;
                    sample.compaction_reason = selection.compaction_reason;
                    // AR015 audit: whether this sample came from a snapshot and
                    // what it cost, so the gate can tell warm from cold without
                    // trusting timings alone.
                    sample.snapshot_hit = from_snapshot ? 1 : 0;
                    sample.snapshot_state = !snapshot_enabled
                                                    ? "off"
                                                    : (from_snapshot ? snapshot_load.state
                                                                     : (base.snapshot_use != 0
                                                                                ? snapshot_load.state
                                                                                : "cold"));
                    sample.snapshot_bytes = from_snapshot ? snapshot_load.bytes : snapshot_save.bytes;
                    sample.snapshot_load_ms = from_snapshot ? snapshot_load.load_ms : 0.0;
                    sample.snapshot_save_ms = snapshot_save.ok ? snapshot_save.save_ms : 0.0;
                    sample.snapshot_key =
                            snapshot_enabled ? snapshot_key.hex().substr(0, 16) : std::string();
                    log_info("case=" + sample.case_id + " mode=" + sample.mode +
                             " quality=" + std::to_string(sample.quality) +
                             " generated=" + generated);
                    out->push_back(sample);
                }
            }
        }
    }

    // AR016 FR-1: stamp the verified execution plan on every sample. This is
    // run-level information, so it is applied once at the end instead of being
    // copied through every construction site.
    if (out) {
        for (auto& sample : *out) {
            sample.device = device_name;
            sample.gpu_layers_requested = placement.requested_layers;
            sample.gpu_layers_offloaded = placement.offloaded_layers;
            sample.gpu_layers_total = placement.total_layers;
            sample.n_threads = backend.context_options().n_threads;
            sample.n_threads_batch = backend.context_options().n_threads_batch;
            sample.n_ubatch = backend.context_options().n_ubatch;
            sample.plan_source = cfg.plan_source ? cfg.plan_source : "default";
            sample.device_state = device_state;
            if (sample.n_threads <= 0) {
                const unsigned hw = std::thread::hardware_concurrency();
                sample.n_threads = static_cast<int32_t>(hw > 0 ? hw : 4);
            }
            if (sample.n_threads_batch <= 0) {
                sample.n_threads_batch = sample.n_threads;
            }
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    log_info("benchmark wall time: " +
             std::to_string(std::chrono::duration<double>(wall_end - wall_begin).count()) + " s");
    LlamaBackend::shutdown();
    return Status::success();
}

} // namespace edge_prefill
