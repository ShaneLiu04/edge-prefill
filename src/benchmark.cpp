#include "edge_prefill/benchmark.h"

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
#include <unordered_map>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/backend.h"
#include "edge_prefill/head_profile.h"
#include "edge_prefill/log.h"
#include "edge_prefill/planner.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/sparse_prefill.h"

namespace edge_prefill {
namespace {

struct Case {
    std::string id;
    std::string task;
    std::string context_path;
    std::string question;
    std::string answer;
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
        cases.push_back(Case{fields[0], fields[1], fields[2], fields[3], fields[4]});
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

    LlamaBackend backend;
    auto st = LlamaBackend::init();
    if (!st.ok()) return st;
    backend.set_n_gpu_layers(cfg.n_gpu_layers);
    st = backend.load_model(base.model_path);
    if (!st.ok()) {
        LlamaBackend::shutdown();
        return st;
    }
    ContextOptions ctx_opts;
    ctx_opts.n_ctx = 0;
    ctx_opts.n_ubatch = 512;
    ctx_opts.n_threads = 0;
    ctx_opts.n_threads_batch = 0;
    ctx_opts.type_k = KvPrecision::F16;
    ctx_opts.type_v = KvPrecision::F16;
    // The attention gather paths (scalar top-k and per-head matrix) require
    // the non-flash layout; keep every mode on the same graph family so
    // dense-vs-sparse timings stay internally comparable.
    ctx_opts.flash_attn = false;
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
            const std::string prompt = build_prompt(backend.model(), base.use_chat_template,
                                                    context, c.question);
            std::vector<llama_token> tokens;
            st = backend.tokenize(prompt, &tokens, true, true);
            if (!st.ok()) continue;
            if (tokens.size() + base.max_new_tokens + 8 > backend.n_ctx()) {
                if (warnings) warnings->push_back("context-too-long:" + c.id);
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
                        if (!prefilter.prefill(tokens, dense_sel, ref_opts, &ref_stats).ok()) {
                            if (warnings) warnings->push_back("ppl-dense-ref-failed:" + c.id);
                        } else {
                            const float* ref_logits = llama_get_logits_ith(backend.context(), -1);
                            double lp = 0.0;
                            if (!score_answer_mean_logprob(backend, ref_logits, tokens.size(),
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
                    run_cfg.budget = budget;
                    run_cfg.budget_overridden = false;
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
                    sopts.weights = run_cfg.weights;
                    Selection selection;
                    BlockSelector selector;
                    st = selector.select(tokens, c.question, pieces, sopts, nullptr, &selection);
                    if (!st.ok()) continue;

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

                    const uint32_t reps = std::max<uint32_t>(1, base.repeats);
                    std::vector<double> ttft_samples;
                    PrefillStats stats;
                    const auto last_begin = std::chrono::steady_clock::now();
                    for (uint32_t r = 0; r < reps; ++r) {
                        const auto begin = std::chrono::steady_clock::now();
                        st = prefilter.prefill(tokens, selection, popts, &stats);
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

                    const float* first_logits = llama_get_logits_ith(backend.context(), -1);
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
                            !score_answer_mean_logprob(backend, first_logits, tokens.size(),
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
                            st = backend.decode(next, static_cast<llama_pos>(tokens.size() + i),
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
                    log_info("case=" + sample.case_id + " mode=" + sample.mode +
                             " quality=" + std::to_string(sample.quality) +
                             " generated=" + generated);
                    out->push_back(sample);
                }
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
