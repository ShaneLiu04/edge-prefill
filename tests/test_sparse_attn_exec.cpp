#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/backend.h"
#include "edge_prefill/benchmark.h"
#include "edge_prefill/config.h"
#include "edge_prefill/metrics.h"
#include "edge_prefill/planner.h"
#include "edge_prefill/prefix_cache.h"
#include "edge_prefill/selector.h"
#include "edge_prefill/sparse_prefill.h"

using namespace edge_prefill;

namespace {

std::string test_model_path() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

struct Loaded {
    LlamaBackend backend;
    std::vector<llama_token> tokens;
};

bool load(Loaded* out) {
    const std::string path = test_model_path();
    if (path.empty()) return false;
    if (!LlamaBackend::init().ok()) return false;
    if (!out->backend.load_model(path).ok()) return false;
    ContextOptions opts;
    opts.n_ctx = 2048;
    opts.n_batch = 512;
    opts.n_ubatch = 128;
    opts.n_threads = 4;
    opts.n_threads_batch = 4;
    // The attention top-k gather path requires non-flash attention
    // (ggml_get_rows produces F32 masks; flash attention requires F16).
    opts.flash_attn = false;
    if (!out->backend.create_context(opts).ok()) return false;

    std::string text;
    for (int i = 0; i < 120; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    if (!out->backend.tokenize(text, &out->tokens).ok()) return false;
    if (out->tokens.size() > 512) out->tokens.resize(512);
    return out->tokens.size() >= 256;
}

std::vector<float> logits_of(const LlamaBackend& backend) {
    const float* data = llama_get_logits_ith(backend.context(), -1);
    if (!data) return {};
    return std::vector<float>(data, data + backend.n_vocab());
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

bool all_finite(const std::vector<float>& values) {
    for (float v : values) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

// Mirrors BenchmarkRunner's prompt construction (chat template with plain
// fallback) so the expected attention index count in the benchmark test is
// computed over byte-identical tokens.
std::string build_prompt(const llama_model* model, const std::string& context,
                         const std::string& question) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        llama_chat_message messages[2];
        messages[0].role = "system";
        messages[0].content = "You answer questions using only the provided context.";
        messages[1].role = "user";
        const std::string user =
            "Context:\n" + context + "\n\nQuestion: " + question +
            "\nAnswer with the shortest exact span.";
        messages[1].content = user.c_str();
        std::vector<char> buf(2 * (user.size() + 128));
        const int32_t len = llama_chat_apply_template(tmpl, messages, 2, true, buf.data(),
                                                      static_cast<int32_t>(buf.size()));
        if (len > 0 && static_cast<size_t>(len) <= buf.size()) {
            return std::string(buf.data(), static_cast<size_t>(len));
        }
    }
    return "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:";
}

struct SparseScenario {
    Selection selection;
    std::vector<int32_t> indices;
};

// Loads the model and derives a non-dense selection plus non-empty gather
// indices, the shared fixture of the execution tests. Returns false when the
// real model is unavailable (caller prints SKIP).
bool setup_sparse_scenario(Loaded* loaded, SparseScenario* out) {
    if (!load(loaded)) return false;
    SelectionOptions sopts;
    sopts.block_size = 32;
    sopts.budget = 0.5;
    sopts.min_dense_tokens = 256;
    std::vector<std::string> pieces(loaded->tokens.size());
    for (size_t i = 0; i < loaded->tokens.size(); ++i) {
        CHECK_OK(loaded->backend.token_to_piece(loaded->tokens[i], &pieces[i]));
    }
    BlockSelector selector;
    CHECK_OK(selector.select(loaded->tokens, "fox rain roof", pieces, sopts, nullptr,
                             &out->selection));
    CHECK(!out->selection.dense);

    AttentionSparsityOptions aopts;
    aopts.block_size = 32;
    // Small relative to the ~8 blocks a 0.5 budget keeps out of 16: a cap
    // that covers every selected block legitimately returns empty indices
    // (covers-all), which would silently degrade this fixture to SKIP.
    aopts.top_k_blocks = 2;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build(out->selection, aopts, &out->indices, &reason));
    // The model is loaded here, so an empty result is a degenerate scenario,
    // not an unavailable model: fail loudly instead of skipping.
    CHECK(!out->indices.empty());
    return !out->indices.empty();
}

SparsePrefiller::Options exec_options() {
    SparsePrefiller::Options popts;
    popts.chunk_size = 128;
    popts.enable_attn_sparse = true;
    popts.enable_prefix_cache = false;
    popts.enable_ffn_skip = false;
    return popts;
}

} // namespace

TEST_CASE(attn_chunk_indices_prefix) {
    const std::vector<int32_t> indices = {3, 10, 50, 100};
    std::vector<int32_t> allowed;

    AttentionSparsityController::chunk_indices(indices, 5, &allowed);
    const std::vector<int32_t> only_three = {3};
    CHECK_EQ(allowed, only_three);

    AttentionSparsityController::chunk_indices(indices, 2, &allowed);
    CHECK(allowed.empty());

    AttentionSparsityController::chunk_indices(indices, 100, &allowed);
    CHECK_EQ(allowed, indices);

    AttentionSparsityController::chunk_indices(indices, 1000, &allowed);
    CHECK_EQ(allowed, indices);

    AttentionSparsityController::chunk_indices(indices, 50, &allowed);
    for (size_t i = 0; i < allowed.size(); ++i) {
        CHECK(allowed[i] <= 50);
        if (i > 0) CHECK(allowed[i] > allowed[i - 1]);
    }
}

TEST_CASE(attn_validate_indices_rules) {
    Selection full;
    full.total_tokens = 256;
    full.token_indices.resize(256);
    std::iota(full.token_indices.begin(), full.token_indices.end(), 0u);

    const std::vector<int32_t> legal = {0, 5, 100, 255};
    CHECK_OK(AttentionSparsityController::validate_indices(legal, full));

    const std::vector<int32_t> duplicate = {3, 3, 7};
    CHECK_ERR(AttentionSparsityController::validate_indices(duplicate, full),
              ErrorCode::InvalidArgument);

    const std::vector<int32_t> not_ascending = {10, 5};
    CHECK_ERR(AttentionSparsityController::validate_indices(not_ascending, full),
              ErrorCode::InvalidArgument);

    const std::vector<int32_t> too_large = {0, 256};
    CHECK_ERR(AttentionSparsityController::validate_indices(too_large, full),
              ErrorCode::InvalidArgument);

    const std::vector<int32_t> negative = {-1, 5};
    CHECK_ERR(AttentionSparsityController::validate_indices(negative, full),
              ErrorCode::InvalidArgument);

    const std::vector<int32_t> missing_last = {0, 5, 100};
    CHECK_ERR(AttentionSparsityController::validate_indices(missing_last, full),
              ErrorCode::InvalidArgument);

    // A selection that keeps only a prefix of the tokens plus the last one:
    // in-range positions outside token_indices are rejected, and the last
    // prompt token remains mandatory.
    Selection subset;
    subset.total_tokens = 256;
    subset.token_indices.resize(128);
    std::iota(subset.token_indices.begin(), subset.token_indices.end(), 0u);
    subset.token_indices.push_back(255u);

    const std::vector<int32_t> subset_legal = {0, 64, 127, 255};
    CHECK_OK(AttentionSparsityController::validate_indices(subset_legal, subset));

    const std::vector<int32_t> not_selected = {0, 200};
    CHECK_ERR(AttentionSparsityController::validate_indices(not_selected, subset),
              ErrorCode::InvalidArgument);

    const std::vector<int32_t> subset_missing_last = {0, 64, 127};
    CHECK_ERR(AttentionSparsityController::validate_indices(subset_missing_last, subset),
              ErrorCode::InvalidArgument);
}

TEST_CASE(sparse_attn_execution_engages) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP sparse_attn_execution_engages (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    Selection selection = scenario.selection;
    selection.attention_indices = scenario.indices;
    selection.attention_top_k = static_cast<uint32_t>(scenario.indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();

    PrefillStats stats;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats));
    CHECK_EQ(stats.attn_top_k, static_cast<int32_t>(scenario.indices.size()));
    CHECK_EQ(stats.attn_indices, static_cast<uint32_t>(scenario.indices.size()));
    CHECK(!stats.attn_fallback);
    CHECK(stats.chunks >= 1);
    CHECK(!stats.chunk_stats.empty());
    // Per-chunk allowed indices are prefixes of one ascending sequence, so
    // the gathered row count can only grow across chunks.
    uint32_t prev_rows = 0;
    for (const ChunkStats& chunk : stats.chunk_stats) {
        CHECK(chunk.attn_rows >= prev_rows);
        prev_rows = chunk.attn_rows;
    }
    CHECK_EQ(stats.chunk_stats.back().attn_rows,
             static_cast<uint32_t>(scenario.indices.size()));
    CHECK_EQ(stats.kv_pos_max, static_cast<int32_t>(loaded.tokens.size()) - 1);
    CHECK(stats.selected_tokens < stats.prompt_tokens);

    const std::vector<float> first_logits = logits_of(loaded.backend);
    CHECK(!first_logits.empty());
    CHECK(all_finite(first_logits));

    // Greedy prefill is deterministic: an identical rerun must pick the
    // same first token.
    PrefillStats stats2;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats2));
    const std::vector<float> second_logits = logits_of(loaded.backend);
    CHECK(!second_logits.empty());
    CHECK_EQ(argmax(first_logits), argmax(second_logits));

    LlamaBackend::shutdown();
}

TEST_CASE(sparse_attn_execution_late_only_indices) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP sparse_attn_execution_late_only_indices (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    Selection selection = scenario.selection;
    CHECK(selection.token_indices.size() >= 32);
    // Only the last 32 selected positions: the early chunks have no allowed
    // rows, so the per-chunk adjustment must disable gather there instead of
    // reading unwritten KV cells.
    std::vector<int32_t> late(selection.token_indices.end() - 32,
                              selection.token_indices.end());
    CHECK_EQ(late.back(), static_cast<int32_t>(selection.total_tokens - 1));
    selection.attention_indices = late;
    selection.attention_top_k = static_cast<uint32_t>(late.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();

    PrefillStats stats;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats));
    CHECK_EQ(stats.attn_top_k, 32);
    CHECK_EQ(stats.attn_indices, 32u);
    CHECK(!stats.attn_fallback);
    CHECK(!stats.chunk_stats.empty());
    CHECK_EQ(stats.chunk_stats.front().attn_rows, 0u);

    const std::vector<float> logits = logits_of(loaded.backend);
    CHECK(!logits.empty());
    CHECK(all_finite(logits));

    LlamaBackend::shutdown();
}

TEST_CASE(sparse_attn_execution_disabled_equivalence) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP sparse_attn_execution_disabled_equivalence (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();

    // Explicitly disabled: non-empty attention_indices must be ignored.
    Selection with_indices = scenario.selection;
    with_indices.attention_indices = scenario.indices;
    with_indices.attention_top_k = static_cast<uint32_t>(scenario.indices.size());
    popts.enable_attn_sparse = false;
    PrefillStats stats_disabled;
    CHECK_OK(prefiller.prefill(loaded.tokens, with_indices, popts, &stats_disabled));
    CHECK_EQ(stats_disabled.attn_top_k, 0);
    CHECK_EQ(stats_disabled.attn_indices, 0u);
    const std::vector<float> l1 = logits_of(loaded.backend);

    // Enabled but empty indices: the same execution path as disabled.
    Selection without_indices = scenario.selection;
    without_indices.attention_indices.clear();
    without_indices.attention_top_k = 0;
    popts.enable_attn_sparse = true;
    PrefillStats stats_enabled;
    CHECK_OK(prefiller.prefill(loaded.tokens, without_indices, popts, &stats_enabled));
    CHECK_EQ(stats_enabled.attn_top_k, 0);
    CHECK_EQ(stats_enabled.attn_indices, 0u);
    const std::vector<float> l2 = logits_of(loaded.backend);

    CHECK(!l1.empty());
    CHECK(!l2.empty());
    CHECK(max_abs_diff(l1, l2) < 1e-5);

    LlamaBackend::shutdown();
}

TEST_CASE(sparse_attn_execution_invalid_rejected) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP sparse_attn_execution_invalid_rejected (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    Selection base = scenario.selection;
    base.attention_indices = scenario.indices;
    base.attention_top_k = static_cast<uint32_t>(scenario.indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    PrefillStats stats;

    Selection out_of_range = base;
    out_of_range.attention_indices.push_back(
        static_cast<int32_t>(base.total_tokens) + 7);
    CHECK_ERR(prefiller.prefill(loaded.tokens, out_of_range, popts, &stats),
              ErrorCode::InvalidArgument);

    Selection reversed = base;
    std::reverse(reversed.attention_indices.begin(), reversed.attention_indices.end());
    CHECK_ERR(prefiller.prefill(loaded.tokens, reversed, popts, &stats),
              ErrorCode::InvalidArgument);

    LlamaBackend::shutdown();
}

TEST_CASE(sparse_attn_execution_dense_ignored) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP sparse_attn_execution_dense_ignored (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    // A prompt below min_dense_tokens with budget 1.0 yields a dense
    // selection, which must never engage attention gather.
    std::vector<llama_token> short_tokens(loaded.tokens.begin(),
                                          loaded.tokens.begin() + 200);
    std::vector<std::string> pieces(short_tokens.size());
    for (size_t i = 0; i < short_tokens.size(); ++i) {
        CHECK_OK(loaded.backend.token_to_piece(short_tokens[i], &pieces[i]));
    }

    SelectionOptions sopts;
    sopts.block_size = 32;
    sopts.budget = 1.0;
    sopts.min_dense_tokens = 512;
    Selection selection;
    BlockSelector selector;
    CHECK_OK(selector.select(short_tokens, "", pieces, sopts, nullptr, &selection));
    CHECK(selection.dense);

    // Legal indices (ascending subset of token_indices containing the last
    // token) that dense execution must ignore.
    std::vector<int32_t> indices;
    for (uint32_t i = 0; i < short_tokens.size(); i += 8) {
        indices.push_back(static_cast<int32_t>(i));
    }
    if (indices.back() != static_cast<int32_t>(short_tokens.size() - 1)) {
        indices.push_back(static_cast<int32_t>(short_tokens.size() - 1));
    }
    selection.attention_indices = indices;
    selection.attention_top_k = static_cast<uint32_t>(indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    PrefillStats stats;
    CHECK_OK(prefiller.prefill(short_tokens, selection, popts, &stats));
    CHECK_EQ(stats.attn_top_k, 0);
    CHECK_EQ(stats.attn_indices, 0u);

    LlamaBackend::shutdown();
}

TEST_CASE(benchmark_repeats_median_and_audit) {
    const std::string model = test_model_path();
    if (model.empty()) {
        std::cout << "SKIP benchmark_repeats_median_and_audit (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "edge_prefill_t004_benchmark";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    CHECK(fs::is_directory(dir));

    std::string context_text = "The magic word is banana. ";
    for (int i = 0; i < 50; ++i) {
        context_text +=
            "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    {
        std::ofstream ctx_file((dir / "ctx.txt").string(), std::ios::binary);
        ctx_file << context_text;
        std::ofstream index_file((dir / "index.tsv").string(), std::ios::binary);
        index_file << "id\ttask\tcontext_path\tquestion\tanswer\n";
        index_file << "qa1\tqa\tctx.txt\tWhat is the magic word?\tbanana\n";
    }
    const std::string question = "What is the magic word?";

    BenchmarkConfig bcfg;
    bcfg.model = model;
    bcfg.data_dir = dir.string();
    bcfg.modes = {Mode::Latency};
    bcfg.budgets = {0.5};
    bcfg.lengths = {0};
    bcfg.repeats = 3;
    bcfg.warmup = 1;
    bcfg.max_new_tokens = 8;
    bcfg.attn_top_k = 4;
    bcfg.attn_top_k_layers = "";
    bcfg.out_dir = dir.string();

    std::vector<Sample> samples;
    std::vector<std::string> warnings;
    CHECK_OK(BenchmarkRunner().run(bcfg, &samples, &warnings));
    CHECK_EQ(samples.size(), 1u);
    if (samples.size() != 1u) return;
    const Sample& sample = samples[0];
    CHECK_EQ(sample.repeats, 3u);
    CHECK(sample.ttft_ms > 0.0);
    CHECK(std::isfinite(sample.ttft_ms));
    CHECK(!sample.dense);

    // Recompute the expected gathered-row count over the same tokens and
    // selection the runner used: identical prompt construction, planner
    // budget, default selector options, and AttentionSparsityController
    // with the plan's top-k.
    LlamaBackend::init();
    {
        LlamaBackend backend;
        CHECK_OK(backend.load_model(model));
        std::vector<llama_token> ctx_tokens;
        CHECK_OK(backend.tokenize(context_text, &ctx_tokens, false, false));
        std::vector<std::string> ctx_pieces(ctx_tokens.size());
        for (size_t i = 0; i < ctx_tokens.size(); ++i) {
            CHECK_OK(backend.token_to_piece(ctx_tokens[i], &ctx_pieces[i]));
        }
        // lengths={0} keeps the whole context; the runner rebuilds it by
        // joining token pieces.
        std::string joined;
        for (const auto& piece : ctx_pieces) joined += piece;
        const std::string prompt = build_prompt(backend.model(), joined, question);
        std::vector<llama_token> tokens;
        CHECK_OK(backend.tokenize(prompt, &tokens, true, true));
        std::vector<std::string> pieces(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            CHECK_OK(backend.token_to_piece(tokens[i], &pieces[i]));
        }

        Config run_cfg;
        run_cfg.mode = Mode::Latency;
        run_cfg.budget = 0.5;
        run_cfg.attn_top_k = bcfg.attn_top_k;
        PlannerInput pin;
        pin.n_tokens = static_cast<uint32_t>(tokens.size());
        pin.mode = Mode::Latency;
        pin.arch_supports_ffn_skip = true;
        pin.n_layer = backend.n_layer();
        pin.n_ubatch = 512;
        Plan plan;
        CHECK_OK(Planner::make_plan(pin, run_cfg, &plan));

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
        CHECK_OK(selector.select(tokens, question, pieces, sopts, nullptr, &selection));

        uint32_t expected_attn_top_k = 0;
        if (plan.attention_top_k > 0 && !selection.dense) {
            AttentionSparsityOptions aopts;
            aopts.block_size = sopts.block_size;
            aopts.top_k_blocks = plan.attention_top_k;
            std::vector<int32_t> indices;
            std::string reason;
            CHECK_OK(AttentionSparsityController::build(selection, aopts, &indices, &reason));
            expected_attn_top_k =
                indices.empty() ? 0 : static_cast<uint32_t>(indices.size());
        }
        CHECK_EQ(sample.attn_top_k, expected_attn_top_k);
    }
    LlamaBackend::shutdown();

    Reporter reporter;
    for (const auto& s : samples) {
        CHECK_OK(reporter.write_sample(s));
    }
    const fs::path csv_path = dir / "raw.csv";
    CHECK_OK(reporter.write_raw_csv(csv_path.string()));
    std::ifstream csv_in(csv_path.string());
    CHECK(csv_in.good());
    std::string header;
    std::getline(csv_in, header);
    CHECK(header.find("attn_top_k") != std::string::npos);
    CHECK(header.find("repeats") != std::string::npos);
}
