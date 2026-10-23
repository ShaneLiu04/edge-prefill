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

bool load(Loaded* out, uint32_t n_ubatch);
bool load(Loaded* out) { return load(out, 128); }

bool load(Loaded* out, uint32_t n_ubatch) {
    const std::string path = test_model_path();
    if (path.empty()) return false;
    if (!LlamaBackend::init().ok()) return false;
    if (!out->backend.load_model(path).ok()) return false;
    ContextOptions opts;
    opts.n_ctx = 2048;
    opts.n_batch = 512;
    opts.n_ubatch = n_ubatch;
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

// ---------------------------------------------------------------------------
// AR004 T001: large-chunk scalar gather regression (UT-C15).
//
// The scalar gather indices are scoped to the outer chunk, so the backend
// must decode each outer chunk in a single llama_decode call. The pre-fix
// binary passed chunk_size=0 to the backend, whose 512 fallback re-split a
// 1024-token outer chunk into two sub-batches; the second sub-batch's K
// view was narrower than the chunk's indices and the gather read unwritten
// cells (out-of-bounds assert in get_rows). This regression reproduces the
// geometry (auto chunk aligned to a 1024 n_ubatch, indices spanning the
// whole active span) and asserts one internal decode record per outer
// chunk, i.e. no re-chunking.
// ---------------------------------------------------------------------------

TEST_CASE(scalar_large_chunk_single_decode_regression) {
    const std::string path = test_model_path();
    if (path.empty()) {
        std::cout << "SKIP scalar_large_chunk_single_decode_regression (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    if (!LlamaBackend::init().ok()) return;
    LlamaBackend backend;
    if (!backend.load_model(path).ok()) return;
    ContextOptions copts;
    copts.n_ctx = 2048;
    copts.n_batch = 2048;
    copts.n_ubatch = 1024;
    copts.n_threads = 4;
    copts.n_threads_batch = 4;
    copts.flash_attn = false;
    CHECK_OK(backend.create_context(copts));

    // ~1200 tokens: with n_ubatch 1024 the outer plan yields two chunks
    // (1024 + remainder), each of which must be a single llama_decode.
    std::string text;
    for (int i = 0; i < 300; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    std::vector<llama_token> tokens;
    CHECK_OK(backend.tokenize(text, &tokens));
    if (tokens.size() > 1500) tokens.resize(1500);
    CHECK(tokens.size() > 1024);

    std::vector<std::string> pieces(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        CHECK_OK(backend.token_to_piece(tokens[i], &pieces[i]));
    }
    SelectionOptions sopts;
    sopts.block_size = 32;
    sopts.budget = 0.5;
    sopts.min_dense_tokens = 256;
    Selection selection;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "fox rain roof", pieces, sopts, nullptr,
                             &selection));
    CHECK(!selection.dense);
    // The active span must exceed the old 512 fallback so a re-chunking
    // backend would inevitably split the first outer chunk.
    CHECK(selection.token_indices.size() > 512);

    AttentionSparsityOptions aopts;
    aopts.block_size = 32;
    aopts.top_k_blocks = 8;
    std::string reason;
    std::vector<int32_t> indices;
    CHECK_OK(AttentionSparsityController::build(selection, aopts, &indices,
                                                &reason));
    CHECK(!indices.empty());
    selection.attention_indices = indices;
    selection.attention_top_k = static_cast<uint32_t>(indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&backend, cache.get());
    SparsePrefiller::Options popts;
    popts.chunk_size = 0;  // AR004 FR-1: auto = align to n_ubatch (1024)
    popts.enable_attn_sparse = true;
    popts.enable_prefix_cache = false;
    popts.enable_ffn_skip = false;

    PrefillStats stats;
    CHECK_OK(prefiller.prefill(tokens, selection, popts, &stats));
    // The active span fits one auto chunk (n_ubatch 1024): exactly one
    // outer chunk, decoded exactly once (no backend re-chunking at the old
    // 512 fallback — the pre-fix binary split it into 512+313 and the
    // second sub-batch gathered rows beyond its K view).
    const uint32_t active = static_cast<uint32_t>(selection.token_indices.size());
    CHECK(active > 512);
    CHECK(active <= 1024);
    CHECK_EQ(stats.chunks, 1u);
    CHECK_EQ(stats.chunk_stats.size(), stats.chunks);
    CHECK_EQ(stats.chunk_stats.back().attn_rows,
             static_cast<uint32_t>(indices.size()));

    const std::vector<float> logits = logits_of(backend);
    CHECK(!logits.empty());
    CHECK(all_finite(logits));

    LlamaBackend::shutdown();
}

// ---------------------------------------------------------------------------
// AR004 T003: ppl quality is a retention ratio against a dense reference,
// not the raw geometric-mean answer probability (which lands near 0 and
// reads as "broken metric"). Also covers EX-8: an unusable ppl answer must
// be reported as a warning and skipped, never silently scored as 0.
// ---------------------------------------------------------------------------

TEST_CASE(benchmark_ppl_quality_retention) {
    const std::string model = test_model_path();
    if (model.empty()) {
        std::cout << "SKIP benchmark_ppl_quality_retention (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "edge_prefill_t003_ppl";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    std::string context_text = "The magic word is banana. ";
    for (int i = 0; i < 50; ++i) {
        context_text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    {
        std::ofstream ctx_file((dir / "ctx.txt").string(), std::ios::binary);
        ctx_file << context_text;
        std::ofstream index_file((dir / "index.tsv").string(), std::ios::binary);
        index_file << "id\ttask\tcontext_path\tquestion\tanswer\n";
        index_file << "ppl1\tppl\tctx.txt\tContinue the passage.\t"
                      "The quick brown fox jumps over the lazy dog.\n";
        index_file << "pplbad\tppl\tctx.txt\tContinue the passage.\t\n";
        index_file << "qa1\tqa\tctx.txt\tWhat is the magic word?\tbanana\n";
    }

    // Dense reference run: the sparse config IS dense, so retention against
    // the dense reference must be exactly 1.
    BenchmarkConfig bcfg;
    bcfg.model = model;
    bcfg.data_dir = dir.string();
    bcfg.modes = {Mode::Dense};
    bcfg.budgets = {1.0};
    bcfg.lengths = {0};
    bcfg.repeats = 1;
    bcfg.warmup = 0;
    bcfg.max_new_tokens = 4;
    bcfg.out_dir = dir.string();
    std::vector<Sample> dense_samples;
    std::vector<std::string> warnings;
    CHECK_OK(BenchmarkRunner().run(bcfg, &dense_samples, &warnings));
    // qa1 + ppl1 run; pplbad must be skipped with an explicit warning.
    CHECK_EQ(dense_samples.size(), 2u);
    bool warned = false;
    for (const auto& w : warnings) warned |= w.find("ppl-answer-invalid") != std::string::npos;
    CHECK(warned);

    const Sample* dense_ppl = nullptr;
    const Sample* qa = nullptr;
    for (const auto& s : dense_samples) {
        if (s.task == "ppl") dense_ppl = &s;
        if (s.task == "qa") qa = &s;
    }
    CHECK(dense_ppl != nullptr);
    if (dense_ppl) {
        // case_id is "<id>@<effective length>" where the effective length
        // is the real context token count (lengths={0} keeps everything).
        CHECK(dense_ppl->case_id.rfind("ppl1@", 0) == 0);
        CHECK_GT(dense_ppl->quality, 0.0);
        CHECK_NEAR(dense_ppl->quality, 1.0, 1e-6);
        CHECK_GT(dense_ppl->ppl_dense, 0.0);
        CHECK_NEAR(dense_ppl->ppl_sparse, dense_ppl->ppl_dense, 1e-6);
    }
    // The qa task must be unaffected by the ppl change: token-overlap F1
    // against the gold answer, finite and positive when the needle is hit.
    CHECK(qa != nullptr);
    if (qa) CHECK(qa->quality >= 0.0 && qa->quality <= 1.0);

    // Sparse run: retention may drop below 1 but must stay in (0, 1] and
    // never read as the old broken ~0 quality.
    bcfg.modes = {Mode::Latency};
    bcfg.budgets = {0.5};
    bcfg.attn_top_k = 4;
    std::vector<Sample> sparse_samples;
    CHECK_OK(BenchmarkRunner().run(bcfg, &sparse_samples, &warnings));
    const Sample* sparse_ppl = nullptr;
    for (const auto& s : sparse_samples) {
        if (s.task == "ppl") sparse_ppl = &s;
    }
    CHECK(sparse_ppl != nullptr);
    if (sparse_ppl) {
        CHECK_GT(sparse_ppl->quality, 0.0);
        CHECK(sparse_ppl->quality <= 1.0 + 1e-9);
        CHECK_GT(sparse_ppl->ppl_sparse, 0.0);
        CHECK_GT(sparse_ppl->ppl_dense, 0.0);
    }

    // CSV audit columns for traceability (FR-5: raw values land in the CSV).
    Reporter reporter;
    for (const auto& s : sparse_samples) CHECK_OK(reporter.write_sample(s));
    const fs::path csv_path = dir / "raw.csv";
    CHECK_OK(reporter.write_raw_csv(csv_path.string()));
    std::ifstream csv_in(csv_path.string());
    std::string header;
    std::getline(csv_in, header);
    CHECK(header.find("ppl_dense") != std::string::npos);
    CHECK(header.find("ppl_sparse") != std::string::npos);
}

// ---------------------------------------------------------------------------
// AR003 T003: per-head gather patch C API + backend wrapper.
//
// The symbols llama_set_prefill_attn_ranked_rows /
// llama_set_prefill_attn_head_top_k / llama_get_prefill_attn_ranked_rows_size
// and the LlamaBackend wrapper methods (set_attn_ranked_rows,
// set_attn_head_top_k, attn_head_top_k_size, attn_per_head_supported) do not
// exist yet, so this translation unit fails to compile until T003 lands
// (Red). Every case below drives the real model (explicit SKIP when
// EDGE_PREFILL_TEST_MODEL is unset) and reuses the AR002 fixture helpers.
// ---------------------------------------------------------------------------

namespace {

int32_t model_n_head(const LlamaBackend& backend) {
    return llama_model_n_head(backend.model());
}

std::vector<int32_t> ascending_rows(int32_t begin, int32_t end) {
    std::vector<int32_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return rows;
}

std::vector<llama_pos> absolute_positions(size_t begin, size_t end) {
    std::vector<llama_pos> pos(end - begin);
    std::iota(pos.begin(), pos.end(), static_cast<llama_pos>(begin));
    return pos;
}

// n == 0 clears both settings; a NULL pointer with n == 0 is the documented
// clear form (same contract as the AR002 indices API).
void clear_per_head_attn(llama_context* ctx) {
    llama_set_prefill_attn_head_top_k(ctx, nullptr, 0);
    llama_set_prefill_attn_ranked_rows(ctx, nullptr, 0);
}

std::vector<int32_t> uniform_head_matrix(int32_t n_layer, int32_t n_head,
                                         int32_t value) {
    return std::vector<int32_t>(static_cast<size_t>(n_layer) *
                                    static_cast<size_t>(n_head),
                                value);
}

std::vector<float> dense_prefill_logits(LlamaBackend& backend,
                                        const std::vector<llama_token>& tokens,
                                        uint32_t chunk) {
    clear_per_head_attn(backend.context());
    CHECK_OK(backend.reset_sequence());
    PrefillStats stats;
    CHECK_OK(backend.prefill(tokens, absolute_positions(0, tokens.size()),
                             chunk, &stats));
    return logits_of(backend);
}

std::vector<float> matrix_prefill_logits(
        LlamaBackend& backend, const std::vector<llama_token>& tokens,
        const std::vector<int32_t>& pool, const std::vector<int32_t>& matrix,
        uint32_t chunk) {
    clear_per_head_attn(backend.context());
    CHECK_OK(backend.reset_sequence());
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(backend.context(), pool.data(),
                                                pool.size()),
             0);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(backend.context(),
                                               matrix.data(), matrix.size()),
             0);
    PrefillStats stats;
    CHECK_OK(backend.prefill(tokens, absolute_positions(0, tokens.size()),
                             chunk, &stats));
    return logits_of(backend);
}

// AR002 scalar path driven exactly like SparsePrefiller::attn_prefill: per
// chunk, gather the ascending prefix of rows already written. Only the
// matrix is cleared up front (the ranked pool must not affect the scalar
// path); the pool is left untouched on purpose.
std::vector<float> scalar_prefill_logits(
        LlamaBackend& backend, const std::vector<llama_token>& tokens,
        const std::vector<int32_t>& rows, uint32_t chunk) {
    llama_set_prefill_attn_head_top_k(backend.context(), nullptr, 0);
    CHECK_OK(backend.reset_sequence());
    for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
        const size_t end =
                std::min(begin + static_cast<size_t>(chunk), tokens.size());
        std::vector<int32_t> allowed;
        for (int32_t row : rows) {
            if (row >= static_cast<int32_t>(end)) break;
            allowed.push_back(row);
        }
        CHECK_OK(backend.set_attn_indices(allowed));
        CHECK_OK(backend.set_attn_top_k(static_cast<int32_t>(allowed.size())));
        PrefillStats stats;
        CHECK_OK(backend.prefill(
                std::vector<llama_token>(tokens.begin() + begin,
                                         tokens.begin() + end),
                absolute_positions(begin, end), 0, &stats));
    }
    backend.set_attn_top_k(0);
    backend.set_attn_indices({});
    return logits_of(backend);
}

} // namespace

TEST_CASE(per_head_ranked_rows_c_api_validation) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_ranked_rows_c_api_validation (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    llama_context* ctx = loaded.backend.context();
    const int32_t n_ctx = static_cast<int32_t>(loaded.backend.n_ctx());

    // NULL context is rejected outright.
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(nullptr, nullptr, 0), 2);

    // Fresh context: empty pool, and n == 0 with NULL rows clears legally.
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), 0);
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, nullptr, 0), 0);

    // NULL rows with n > 0 is invalid.
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, nullptr, 3), 2);

    // Out-of-range rows (>= n_ctx, negative) are rejected and leave the
    // pool unchanged (no half-written state).
    const std::vector<int32_t> too_large = {0, n_ctx};
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, too_large.data(),
                                                too_large.size()),
             2);
    const std::vector<int32_t> negative = {-1};
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, negative.data(),
                                                negative.size()),
             2);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), 0);

    // Duplicate rows are legal: the pool is a ranked order, not a set.
    const std::vector<int32_t> ranked = {7, 3, 7, 1};
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, ranked.data(),
                                                ranked.size()),
             0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), 4);

    // Max boundary: n == n_ctx is a legal full pool.
    const std::vector<int32_t> full_pool = ascending_rows(0, n_ctx);
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, full_pool.data(),
                                                full_pool.size()),
             0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), n_ctx);

    // n == 0 clears and restores the original path.
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, nullptr, 0), 0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), 0);

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_ranked_rows_full_pool_dense_equivalence) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_ranked_rows_full_pool_dense_equivalence (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    llama_context* ctx = backend.context();
    const int32_t n_ctx = static_cast<int32_t>(backend.n_ctx());
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);

    // The max boundary n == n_ctx must be accepted (checked in the API
    // validation case too; re-checked here to fail close to the execution).
    const std::vector<int32_t> full_pool = ascending_rows(0, n_ctx);
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, full_pool.data(),
                                                full_pool.size()),
             0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), n_ctx);

    // k == pool length: every group gathers the whole (live-filtered) pool,
    // i.e. all written rows in ascending order, which is exactly dense.
    const std::vector<int32_t> matrix =
            uniform_head_matrix(n_layer, n_head, n_ctx);
    const std::vector<float> dense =
            dense_prefill_logits(backend, loaded.tokens, 128);
    const std::vector<float> gathered = matrix_prefill_logits(
            backend, loaded.tokens, full_pool, matrix, 128);
    CHECK(!dense.empty());
    CHECK(!gathered.empty());
    CHECK(max_abs_diff(dense, gathered) < 1e-5);
    CHECK_EQ(argmax(dense), argmax(gathered));

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_head_top_k_c_api_validation) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_head_top_k_c_api_validation (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    llama_context* ctx = backend.context();
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);
    const size_t matrix_len =
            static_cast<size_t>(n_layer) * static_cast<size_t>(n_head);
    CHECK(matrix_len > 0);

    // Half-write guard: a non-empty matrix with an empty pool is rejected.
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, nullptr, 0), 0);
    const std::vector<int32_t> dense_matrix(matrix_len, -1);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, dense_matrix.data(),
                                               matrix_len),
             2);

    // With a pool set, length must be exactly n_layer * n_head.
    const std::vector<int32_t> pool = {0, 1, 2};
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, pool.data(), pool.size()),
             0);
    const std::vector<int32_t> short_matrix(matrix_len - 1, -1);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, short_matrix.data(),
                                               matrix_len - 1),
             2);
    const std::vector<int32_t> long_matrix(matrix_len + 1, -1);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, long_matrix.data(),
                                               matrix_len + 1),
             2);

    // Element rules: 0 and < -1 are invalid; a failed set leaves the matrix
    // unset (state unchanged).
    std::vector<int32_t> zero_matrix(matrix_len, -1);
    zero_matrix[0] = 0;
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, zero_matrix.data(),
                                               matrix_len),
             2);
    std::vector<int32_t> neg_matrix(matrix_len, -1);
    neg_matrix[0] = -2;
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, neg_matrix.data(),
                                               matrix_len),
             2);
    CHECK_EQ(backend.attn_head_top_k_size(), 0);

    // A legal all-dense matrix sets and the size query is consistent.
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, dense_matrix.data(),
                                               matrix_len),
             0);
    CHECK_EQ(backend.attn_head_top_k_size(),
             static_cast<int32_t>(matrix_len));

    // Minimal legal value: k == 1 per head.
    const std::vector<int32_t> one_matrix(matrix_len, 1);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, one_matrix.data(),
                                               matrix_len),
             0);
    CHECK_EQ(backend.attn_head_top_k_size(),
             static_cast<int32_t>(matrix_len));

    // n == 0 clears the matrix and restores the AR002 scalar path; the
    // ranked pool survives the matrix clear.
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx),
             static_cast<int32_t>(pool.size()));
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, nullptr, 0), 0);
    CHECK_EQ(backend.attn_head_top_k_size(), 0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx),
             static_cast<int32_t>(pool.size()));

    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, nullptr, 0), 0);
    LlamaBackend::shutdown();
}

TEST_CASE(per_head_backend_wrapper_status_mapping) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_backend_wrapper_status_mapping (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    // The patch is applied in this test environment, so symbol probing
    // must report the per-head path as supported.
    CHECK(backend.attn_per_head_supported());

    // C API return 2 maps to Status::InvalidArgument.
    const std::vector<int32_t> negative_row = {-1};
    CHECK_ERR(backend.set_attn_ranked_rows(negative_row),
              ErrorCode::InvalidArgument);
    const std::vector<int32_t> out_of_range_row = {
            static_cast<int32_t>(backend.n_ctx())};
    CHECK_ERR(backend.set_attn_ranked_rows(out_of_range_row),
              ErrorCode::InvalidArgument);

    // Legal ranked pool (duplicates allowed, ranked order) sets fine.
    const std::vector<int32_t> pool = {5, 3, 5, 0};
    CHECK_OK(backend.set_attn_ranked_rows(pool));
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(backend.context()),
             static_cast<int32_t>(pool.size()));

    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    const size_t matrix_len =
            static_cast<size_t>(n_layer) * static_cast<size_t>(n_head);

    CHECK_EQ(backend.attn_head_top_k_size(), 0);
    const std::vector<int32_t> wrong_len(matrix_len - 1, -1);
    CHECK_ERR(backend.set_attn_head_top_k(wrong_len),
              ErrorCode::InvalidArgument);
    std::vector<int32_t> zero_elem(matrix_len, 1);
    zero_elem[0] = 0;
    CHECK_ERR(backend.set_attn_head_top_k(zero_elem),
              ErrorCode::InvalidArgument);
    std::vector<int32_t> neg_elem(matrix_len, 1);
    neg_elem[0] = -2;
    CHECK_ERR(backend.set_attn_head_top_k(neg_elem),
              ErrorCode::InvalidArgument);

    // Legal matrix sets fine and the size query matches.
    const std::vector<int32_t> matrix(matrix_len, -1);
    CHECK_OK(backend.set_attn_head_top_k(matrix));
    CHECK_EQ(backend.attn_head_top_k_size(),
             static_cast<int32_t>(matrix_len));

    // Clearing through the wrapper keeps Status::Ok.
    CHECK_OK(backend.set_attn_head_top_k({}));
    CHECK_EQ(backend.attn_head_top_k_size(), 0);
    CHECK_OK(backend.set_attn_ranked_rows({}));
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(backend.context()), 0);

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_matrix_clear_restores_scalar_path) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_matrix_clear_restores_scalar_path (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    llama_context* ctx = backend.context();
    const int32_t n = static_cast<int32_t>(loaded.tokens.size());
    const int32_t k = 64;
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);

    // The same gather rows for both paths: the matrix pool is ascending, so
    // the per-group "pool prefix k, live-filtered, ascending" set equals the
    // scalar "first k ascending rows" set on every chunk.
    const std::vector<int32_t> pool = ascending_rows(0, n);
    const std::vector<int32_t> scalar_rows = ascending_rows(0, k);
    const std::vector<int32_t> matrix =
            uniform_head_matrix(n_layer, n_head, k);

    // Scalar baseline before the matrix ever engages.
    clear_per_head_attn(ctx);
    const std::vector<float> scalar_before =
            scalar_prefill_logits(backend, loaded.tokens, scalar_rows, 128);
    CHECK(!scalar_before.empty());

    // Engage the matrix path end to end.
    clear_per_head_attn(ctx);
    CHECK_OK(backend.reset_sequence());
    CHECK_EQ(llama_set_prefill_attn_ranked_rows(ctx, pool.data(), pool.size()),
             0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), n);
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, matrix.data(),
                                               matrix.size()),
             0);
    CHECK_EQ(backend.attn_head_top_k_size(),
             static_cast<int32_t>(matrix.size()));
    PrefillStats stats;
    CHECK_OK(backend.prefill(loaded.tokens, absolute_positions(0, loaded.tokens.size()), 128, &stats));

    // n == 0 on the matrix restores the AR002 scalar path; the pool itself
    // must survive the clear untouched.
    CHECK_EQ(llama_set_prefill_attn_head_top_k(ctx, nullptr, 0), 0);
    CHECK_EQ(backend.attn_head_top_k_size(), 0);
    CHECK_EQ(llama_get_prefill_attn_ranked_rows_size(ctx), n);

    // The scalar path must ignore the still-set pool and reproduce the
    // baseline logits (FR-1 AC4: cleared matrix == scalar semantics).
    const std::vector<float> scalar_after =
            scalar_prefill_logits(backend, loaded.tokens, scalar_rows, 128);
    CHECK(!scalar_after.empty());
    CHECK(max_abs_diff(scalar_before, scalar_after) < 1e-5);
    CHECK_EQ(argmax(scalar_before), argmax(scalar_after));

    clear_per_head_attn(ctx);
    LlamaBackend::shutdown();
}

TEST_CASE(per_head_all_dense_matrix_equals_dense) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_all_dense_matrix_equals_dense (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);
    const int32_t n = static_cast<int32_t>(loaded.tokens.size());
    const std::vector<int32_t> pool = ascending_rows(0, n);

    const std::vector<float> dense =
            dense_prefill_logits(backend, loaded.tokens, 128);
    CHECK(!dense.empty());

    // All -1: every head is a retrieval head, direct pass-through with zero
    // gather overhead, so the run is dense-equivalent.
    const std::vector<int32_t> all_dense =
            uniform_head_matrix(n_layer, n_head, -1);
    const std::vector<float> run_all_dense = matrix_prefill_logits(
            backend, loaded.tokens, pool, all_dense, 128);
    CHECK(!run_all_dense.empty());
    CHECK(max_abs_diff(dense, run_all_dense) < 1e-5);

    // Single all-dense layer, remaining layers sparse with k == pool length:
    // the sparse groups gather the whole live-filtered pool, so the overall
    // run stays dense-equivalent while splitting dense/sparse layers.
    std::vector<int32_t> one_layer_dense =
            uniform_head_matrix(n_layer, n_head, n);
    for (int32_t h = 0; h < n_head; ++h) one_layer_dense[h] = -1;
    const std::vector<float> run_one_layer = matrix_prefill_logits(
            backend, loaded.tokens, pool, one_layer_dense, 128);
    CHECK(!run_one_layer.empty());
    CHECK(max_abs_diff(dense, run_one_layer) < 1e-5);

    // Dense/sparse mixed inside one layer: dense heads pass through while
    // same-k sparse heads gather the full pool; the head-dim concat of the
    // group outputs must remain dense-equivalent.
    std::vector<int32_t> mixed = uniform_head_matrix(n_layer, n_head, -1);
    for (int32_t h = 1; h < n_head; h += 2) mixed[h] = n;
    const std::vector<float> run_mixed = matrix_prefill_logits(
            backend, loaded.tokens, pool, mixed, 128);
    CHECK(!run_mixed.empty());
    CHECK(max_abs_diff(dense, run_mixed) < 1e-5);

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_single_group_matches_scalar) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_single_group_matches_scalar (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);
    const int32_t n = static_cast<int32_t>(loaded.tokens.size());
    const int32_t k = 64;

    // Degenerate single-group case: every head sparse with the same k. With
    // an ascending pool, "pool prefix k, live-filtered, ascending" equals
    // the AR002 scalar "first k ascending rows" gather on every chunk, so
    // both paths must agree (design 4.5: all-sparse same-value == scalar).
    const std::vector<int32_t> pool = ascending_rows(0, n);
    const std::vector<int32_t> scalar_rows = ascending_rows(0, k);
    const std::vector<int32_t> matrix = uniform_head_matrix(n_layer, n_head, k);

    const std::vector<float> scalar =
            scalar_prefill_logits(backend, loaded.tokens, scalar_rows, 128);
    const std::vector<float> grouped = matrix_prefill_logits(
            backend, loaded.tokens, pool, matrix, 128);
    CHECK(!scalar.empty());
    CHECK(!grouped.empty());
    CHECK(max_abs_diff(scalar, grouped) < 1e-5);
    CHECK_EQ(argmax(scalar), argmax(grouped));

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_first_chunk_empty_prefix) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_first_chunk_empty_prefix (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    CHECK(n_head > 0);

    // A short prompt keeps live n_kv tiny; the only pool row sits at the far
    // end of the context, so every group filters to an empty prefix on every
    // chunk and must fall back to dense attention over the written rows.
    std::vector<llama_token> short_tokens(loaded.tokens.begin(),
                                          loaded.tokens.begin() + 16);
    const std::vector<int32_t> pool = {
            static_cast<int32_t>(backend.n_ctx()) - 1};
    const std::vector<int32_t> matrix = uniform_head_matrix(n_layer, n_head, 1);

    const std::vector<float> first = matrix_prefill_logits(
            backend, short_tokens, pool, matrix, 8);
    CHECK(!first.empty());
    CHECK(all_finite(first));

    // Greedy prefill is deterministic: an identical rerun picks the same
    // first token even though every group ran the empty-prefix fallback.
    const std::vector<float> second = matrix_prefill_logits(
            backend, short_tokens, pool, matrix, 8);
    CHECK(!second.empty());
    CHECK(all_finite(second));
    CHECK_EQ(argmax(first), argmax(second));

    LlamaBackend::shutdown();
}

TEST_CASE(per_head_group_limit_degrades_layer_to_dense) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP per_head_group_limit_degrades_layer_to_dense (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend& backend = loaded.backend;
    const int32_t n_layer = backend.n_layer();
    const int32_t n_head = model_n_head(backend);
    if (n_head < 5) {
        // 5 distinct top-k values within one layer are impossible with
        // fewer than 5 heads; the boundary is untestable on this model.
        std::cout << "SKIP per_head_group_limit_degrades_layer_to_dense (n_head < 5)\n";
        return;
    }
    const int32_t n = static_cast<int32_t>(loaded.tokens.size());
    const std::vector<int32_t> pool = ascending_rows(0, n);

    // Layer 0 carries 5 distinct values -> the layer safely degrades to
    // dense; every other layer is all-dense (-1), so the whole run must be
    // dense-equivalent (a wrong degradation cannot hide behind sparse
    // layers here).
    std::vector<int32_t> five_distinct =
            uniform_head_matrix(n_layer, n_head, -1);
    for (int32_t h = 0; h < n_head; ++h) five_distinct[h] = 1 + h % 5;
    const std::vector<float> dense =
            dense_prefill_logits(backend, loaded.tokens, 128);
    const std::vector<float> degraded = matrix_prefill_logits(
            backend, loaded.tokens, pool, five_distinct, 128);
    CHECK(!degraded.empty());
    CHECK(all_finite(degraded));
    CHECK(max_abs_diff(dense, degraded) < 1e-5);

    // Contrast: exactly 4 distinct values stays within the group limit and
    // executes the normal grouped gather path.
    std::vector<int32_t> four_distinct =
            uniform_head_matrix(n_layer, n_head, -1);
    for (int32_t h = 0; h < n_head; ++h) four_distinct[h] = 1 + h % 4;
    const std::vector<float> grouped = matrix_prefill_logits(
            backend, loaded.tokens, pool, four_distinct, 128);
    const std::vector<float> grouped_again = matrix_prefill_logits(
            backend, loaded.tokens, pool, four_distinct, 128);
    CHECK(!grouped.empty());
    CHECK(all_finite(grouped));
    CHECK(all_finite(grouped_again));
    CHECK_EQ(argmax(grouped), argmax(grouped_again));

    LlamaBackend::shutdown();
}

// AR004 FR-2: pooled compressed branches execute on the real model. The
// pool branch must not disturb prefill correctness: logits stay finite and
// greedy prefill stays deterministic.
TEST_CASE(pool_branches_execution_engages) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP pool_branches_execution_engages (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    // The pool set must be non-empty for this scenario to exercise the
    // branch; verify with the planner before running.
    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string pool_reason;
    CHECK_OK(AttentionSparsityController::build_pool_blocks(
            scenario.selection, scenario.indices, 32, &starts, &sizes,
            &max_pos, &pool_reason));
    CHECK(!starts.empty());

    Selection selection = scenario.selection;
    selection.attention_indices = scenario.indices;
    selection.attention_top_k = static_cast<uint32_t>(scenario.indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    popts.enable_pool_blocks = true;

    PrefillStats stats;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats));
    CHECK(!stats.attn_fallback);
    const std::vector<float> first_logits = logits_of(loaded.backend);
    CHECK(!first_logits.empty());
    CHECK(all_finite(first_logits));

    // Deterministic rerun with the pooled branch must reproduce the same
    // greedy first token.
    PrefillStats stats2;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats2));
    const std::vector<float> second_logits = logits_of(loaded.backend);
    CHECK_EQ(argmax(first_logits), argmax(second_logits));

    LlamaBackend::shutdown();
}

// AR004 FR-2 zero-regression: an empty pool set (everything gathered)
// leaves the graph untouched — the flag is inert when the planner pools
// nothing, so logits are bit-identical to the flag-off run.
TEST_CASE(pool_branches_empty_set_equivalence) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP pool_branches_empty_set_equivalence (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    // Gather EVERY selected token: the planner pools nothing.
    Selection selection = scenario.selection;
    selection.attention_indices.clear();
    selection.attention_indices.reserve(scenario.selection.token_indices.size());
    for (uint32_t t : scenario.selection.token_indices) {
        selection.attention_indices.push_back(static_cast<int32_t>(t));
    }
    selection.attention_top_k =
            static_cast<uint32_t>(scenario.selection.token_indices.size());

    std::vector<int32_t> starts;
    std::vector<int32_t> sizes;
    std::vector<int32_t> max_pos;
    std::string pool_reason;
    CHECK_OK(AttentionSparsityController::build_pool_blocks(
            selection, selection.attention_indices, 32, &starts, &sizes,
            &max_pos, &pool_reason));
    CHECK(starts.empty());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    PrefillStats stats_off;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats_off));
    const std::vector<float> logits_off = logits_of(loaded.backend);

    popts.enable_pool_blocks = true;
    PrefillStats stats_on;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats_on));
    const std::vector<float> logits_on = logits_of(loaded.backend);

    CHECK_EQ(logits_off.size(), logits_on.size());
    for (size_t i = 0; i < logits_off.size(); ++i) {
        CHECK_EQ(logits_off[i], logits_on[i]);
    }

    LlamaBackend::shutdown();
}

// AR004 FR-3: query-chunk gating executes on the real model. Every query
// group gathers its own rows; logits stay finite and greedy prefill stays
// deterministic.
TEST_CASE(qg_execution_engages) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP qg_execution_engages (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    // The lexical per-position scores retained by the selector drive the
    // gating plan.
    CHECK_EQ(scenario.selection.per_pos_scores.size(),
             scenario.selection.total_tokens);

    QueryGatingOptions qg_opts;
    qg_opts.query_chunk_size = 64;
    qg_opts.query_chunk_max = 16;
    qg_opts.top_k_per_group = 128;
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> bounds;
    std::string qg_reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            scenario.selection, scenario.selection.per_pos_scores, qg_opts,
            &groups, &bounds, &qg_reason));
    CHECK(groups.size() >= 2);

    Selection selection = scenario.selection;
    selection.attention_indices = scenario.indices;
    selection.attention_top_k = static_cast<uint32_t>(scenario.indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    popts.enable_query_gating = true;
    popts.qg_group_rows = groups;
    popts.qg_bounds = bounds;

    PrefillStats stats;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats));
    CHECK(stats.query_gating);
    CHECK_EQ(stats.qg_groups, static_cast<uint32_t>(groups.size()));
    CHECK(!stats.attn_fallback);
    const std::vector<float> first_logits = logits_of(loaded.backend);
    CHECK(!first_logits.empty());
    CHECK(all_finite(first_logits));

    PrefillStats stats2;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats2));
    const std::vector<float> second_logits = logits_of(loaded.backend);
    CHECK_EQ(argmax(first_logits), argmax(second_logits));

    LlamaBackend::shutdown();
}

// AR004 FR-3 zero-regression: an empty plan leaves the scalar path
// untouched — the flag is inert without group rows, so logits are
// bit-identical to the plain sparse run.
TEST_CASE(qg_empty_plan_equivalence) {
    Loaded loaded;
    SparseScenario scenario;
    if (!setup_sparse_scenario(&loaded, &scenario)) {
        std::cout << "SKIP qg_empty_plan_equivalence (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    Selection selection = scenario.selection;
    selection.attention_indices = scenario.indices;
    selection.attention_top_k = static_cast<uint32_t>(scenario.indices.size());

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts = exec_options();
    PrefillStats stats_off;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats_off));
    const std::vector<float> logits_off = logits_of(loaded.backend);

    popts.enable_query_gating = true;  // no group rows: inert
    PrefillStats stats_on;
    CHECK_OK(prefiller.prefill(loaded.tokens, selection, popts, &stats_on));
    const std::vector<float> logits_on = logits_of(loaded.backend);
    CHECK(!stats_on.query_gating);

    CHECK_EQ(logits_off.size(), logits_on.size());
    for (size_t i = 0; i < logits_off.size(); ++i) {
        CHECK_EQ(logits_off[i], logits_on[i]);
    }

    LlamaBackend::shutdown();
}

// AR004 FR-3: the patch C API validates its state machine — group budget,
// append-only group order, ascending rows, bounds length.
TEST_CASE(qg_capi_validation) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP qg_capi_validation (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    CHECK_ERR(loaded.backend.set_qg_count(17), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_count(-1), ErrorCode::InvalidArgument);
    CHECK_OK(loaded.backend.set_qg_count(2));
    // Groups must be configured in ascending order.
    CHECK_ERR(loaded.backend.set_qg_rows(1, {4, 8}), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_rows(0, {8, 4}), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_rows(0, {4, 4}), ErrorCode::InvalidArgument);
    CHECK_OK(loaded.backend.set_qg_rows(0, {4, 8}));
    CHECK_OK(loaded.backend.set_qg_rows(1, {}));
    // Bounds: wrong length, wrong start, non-ascending.
    CHECK_ERR(loaded.backend.set_qg_bounds({0, 4}), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_bounds({1, 4, 8}), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_bounds({0, 8, 4}), ErrorCode::InvalidArgument);
    CHECK_OK(loaded.backend.set_qg_bounds({0, 4, 8}));
    CHECK_EQ(loaded.backend.qg_count(), 2);
    // Reset restores a clean state.
    CHECK_OK(loaded.backend.set_qg_count(0));
    CHECK_EQ(loaded.backend.qg_count(), 0);
    CHECK_ERR(loaded.backend.set_qg_rows(0, {0}), ErrorCode::InvalidArgument);
    CHECK_ERR(loaded.backend.set_qg_bounds({0, 1}), ErrorCode::InvalidArgument);

    LlamaBackend::shutdown();
}
