#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/ffn_skip.h"
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

std::vector<float> dense_run(LlamaBackend* backend, const std::vector<llama_token>& tokens,
                             uint32_t chunk_size) {
    std::vector<llama_pos> positions(tokens.size());
    std::iota(positions.begin(), positions.end(), 0);
    if (!backend->reset_sequence().ok()) return {};
    PrefillStats stats;
    if (!backend->prefill(tokens, positions, chunk_size, &stats).ok()) return {};
    return logits_of(*backend);
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return worst;
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

} // namespace

namespace {

std::string generate(LlamaBackend* backend, SparsePrefiller* prefiller,
                     const std::vector<llama_token>& tokens, const Selection& selection,
                     uint32_t max_new_tokens) {
    SparsePrefiller::Options opts;
    opts.chunk_size = 512;
    opts.enable_prefix_cache = false;
    opts.enable_ffn_skip = false;
    uint64_t skip_mask = 0;
    FfnSkipController::auto_mask(backend->n_layer(), 0.25, &skip_mask);
    opts.ffn_skip_mask = skip_mask;
    PrefillStats stats;
    if (!prefiller->prefill(tokens, selection, opts, &stats).ok()) return "<prefill-failed>";
    const float* data = llama_get_logits_ith(backend->context(), -1);
    if (!data) return "<no-logits>";
    std::vector<float> logits(data, data + backend->n_vocab());
    const auto pick = [](const std::vector<float>& values) {
        return static_cast<llama_token>(
            std::max_element(values.begin(), values.end()) - values.begin());
    };
    llama_token next = pick(logits);
    std::string out;
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        std::string piece;
        backend->token_to_piece(next, &piece);
        out += piece;
        if (next == llama_vocab_eos(backend->vocab())) break;
        std::vector<float> step;
        if (!backend->decode(next, static_cast<llama_pos>(tokens.size() + i), &step).ok()) break;
        next = pick(step);
    }
    return out;
}

} // namespace

TEST_CASE(integration_sparse_needle) {
    const std::string path = test_model_path();
    if (path.empty()) {
        std::cout << "SKIP integration_sparse_needle (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend::init();
    LlamaBackend backend;
    CHECK_OK(backend.load_model(path));
    ContextOptions ctx;
    ctx.n_ctx = 10240;
    ctx.n_batch = 512;
    ctx.n_ubatch = 512;
    ctx.n_threads = 8;
    CHECK_OK(backend.create_context(ctx));

    std::string fact = "The secret passcode for vault 1 is MAGENTA-FALCON-7000. ";
    std::string body;
    for (int i = 0; i < 330; ++i) {
        if (i == 35) body += fact;
        body += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    const std::string text =
        "<|im_start|>system\nYou answer questions using only the provided context."
        "<|im_end|>\n<|im_start|>user\nContext:\n" + body +
        "\n\nQuestion: What is the secret passcode for vault 1?"
        "\nAnswer with the shortest exact span.<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> tokens;
    CHECK_OK(backend.tokenize(text, &tokens, true, true));

    std::vector<std::string> pieces(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        CHECK_OK(backend.token_to_piece(tokens[i], &pieces[i]));
    }

    SelectionOptions sopts;
    sopts.block_size = 32;
    sopts.budget = 0.5;
    sopts.min_dense_tokens = 128;
    Selection selection;
    BlockSelector selector;
    CHECK_OK(selector.select(tokens, "secret passcode vault 1", pieces, sopts, nullptr,
                             &selection));
    std::cout << "tokens=" << tokens.size() << " selected=" << selection.selected_tokens
              << " dense=" << selection.dense << "\n";

    std::unique_ptr<PrefixCache> cache;
    SparsePrefiller prefiller(&backend, cache.get());
    const std::string generated = generate(&backend, &prefiller, tokens, selection, 12);
    std::cout << "sparse generated=" << generated << "\n";
    CHECK(generated.find("MAGENTA") != std::string::npos);

    Selection dense_selection;
    auto dense_opts = sopts;
    dense_opts.budget = 1.0;
    CHECK_OK(selector.select(tokens, "secret passcode vault 1", pieces, dense_opts, nullptr,
                             &dense_selection));
    const std::string dense_generated =
        generate(&backend, &prefiller, tokens, dense_selection, 12);
    std::cout << "dense generated=" << dense_generated << "\n";
    CHECK(dense_generated.find("MAGENTA") != std::string::npos);

    LlamaBackend::shutdown();
}

TEST_CASE(integration_attn_top_k) {
    const std::string path = test_model_path();
    if (path.empty()) {
        std::cout << "SKIP integration_attn_top_k (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    LlamaBackend::init();
    LlamaBackend backend;
    CHECK_OK(backend.load_model(path));
    ContextOptions ctx;
    ctx.n_ctx = 2048;
    ctx.n_batch = 512;
    ctx.n_ubatch = 512;
    ctx.n_threads = 4;
    ctx.n_threads_batch = 4;
    // The attention top-k gather path requires non-flash attention
    // (ggml_get_rows produces F32 masks; flash attention requires F16).
    ctx.flash_attn = false;
    CHECK_OK(backend.create_context(ctx));

    std::string text;
    for (int i = 0; i < 120; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    std::vector<llama_token> tokens;
    CHECK_OK(backend.tokenize(text, &tokens));
    if (tokens.size() > 512) tokens.resize(512);
    CHECK(tokens.size() >= 256);

    // --- API validation ---
    CHECK(!backend.set_attn_top_k(-1).ok());
    CHECK_OK(backend.set_attn_top_k(0));
    CHECK_EQ(backend.attn_top_k(), 0);

    std::vector<int32_t> bad_not_ascending = {5, 3, 7};
    CHECK(!backend.set_attn_indices(bad_not_ascending).ok());
    std::vector<int32_t> bad_duplicate = {3, 3, 7};
    CHECK(!backend.set_attn_indices(bad_duplicate).ok());
    std::vector<int32_t> bad_out_of_range = {0, 100000};
    CHECK(!backend.set_attn_indices(bad_out_of_range).ok());
    CHECK_EQ(backend.attn_indices_size(), 0);

    // --- Boundary: top_k == n_tokens means no reduction -> gather skipped,
    // logits must be identical to dense (same code path). ---
    const auto dense = dense_run(&backend, tokens, 512);
    CHECK(!dense.empty());

    std::vector<int32_t> all_indices(tokens.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CHECK_OK(backend.set_attn_indices(all_indices));
    CHECK_EQ(backend.attn_indices_size(), static_cast<int32_t>(tokens.size()));
    CHECK_OK(backend.set_attn_top_k(static_cast<int32_t>(tokens.size())));

    const auto full_gather = dense_run(&backend, tokens, 512);
    CHECK(!full_gather.empty());
    CHECK_EQ(argmax(dense), argmax(full_gather));

    // --- Functional: top_k < n exercises the real gather path; logits must
    // remain finite and differ from dense (proving gather is active). ---
    std::vector<int32_t> subset_indices;
    // Keep every other position; strictly ascending by construction.
    for (size_t i = 0; i < tokens.size(); i += 2) {
        subset_indices.push_back(static_cast<int32_t>(i));
    }
    CHECK_OK(backend.set_attn_indices(subset_indices));
    CHECK_OK(backend.set_attn_top_k(static_cast<int32_t>(subset_indices.size())));

    const auto sparse = dense_run(&backend, tokens, 512);
    CHECK(!sparse.empty());
    {
        double worst = 0.0;
        for (size_t i = 0; i < sparse.size(); ++i) {
            CHECK(std::isfinite(sparse[i]));
            worst = std::max(worst, std::fabs(static_cast<double>(sparse[i])));
        }
        CHECK(worst < 1e6);
    }
    // Gather with half the positions must change the logits.
    CHECK(max_abs_diff(dense, sparse) > 1e-6);

    // --- Cleanup: disabling restores the dense path exactly. ---
    CHECK_OK(backend.set_attn_top_k(0));
    const auto restored = dense_run(&backend, tokens, 512);
    CHECK(!restored.empty());
    CHECK(max_abs_diff(dense, restored) < 1e-5);

    LlamaBackend::shutdown();
}

TEST_CASE(integration_real_model) {
    Loaded loaded;
    if (!load(&loaded)) {
        std::cout << "SKIP integration_real_model (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }

    std::vector<llama_pos> positions(loaded.tokens.size());

    std::iota(positions.begin(), positions.end(), 0);

    const auto full = dense_run(&loaded.backend, loaded.tokens, 512);
    CHECK(!full.empty());
    const auto chunked = dense_run(&loaded.backend, loaded.tokens, 64);
    CHECK(!chunked.empty());
    if (!full.empty() && !chunked.empty()) {
        CHECK(max_abs_diff(full, chunked) < 5e-3);
        CHECK_EQ(argmax(full), argmax(chunked));
    }

    uint64_t effective = 0;
    std::string warning;
    auto st = FfnSkipController::validate(loaded.backend, 0x3, &effective, &warning);
    const std::string arch = loaded.backend.arch();
    if (arch == "llama" || arch == "qwen2" || arch == "qwen3") {
        CHECK_OK(st);
        CHECK_EQ(effective, 0x3u);
    }

    SelectionOptions opts;
    opts.block_size = 32;
    opts.budget = 0.5;
    opts.min_dense_tokens = 128;
    std::vector<std::string> pieces(loaded.tokens.size());
    for (size_t i = 0; i < loaded.tokens.size(); ++i) {
        CHECK_OK(loaded.backend.token_to_piece(loaded.tokens[i], &pieces[i]));
    }
    Selection sel;
    BlockSelector selector;
    CHECK_OK(selector.select(loaded.tokens, "fox rain roof", pieces, opts, nullptr, &sel));
    CHECK(!sel.dense);
    CHECK(sel.selected_tokens <= static_cast<uint32_t>(std::ceil(loaded.tokens.size() * 0.5 * 1.06)));

    std::unique_ptr<PrefixCache> cache;
    CHECK_OK(PrefixCache::create(64ull * 1024 * 1024,
                                 loaded.backend.fingerprint().to_key(), &cache));
    SparsePrefiller prefiller(&loaded.backend, cache.get());
    SparsePrefiller::Options popts;
    popts.chunk_size = 512;
    popts.enable_prefix_cache = false;
    popts.enable_ffn_skip = false;
    PrefillStats stats;
    CHECK_OK(prefiller.prefill(loaded.tokens, sel, popts, &stats));
    CHECK_EQ(stats.selected_tokens, sel.selected_tokens);
    CHECK(stats.chunks >= 1);
    CHECK_EQ(stats.kv_pos_max, static_cast<int32_t>(loaded.tokens.size()) - 1);

    Selection full_sel;
    auto dense_opts = opts;
    dense_opts.budget = 1.0;
    CHECK_OK(selector.select(loaded.tokens, "", pieces, dense_opts, nullptr, &full_sel));
    CHECK(full_sel.dense);

    popts.enable_prefix_cache = true;
    CHECK_OK(prefiller.prefill(loaded.tokens, full_sel, popts, &stats));
    CHECK_OK(prefiller.prefill(loaded.tokens, full_sel, popts, &stats));
    CHECK_EQ(cache->stats().hits, 1u);
    CHECK_EQ(stats.chunks, 0u);

    LlamaBackend::shutdown();
}
