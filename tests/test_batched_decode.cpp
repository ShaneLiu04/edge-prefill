#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"

using namespace edge_prefill;

namespace {

std::string batch_model_path() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

// Builds `count` distinct token sequences from one piece of real text so the
// sequences are valid for this vocab without hand-picking token ids.
bool build_sequences(LlamaBackend* backend, int32_t count, int32_t length,
                     std::vector<std::vector<llama_token>>* out) {
    std::string text;
    for (int i = 0; i < 200; ++i) {
        text += "Vault " + std::to_string(i) + " uses a rotating passcode. ";
        text += "The keeper writes it on a card near the door. ";
    }
    std::vector<llama_token> all;
    if (!backend->tokenize(text, &all).ok()) return false;
    if (static_cast<int32_t>(all.size()) < count * length) return false;
    out->clear();
    for (int32_t s = 0; s < count; ++s) {
        const size_t start = static_cast<size_t>(s) * static_cast<size_t>(length);
        out->emplace_back(all.begin() + start, all.begin() + start + length);
    }
    return true;
}

std::vector<float> copy_logits(const LlamaBackend& backend, int32_t index) {
    const float* data = llama_get_logits_ith(backend.context(), index);
    if (!data) return {};
    return std::vector<float>(data, data + backend.n_vocab());
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    }
    return worst;
}

int32_t argmax(const std::vector<float>& v) {
    if (v.empty()) return -1;
    return static_cast<int32_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

} // namespace

// AR017 FR-1: one decode forward servicing several sequences must produce the
// same next-token distribution as decoding them one at a time. If this fails the
// whole batching design is wrong, so it is the first test.
TEST_CASE(batched_decode_matches_per_sequence_decode) {
    const std::string model = batch_model_path();
    if (model.empty()) {
        std::printf("SKIP batched_decode_matches_per_sequence_decode (no model)\n");
        return;
    }
    LlamaBackend backend;
    CHECK_OK(LlamaBackend::init());
    CHECK_OK(backend.load_model(model));
    ContextOptions opts;
    opts.n_ctx = 2048;
    opts.n_batch = 512;
    opts.n_ubatch = 128;
    opts.n_threads = 4;
    opts.n_threads_batch = 4;
    opts.n_seq_max = 4;
    CHECK_OK(backend.create_context(opts));

    std::vector<std::vector<llama_token>> seqs;
    CHECK(build_sequences(&backend, 3, 24, &seqs));

    // Serial arm: prefill each sequence on its own seq id, remember the logits of
    // the last prompt token, then decode one step.
    std::vector<std::vector<float>> serial;
    std::vector<llama_token> next_tokens;
    for (size_t s = 0; s < seqs.size(); ++s) {
        const int32_t seq = static_cast<int32_t>(s);
        CHECK_OK(backend.reset_sequence(seq));
        std::vector<llama_pos> positions(seqs[s].size());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = static_cast<llama_pos>(i);
        }
        PrefillStats stats;
        CHECK_OK(backend.prefill(seqs[s], positions, 0, &stats, seq));
        const std::vector<float> logits = copy_logits(backend, -1);
        CHECK(!logits.empty());
        const llama_token next = argmax(logits);
        next_tokens.push_back(next);
        std::vector<float> step;
        CHECK_OK(backend.decode_batch({next}, {seq},
                                      {static_cast<llama_pos>(seqs[s].size())},
                                      nullptr));
        step = copy_logits(backend, 0);
        CHECK(!step.empty());
        serial.push_back(std::move(step));
    }

    // Batched arm: same prefixes, then one decode_batch carrying all three next
    // tokens at once.
    for (size_t s = 0; s < seqs.size(); ++s) {
        const int32_t seq = static_cast<int32_t>(s);
        CHECK_OK(backend.reset_sequence(seq));
        std::vector<llama_pos> positions(seqs[s].size());
        for (size_t i = 0; i < positions.size(); ++i) {
            positions[i] = static_cast<llama_pos>(i);
        }
        PrefillStats stats;
        CHECK_OK(backend.prefill(seqs[s], positions, 0, &stats, seq));
        const std::vector<float> logits = copy_logits(backend, -1);
        CHECK_EQ(argmax(logits), next_tokens[s]);
    }
    std::vector<int32_t> seq_ids{0, 1, 2};
    std::vector<llama_pos> positions;
    for (const auto& s : seqs) positions.push_back(static_cast<llama_pos>(s.size()));
    CHECK_OK(backend.decode_batch(next_tokens, seq_ids, positions, nullptr));
    for (size_t s = 0; s < seqs.size(); ++s) {
        const std::vector<float> batched = copy_logits(backend, static_cast<int32_t>(s));
        CHECK(!batched.empty());
        const double diff = max_abs_diff(serial[s], batched);
        std::printf("  seq %zu: max|serial-batched| = %.6g\n", s, diff);
        CHECK(diff < 1e-3);
        CHECK_EQ(argmax(serial[s]), argmax(batched));
    }
    LlamaBackend::shutdown();
}

// AR017 FR-1: the batch primitive must reject ambiguous batches instead of
// silently letting two tokens overwrite one sequence's KV.
TEST_CASE(batched_decode_rejects_duplicate_sequence) {
    const std::string model = batch_model_path();
    if (model.empty()) {
        std::printf("SKIP batched_decode_rejects_duplicate_sequence (no model)\n");
        return;
    }
    LlamaBackend backend;
    CHECK_OK(LlamaBackend::init());
    CHECK_OK(backend.load_model(model));
    ContextOptions opts;
    opts.n_ctx = 512;
    opts.n_batch = 256;
    opts.n_ubatch = 64;
    opts.n_threads = 4;
    opts.n_threads_batch = 4;
    opts.n_seq_max = 4;
    CHECK_OK(backend.create_context(opts));

    const Status st = backend.decode_batch({1, 2, 3}, {0, 0, 1}, {0, 1, 0}, nullptr);
    CHECK(!st.ok());
    CHECK_EQ(static_cast<int>(st.code), static_cast<int>(ErrorCode::InvalidArgument));
    LlamaBackend::shutdown();
}
