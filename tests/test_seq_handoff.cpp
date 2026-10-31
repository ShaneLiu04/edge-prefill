#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "test_framework.h"

using namespace edge_prefill;

namespace {

std::string test_model_path() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

void argmax_diff(const std::vector<float>& a, const std::vector<float>& b,
                 float* max_diff) {
    CHECK_EQ(a.size(), b.size());
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    *max_diff = worst;
}

} // namespace

// AR018 FR-2: a sequence serialised out of one context and restored into a
// second context must continue identically. This is the whole basis of the
// prefill/decode pipeline, so it is asserted on logits, not on the bytes.
TEST_CASE(seq_state_handoff_is_lossless) {
    const std::string path = test_model_path();
    if (path.empty()) {
        std::cout << "SKIP seq_state_handoff_is_lossless (EDGE_PREFILL_TEST_MODEL unset)\n";
        return;
    }
    CHECK_OK(LlamaBackend::init());

    LlamaBackend src;
    LlamaBackend dst;
    CHECK_OK(src.load_model(path));
    CHECK_OK(dst.load_model(path));

    ContextOptions opts;
    opts.n_ctx = 256;
    opts.n_batch = 128;
    opts.n_ubatch = 128;
    opts.n_threads = 4;
    opts.n_threads_batch = 4;
    opts.flash_attn = false;
    CHECK_OK(src.create_context(opts));
    CHECK_OK(dst.create_context(opts));

    // A short prompt, prefilled into sequence 0 of each context. Both contexts
    // see the same tokens at the same positions, so a decode step must agree
    // even before the handoff -- this is the control case.
    std::vector<llama_token> tokens;
    CHECK_OK(src.tokenize("The quick brown fox jumps over the lazy dog.", &tokens));
    CHECK(!tokens.empty());

    std::vector<llama_pos> positions(tokens.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        positions[i] = static_cast<llama_pos>(i);
    }
    PrefillStats stats;
    CHECK_OK(src.prefill(tokens, positions, 0, &stats));
    CHECK_OK(dst.prefill(tokens, positions, 0, &stats));

    // Handoff: copy the source sequence and restore it into the destination.
    std::vector<uint8_t> state;
    CHECK_OK(src.get_seq_state(0, &state));
    CHECK(state.size() > 16);
    CHECK_OK(dst.set_seq_state(0, state));
    CHECK_EQ(dst.seq_pos_max(0), src.seq_pos_max(0));

    // Continue one token on both contexts and compare the distributions.
    const llama_pos pending_pos = src.seq_pos_max(0);
    const llama_token next = tokens.back();
    std::vector<float> src_logits;
    std::vector<float> dst_logits;
    CHECK_OK(src.decode(next, src.seq_pos_max(0) + 1, &src_logits));
    CHECK_OK(dst.decode(next, dst.seq_pos_max(0) + 1, &dst_logits));

    float max_diff = 0.0f;
    argmax_diff(src_logits, dst_logits, &max_diff);
    CHECK(max_diff <= 1e-6f);

    // Restoring over a dirty sequence must not keep stale cells: write a
    // different prompt into the destination first, then restore.
    std::vector<llama_token> other;
    CHECK_OK(dst.tokenize("Completely unrelated text for cache pollution.", &other));
    std::vector<llama_pos> other_pos(other.size());
    for (size_t i = 0; i < other_pos.size(); ++i) {
        other_pos[i] = static_cast<llama_pos>(i);
    }
    CHECK_OK(dst.reset_sequence(0));
    CHECK_OK(dst.prefill(other, other_pos, 0, &stats));
    CHECK_OK(dst.set_seq_state(0, state));
    // The source has advanced one token since the snapshot, so the restored
    // sequence must come back at the snapshotted position, not at src's head.
    CHECK_EQ(dst.seq_pos_max(0), pending_pos);
    dst_logits.clear();
    CHECK_OK(dst.decode(next, dst.seq_pos_max(0) + 1, &dst_logits));
    float pollute_diff = 0.0f;
    argmax_diff(src_logits, dst_logits, &pollute_diff);
    CHECK(pollute_diff <= 1e-6f);

    // A truncated payload must be rejected rather than silently accepted.
    std::vector<uint8_t> broken(state.begin(), state.begin() + state.size() / 2);
    CHECK(!dst.set_seq_state(1, broken).ok());

    LlamaBackend::shutdown();
}
