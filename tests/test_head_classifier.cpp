#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/head_classifier.h"
#include "edge_prefill/head_profile.h"
#include "edge_prefill/status.h"

using namespace edge_prefill;

namespace {

std::string test_model_path() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

// Loads the real model with a small context so the n_ctx boundary cases are
// cheap to trigger, and hardware-concurrency threads so the per-head probe
// sweep finishes fast.
bool load_backend(LlamaBackend* backend, uint32_t n_ctx) {
    const std::string path = test_model_path();
    if (path.empty()) return false;
    if (!LlamaBackend::init().ok()) return false;
    if (!backend->load_model(path).ok()) return false;
    ContextOptions opts;
    opts.n_ctx = n_ctx;
    opts.n_batch = n_ctx;
    opts.n_ubatch = 128;
    // 0 = let llama.cpp pick hardware concurrency.
    opts.n_threads = 0;
    opts.n_threads_batch = 0;
    // Per-head sparse attention goes through the non-flash gather path.
    opts.flash_attn = false;
    return backend->create_context(opts).ok();
}

// Probe geometry for the real-model cases: an 8x4 repeated sequence (32
// tokens). Shorter probes dilute single-head ablation below measurability
// (the remaining dense heads compensate and no top-1 flips), so this is the
// smallest shape that still yields real flip-rate signal while keeping the
// 393-forward sweep affordable.
HeadProbeParams small_params() {
    HeadProbeParams params;
    params.block_len = 8;
    params.repeats = 4;
    params.sink = 2;
    params.recent = 4;
    params.probe_k = 4;
    params.seed = 7;
    return params;
}

std::string profile_path(const std::string& name) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path() / "edge_prefill_ar003_t002_head_classifier";
    // Idempotent: several saves share the directory within one test.
    fs::create_directories(dir, ec);
    return (dir / name).string();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::string();
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

bool message_mentions(const Status& st, const std::string& needle) {
    return st.message.find(needle) != std::string::npos;
}

// Validation messages may name a field after the struct member ("block_len")
// or the CLI flag ("probe-block-len"); pin the presence of the name, not its
// spelling.
bool message_mentions_field(const Status& st, const std::string& snake,
                            const std::string& hyphen) {
    return message_mentions(st, snake) || message_mentions(st, hyphen);
}

void check_heads_complete_and_ordered(const HeadProfile& profile) {
    CHECK_EQ(profile.heads.size(),
             static_cast<size_t>(profile.n_layer) * static_cast<size_t>(profile.n_head));
    for (int32_t layer = 0; layer < profile.n_layer; ++layer) {
        for (int32_t head = 0; head < profile.n_head; ++head) {
            const size_t idx =
                static_cast<size_t>(layer) * static_cast<size_t>(profile.n_head) +
                static_cast<size_t>(head);
            CHECK_EQ(profile.heads[idx].layer, layer);
            CHECK_EQ(profile.heads[idx].head, head);
        }
    }
}

} // namespace

TEST_CASE(classify_profile_structure) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_profile_structure (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }

    const HeadProbeParams params = small_params();
    HeadClassifier classifier(0.10);
    HeadProfile profile;
    CHECK_OK(classifier.classify(&backend, params, "sha-test-ar003", &profile));

    CHECK_EQ(profile.model_sha, std::string("sha-test-ar003"));
    CHECK_EQ(profile.n_layer, backend.n_layer());
    CHECK_EQ(profile.n_head, backend.n_head());
    check_heads_complete_and_ordered(profile);

    CHECK_EQ(profile.probe.block_len, params.block_len);
    CHECK_EQ(profile.probe.repeats, params.repeats);
    CHECK_EQ(profile.probe.sink, params.sink);
    CHECK_EQ(profile.probe.recent, params.recent);
    CHECK_EQ(profile.probe.probe_k, params.probe_k);
    CHECK_EQ(profile.probe.seed, params.seed);

    bool any_positive = false;
    for (int32_t layer = 0; layer < profile.n_layer; ++layer) {
        for (int32_t head = 0; head < profile.n_head; ++head) {
            const size_t idx =
                static_cast<size_t>(layer) * static_cast<size_t>(profile.n_head) +
                static_cast<size_t>(head);
            const HeadProfileEntry& entry = profile.heads[idx];
            CHECK(std::isfinite(entry.echo_score));
            CHECK(std::isfinite(entry.induction_score));
            CHECK(entry.echo_score >= 0.0 && entry.echo_score <= 1.0);
            CHECK(entry.induction_score >= 0.0 && entry.induction_score <= 1.0);
            // retrieval <=> max(echo, induction) > threshold, by contract.
            CHECK_EQ(entry.retrieval,
                     std::max(entry.echo_score, entry.induction_score) > 0.10);
            if (std::max(entry.echo_score, entry.induction_score) > 0.0) {
                any_positive = true;
            }
        }
    }
    // Scores must come from real forwards: an all-zero synthetic profile
    // would pass every structural check above, so require at least one head
    // whose probed top-1 actually flipped somewhere.
    CHECK(any_positive);
}

TEST_CASE(classify_deterministic_output) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_deterministic_output (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }

    const HeadProbeParams params = small_params();
    HeadClassifier classifier(0.10);
    HeadProfile first;
    HeadProfile second;
    CHECK_OK(classifier.classify(&backend, params, "sha-det-ar003", &first));
    CHECK_OK(classifier.classify(&backend, params, "sha-det-ar003", &second));

    CHECK_EQ(first.heads.size(), second.heads.size());
    for (size_t i = 0; i < first.heads.size(); ++i) {
        CHECK_EQ(first.heads[i].layer, second.heads[i].layer);
        CHECK_EQ(first.heads[i].head, second.heads[i].head);
        // Exact equality: determinism must survive identical reruns.
        CHECK_EQ(first.heads[i].echo_score, second.heads[i].echo_score);
        CHECK_EQ(first.heads[i].induction_score, second.heads[i].induction_score);
        CHECK_EQ(first.heads[i].retrieval, second.heads[i].retrieval);
    }

    const std::string path_first = profile_path("deterministic_first.json");
    const std::string path_second = profile_path("deterministic_second.json");
    CHECK_OK(save_head_profile(path_first, first));
    CHECK_OK(save_head_profile(path_second, second));
    CHECK_EQ(read_file(path_first), read_file(path_second));

    HeadProfile reloaded;
    CHECK_OK(load_head_profile(path_first, &reloaded));
    CHECK_EQ(reloaded.model_sha, std::string("sha-det-ar003"));
    CHECK_EQ(reloaded.heads.size(), first.heads.size());
}

TEST_CASE(classify_threshold_extremes) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_threshold_extremes (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }
    const HeadProbeParams params = small_params();

    // Scores live in [0,1]: a threshold above 1 can never be exceeded.
    HeadClassifier unreachable(2.0);
    HeadProfile all_normal;
    CHECK_OK(unreachable.classify(&backend, params, "sha-thr-ar003", &all_normal));
    check_heads_complete_and_ordered(all_normal);
    for (const HeadProfileEntry& entry : all_normal.heads) {
        CHECK(!entry.retrieval);
    }

    // A negative threshold is exceeded even by all-zero scores, so every
    // head must be flagged retrieval.
    HeadClassifier always(-1.0);
    HeadProfile all_retrieval;
    CHECK_OK(always.classify(&backend, params, "sha-thr-ar003", &all_retrieval));
    check_heads_complete_and_ordered(all_retrieval);
    for (const HeadProfileEntry& entry : all_retrieval.heads) {
        CHECK(entry.retrieval);
    }
}

TEST_CASE(classify_probe_exceeds_n_ctx_fails) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_probe_exceeds_n_ctx_fails (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }
    HeadProbeParams params = small_params();
    // The runtime pads a small n_ctx request up (e.g. 64 -> 256), so the
    // probe must exceed the *effective* context, not the requested one:
    // a double-size block makes probe_len strictly greater than whatever
    // n_ctx the backend reports.
    params.block_len = backend.n_ctx();
    params.repeats = 2;  // probe_len = 2 * n_ctx > n_ctx always
    HeadClassifier classifier(0.10);
    HeadProfile profile;
    const Status st = classifier.classify(&backend, params, "sha-ctx-ar003", &profile);
    CHECK(!st.ok());
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions(st, "n_ctx"));
    CHECK(message_mentions(st, "probe-repeats") ||
          message_mentions(st, "probe-block-len"));
}

TEST_CASE(classify_probe_exactly_n_ctx_runs) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_probe_exactly_n_ctx_runs (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }
    HeadProbeParams params = small_params();
    params.block_len = 32;
    params.repeats = 2;  // 32 * 2 = 64 == n_ctx: boundary must be legal
    HeadClassifier classifier(0.10);
    HeadProfile profile;
    CHECK_OK(classifier.classify(&backend, params, "sha-fit-ar003", &profile));
    check_heads_complete_and_ordered(profile);
}

TEST_CASE(classify_invalid_params_rejected) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP classify_invalid_params_rejected (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }
    const std::string sha = "sha-invalid-ar003";
    HeadClassifier classifier(0.10);
    HeadProfile profile;

    HeadProbeParams params = small_params();
    params.repeats = 1;
    Status st = classifier.classify(&backend, params, sha, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions_field(st, "repeats", "probe-repeats"));

    params = small_params();
    params.block_len = 0;
    st = classifier.classify(&backend, params, sha, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions_field(st, "block_len", "block-len"));

    params = small_params();
    params.probe_k = 0;
    st = classifier.classify(&backend, params, sha, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions_field(st, "probe_k", "probe-k"));

    params = small_params();
    params.sink = 0;
    params.recent = 0;
    st = classifier.classify(&backend, params, sha, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions(st, "sink") || message_mentions(st, "recent"));

    st = classifier.classify(nullptr, small_params(), sha, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions(st, "backend"));

    st = classifier.classify(&backend, small_params(), sha, nullptr);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions(st, "out"));
}

TEST_CASE(probe_argmax_empty_tokens_fails) {
    LlamaBackend backend;
    if (!load_backend(&backend, 64)) {
        std::cout << "SKIP probe_argmax_empty_tokens_fails (EDGE_PREFILL_TEST_MODEL unset or load failed)\n";
        return;
    }
    std::vector<llama_token> tokens;
    std::vector<int32_t> top1;
    CHECK_ERR(backend.probe_argmax(tokens, &top1), ErrorCode::InvalidArgument);

    // Cheap positive path: one short real forward must yield exactly one
    // argmax per position, each a valid vocab id.
    CHECK_OK(backend.tokenize("The quick brown fox jumps over the lazy dog.", &tokens));
    CHECK(!tokens.empty());
    CHECK_OK(backend.probe_argmax(tokens, &top1));
    CHECK_EQ(top1.size(), tokens.size());
    for (int32_t id : top1) {
        CHECK(id >= 0);
        CHECK(id < backend.n_vocab());
    }
}

TEST_CASE(save_head_profile_roundtrip) {
    HeadProfile profile;
    profile.model_sha = "sha-roundtrip-ar003";
    profile.n_layer = 4;
    profile.n_head = 2;
    // Non-default probe values so a saver that silently falls back to the
    // struct defaults cannot pass by accident.
    profile.probe.block_len = 12;
    profile.probe.repeats = 3;
    profile.probe.sink = 1;
    profile.probe.recent = 5;
    profile.probe.probe_k = 6;
    profile.probe.seed = 99;
    profile.heads.push_back(HeadProfileEntry{0, 0, 0.123456, 0.034521, true});
    profile.heads.push_back(HeadProfileEntry{3, 1, 0.077777, 0.211122, false});

    const std::string path_first = profile_path("roundtrip_first.json");
    CHECK_OK(save_head_profile(path_first, profile));

    HeadProfile loaded;
    CHECK_OK(load_head_profile(path_first, &loaded));
    CHECK_EQ(loaded.model_sha, std::string("sha-roundtrip-ar003"));
    CHECK_EQ(loaded.n_layer, 4);
    CHECK_EQ(loaded.n_head, 2);
    CHECK_EQ(loaded.probe.block_len, 12u);
    CHECK_EQ(loaded.probe.repeats, 3u);
    CHECK_EQ(loaded.probe.sink, 1u);
    CHECK_EQ(loaded.probe.recent, 5u);
    CHECK_EQ(loaded.probe.probe_k, 6u);
    CHECK_EQ(loaded.probe.seed, 99ull);
    CHECK_EQ(loaded.heads.size(), 2u);
    CHECK_EQ(loaded.heads[0].layer, 0);
    CHECK_EQ(loaded.heads[0].head, 0);
    CHECK_NEAR(loaded.heads[0].echo_score, 0.123456, 1e-9);
    CHECK_NEAR(loaded.heads[0].induction_score, 0.034521, 1e-9);
    CHECK(loaded.heads[0].retrieval);
    CHECK_EQ(loaded.heads[1].layer, 3);
    CHECK_EQ(loaded.heads[1].head, 1);
    CHECK_NEAR(loaded.heads[1].echo_score, 0.077777, 1e-9);
    CHECK_NEAR(loaded.heads[1].induction_score, 0.211122, 1e-9);
    CHECK(!loaded.heads[1].retrieval);

    // Deterministic serialization: the same profile must produce a
    // byte-identical file on a second save.
    const std::string path_second = profile_path("roundtrip_second.json");
    CHECK_OK(save_head_profile(path_second, profile));
    CHECK_EQ(read_file(path_first), read_file(path_second));
}

TEST_CASE(save_head_profile_rejects_unwritable_path) {
    HeadProfile profile;
    profile.model_sha = "sha-io-ar003";
    profile.n_layer = 1;
    profile.n_head = 1;

    // The target path is a directory: opening it as a file for writing fails
    // on every supported platform, and the saver must surface that as Io.
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir =
        fs::temp_directory_path() / "edge_prefill_ar003_t002_head_classifier_io_dir";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    CHECK(fs::is_directory(dir));
    CHECK_ERR(save_head_profile(dir.string(), profile), ErrorCode::Io);
}
