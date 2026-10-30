// AR015: a snapshot must continue a sequence exactly like re-running the
// prefill did. This is the test that lets the rest of the AR exist: if the
// restored state yields a different next-token distribution, no amount of
// gating would make reuse safe.
#include "test_framework.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "edge_prefill/backend.h"
#include "edge_prefill/kv_snapshot.h"

using namespace edge_prefill;

namespace {

std::string snapshot_test_model() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

std::string temp_dir(const char* tag) {
    return std::string("/tmp/edge_prefill_snap_") + tag + "_" +
           std::to_string(static_cast<long>(::getpid()));
}

ContextOptions snapshot_ctx() {
    ContextOptions opts;
    opts.n_ctx = 512;
    opts.n_batch = 256;
    opts.n_ubatch = 128;
    opts.n_threads = 4;
    opts.n_threads_batch = 4;
    return opts;
}

std::vector<float> logits_of(const LlamaBackend& backend) {
    const float* data = llama_get_logits_ith(backend.context(), -1);
    if (!data) return {};
    return std::vector<float>(data, data + backend.n_vocab());
}

SnapshotKey make_key(const LlamaBackend& backend, const std::vector<llama_token>& tokens,
                     const std::vector<llama_pos>& positions) {
    SnapshotKey key;
    key.model_key = backend.fingerprint().to_key();
    key.policy_key = "test-policy";
    key.kv_type = "f16";
    key.n_ctx = llama_n_ctx(backend.context());
    key.n_tokens = static_cast<uint32_t>(tokens.size());
    key.token_hash = hash_tokens(tokens);
    key.position_hash = hash_positions(positions);
    return key;
}

} // namespace

TEST_CASE(snapshot_restores_sequence_and_logits) {
    const std::string path = snapshot_test_model();
    if (path.empty()) {
        std::cout << "SKIP snapshot_restores_sequence_and_logits (no model)\n";
        return;
    }
    if (!LlamaBackend::init().ok()) {
        std::cout << "SKIP snapshot_restores_sequence_and_logits (init failed)\n";
        return;
    }
    LlamaBackend backend;
    CHECK_OK(backend.load_model(path));
    CHECK_OK(backend.create_context(snapshot_ctx()));

    std::string text;
    for (int i = 0; i < 40; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    std::vector<llama_token> tokens;
    CHECK_OK(backend.tokenize(text, &tokens));
    CHECK(tokens.size() >= 128);
    tokens.resize(128);

    // AR013 positions: the sparse path leaves gaps, so the snapshot must also
    // restore a sequence whose positions are not 0..n-1.
    std::vector<llama_pos> positions;
    for (size_t i = 0; i < tokens.size(); ++i) {
        positions.push_back(static_cast<llama_pos>(i + i / 8));
    }

    // Cold: prefill everything exactly like the bench does, and keep both the
    // sequence state and the last-token logits.
    CHECK_OK(backend.reset_sequence());
    PrefillStats stats;
    CHECK_OK(backend.prefill(tokens, positions, 256, &stats));
    const std::vector<float> cold_logits = logits_of(backend);
    CHECK(!cold_logits.empty());

    SnapshotPayload payload;
    CHECK_OK(backend.save_state(&payload.state));
    payload.logits = cold_logits;
    payload.n_vocab = static_cast<uint32_t>(cold_logits.size());

    const std::string dir = temp_dir("store");
    KvSnapshotStore* store = nullptr;
    CHECK_OK(KvSnapshotStore::create(dir, 0, &store));
    CHECK(store != nullptr);
    const SnapshotKey key = make_key(backend, tokens, positions);
    const SnapshotSaveResult saved = store->save(key, tokens, payload);
    CHECK(saved.ok);
    CHECK(saved.bytes > static_cast<uint64_t>(payload.state.size()));

    // Warm: fresh context, restore the payload, and continue generating. The
    // logits are the stored bytes, so the first token is identical by
    // construction; the KV must then continue identically as well.
    CHECK_OK(backend.create_context(snapshot_ctx()));
    SnapshotPayload loaded;
    const SnapshotLoadResult hit = store->load(key, tokens, &loaded);
    CHECK(hit.hit);
    CHECK_EQ(hit.state, std::string("warm"));
    CHECK_EQ(loaded.n_vocab, payload.n_vocab);
    CHECK_EQ(loaded.logits.size(), cold_logits.size());
    CHECK(std::equal(loaded.logits.begin(), loaded.logits.end(), cold_logits.begin()));
    CHECK_OK(backend.load_state(loaded.state));

    auto step = [&](const std::vector<float>& start_logits) {
        std::vector<llama_token> out;
        std::vector<float> logits = start_logits;
        for (int i = 0; i < 4; ++i) {
            const auto it = std::max_element(logits.begin(), logits.end());
            const llama_token next = static_cast<llama_token>(it - logits.begin());
            out.push_back(next);
            std::vector<float> step_logits;
            if (!backend.decode(next, positions.back() + 1 + i, &step_logits).ok()) break;
            logits = step_logits;
        }
        return out;
    };
    const std::vector<llama_token> warm_tokens = step(loaded.logits);

    // Cold continuation in a fresh context, from scratch.
    CHECK_OK(backend.create_context(snapshot_ctx()));
    CHECK_OK(backend.prefill(tokens, positions, 256, &stats));
    const std::vector<llama_token> cold_tokens = step(logits_of(backend));
    CHECK_EQ(warm_tokens.size(), cold_tokens.size());
    CHECK(std::equal(warm_tokens.begin(), warm_tokens.end(), cold_tokens.begin()));

    delete store;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_CASE(snapshot_misses_are_explicit) {
    const std::string path = snapshot_test_model();
    if (path.empty()) {
        std::cout << "SKIP snapshot_misses_are_explicit (no model)\n";
        return;
    }
    if (!LlamaBackend::init().ok()) {
        std::cout << "SKIP snapshot_misses_are_explicit (init failed)\n";
        return;
    }
    LlamaBackend backend;
    CHECK_OK(backend.load_model(path));
    CHECK_OK(backend.create_context(snapshot_ctx()));

    const std::string dir = temp_dir("miss");
    KvSnapshotStore* store = nullptr;
    CHECK_OK(KvSnapshotStore::create(dir, 0, &store));

    std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
    std::vector<llama_pos> positions = {0, 1, 2, 3, 4};
    SnapshotKey key = make_key(backend, tokens, positions);
    SnapshotPayload payload;
    payload.state.assign(64, 0x5a);
    payload.logits.assign(8, 0.5f);
    payload.n_vocab = 8;

    SnapshotPayload out;
    SnapshotLoadResult miss = store->load(key, tokens, &out);
    CHECK(!miss.hit);
    CHECK_EQ(miss.state, std::string("miss:absent"));

    CHECK(store->save(key, tokens, payload).ok);
    SnapshotLoadResult hit = store->load(key, tokens, &out);
    CHECK(hit.hit);

    // A different log under the same key is a token mismatch, never a reuse.
    std::vector<llama_token> other = {1, 2, 3, 4, 6};
    SnapshotLoadResult wrong = store->load(key, other, &out);
    CHECK(!wrong.hit);
    CHECK_EQ(wrong.state, std::string("miss:tokens"));

    // Corrupt payload: same size, different bytes.
    const std::string kv_path = dir + "/" + key.hex() + ".kv";
    std::vector<uint8_t> corrupt(64, 0xa5);
    {
        std::FILE* f = std::fopen(kv_path.c_str(), "wb");
        CHECK(f != nullptr);
        if (f) {
            std::fwrite(corrupt.data(), 1, corrupt.size(), f);
            std::fclose(f);
        }
    }
    SnapshotLoadResult bad = store->load(key, tokens, &out);
    CHECK(!bad.hit);
    CHECK_EQ(bad.state, std::string("miss:corrupt"));

    delete store;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}


namespace {

SnapshotKey handmade_key() {
    SnapshotKey key;
    key.model_key = "model-a";
    key.policy_key = "policy-a";
    key.kv_type = "f16";
    key.n_tokens = 4;
    key.n_ctx = 512;
    key.token_hash = hash_tokens({1, 2, 3, 4});
    key.position_hash = hash_positions({0, 1, 2, 3});
    return key;
}

} // namespace

TEST_CASE(snapshot_key_is_stable_and_sensitive) {
    const SnapshotKey base = handmade_key();
    CHECK_EQ(base.hex().size(), 32u);
    CHECK_EQ(base.hex(), handmade_key().hex());

    // Every element of the material must change the name, otherwise a stale
    // snapshot could be reused under a different model or policy.
    std::vector<SnapshotKey> variants;
    SnapshotKey k = base;
    k.model_key = "model-b";
    variants.push_back(k);
    k = base;
    k.policy_key = "policy-b";
    variants.push_back(k);
    k = base;
    k.kv_type = "q8_0";
    variants.push_back(k);
    k = base;
    k.n_ctx = 1024;
    variants.push_back(k);
    k = base;
    k.n_tokens = 5;
    variants.push_back(k);
    k = base;
    k.token_hash = base.token_hash + 1;
    variants.push_back(k);
    k = base;
    k.position_hash = base.position_hash + 1;
    variants.push_back(k);
    for (const SnapshotKey& variant : variants) {
        CHECK(variant.hex() != base.hex());
    }
}

TEST_CASE(snapshot_manifest_roundtrip_and_rejects) {
    SnapshotKey key = handmade_key();
    SnapshotManifest m;
    m.model_key = key.model_key;
    m.policy_key = key.policy_key;
    m.kv_type = key.kv_type;
    m.token_hash = key.token_hash;
    m.position_hash = key.position_hash;
    m.state_hash = 0x1234;
    m.bytes = 4096;
    m.logits_bytes = 1024;
    m.n_vocab = 8;
    m.n_tokens = key.n_tokens;
    m.n_ctx = key.n_ctx;
    m.created = 1700000000;

    SnapshotManifest parsed;
    CHECK(SnapshotManifest::parse(m.serialize(), &parsed));
    CHECK_EQ(parsed.model_key, m.model_key);
    CHECK_EQ(parsed.policy_key, m.policy_key);
    CHECK_EQ(parsed.n_tokens, m.n_tokens);
    CHECK_EQ(parsed.n_ctx, m.n_ctx);
    CHECK_EQ(parsed.token_hash, m.token_hash);
    CHECK_EQ(parsed.position_hash, m.position_hash);
    CHECK_EQ(parsed.state_hash, m.state_hash);
    CHECK_EQ(parsed.bytes, m.bytes);
    CHECK(parsed.matches(key, 4096, 1024));
    CHECK(!parsed.matches(key, 4097, 1024));

    CHECK(!SnapshotManifest::parse("model_key=x\n", &parsed));
    CHECK(!SnapshotManifest::parse("version=v9\nmodel_key=x\n", &parsed));
    CHECK(!SnapshotManifest::parse("version=v1\nbroken line\n", &parsed));
    CHECK(!SnapshotManifest::parse("version=v1\nbytes=abc\n", &parsed));
}

TEST_CASE(snapshot_store_evicts_oldest_and_ignores_tmp) {
    const std::string dir = temp_dir("evict");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    // Every snapshot costs 100 bytes of state plus a ~200 byte manifest, so a
    // 700 byte budget holds two of them.
    KvSnapshotStore* store = nullptr;
    CHECK_OK(KvSnapshotStore::create(dir, 700, &store));

    const SnapshotKey key = handmade_key();
    const std::vector<llama_token> tokens = {1, 2, 3, 4};
    SnapshotPayload payload;
    payload.state.assign(100, 0x11);
    payload.logits.assign(4, 0.25f);
    payload.n_vocab = 4;
    CHECK(store->save(key, tokens, payload).ok);

    // A stale temp file must not count towards the limit.
    {
        std::FILE* f = std::fopen((dir + "/stale.kv.tmp").c_str(), "wb");
        CHECK(f != nullptr);
        if (f) {
            std::vector<uint8_t> junk(1000, 0x22);
            std::fwrite(junk.data(), 1, junk.size(), f);
            std::fclose(f);
        }
    }
    CHECK(store->usage_bytes() < 700);

    // Two more snapshots push the directory over the limit; the oldest one (the
    // key saved first) is evicted, the newest still hits.
    SnapshotKey newest = key;
    std::vector<llama_token> newest_tokens = tokens;
    for (int i = 0; i < 2; ++i) {
        std::vector<llama_token> other_tokens = {1, 2, 3, static_cast<llama_token>(5 + i)};
        SnapshotKey other = key;
        other.token_hash = hash_tokens(other_tokens);
        CHECK(store->save(other, other_tokens, payload).ok);
        newest = other;
        newest_tokens = other_tokens;
    }
    CHECK(store->usage_bytes() <= 700);

    SnapshotPayload out;
    CHECK(store->load(newest, newest_tokens, &out).hit);
    CHECK(!store->load(key, tokens, &out).hit);
    delete store;
    std::filesystem::remove_all(dir, ec);
}
