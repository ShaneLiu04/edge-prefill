#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/status.h"

namespace edge_prefill {

// AR015: content-addressed KV snapshots.
//
// A snapshot is only valid for one exact sequence: the same model build, the
// same admission policy, the same KV precision, the same context size and the
// same tokens at the same positions. Anything else would silently feed the
// model someone else's KV, so every one of those inputs is part of the key and
// is re-checked against the manifest on load.
struct SnapshotKey {
    std::string model_key;   // GGUF sha256 + geometry + llama.cpp commit
    std::string policy_key;  // admission knobs (mode/budget/spans/compaction...)
    std::string kv_type;     // f16 / q8_0 / ...
    uint64_t token_hash = 0;
    uint64_t position_hash = 0;
    uint32_t n_tokens = 0;
    uint32_t n_ctx = 0;

    // 32 lowercase hex chars; used as the file stem.
    std::string hex() const;
    std::string material() const;
};

// FNV-1a over the log, used for both hashes. Not a security boundary: the
// manifest check exists to catch stale or corrupted caches, not attackers.
uint64_t hash_tokens(const std::vector<llama_token>& tokens);
uint64_t hash_positions(const std::vector<llama_pos>& positions);
uint64_t hash_bytes(const uint8_t* data, size_t size);

// What a loaded snapshot hands back: the sequence state plus the logits of the
// last prompt token. Storing the logits makes the warm path compute-free and
// bit-identical to the cold path (a 1-token decode to rebuild them costs more
// than the snapshot itself and drifts with the batch shape).
struct SnapshotPayload {
    std::vector<uint8_t> state;
    std::vector<float> logits;
    uint32_t n_vocab = 0;
};

struct SnapshotSaveResult {
    bool ok = false;
    uint64_t bytes = 0;
    double save_ms = 0.0;
    std::string reason;
};

struct SnapshotLoadResult {
    bool hit = false;
    uint64_t bytes = 0;
    double load_ms = 0.0;
    // "warm" on a hit; otherwise "miss:absent" / "miss:manifest" / "miss:size"
    // / "miss:corrupt" / "miss:tokens".
    std::string state;
};

// Process-independent snapshot directory. Files are `<key>.kv` (opaque state)
// and `<key>.meta` (line-based manifest, written last so a crash cannot leave a
// manifest pointing at a half-written state).
class KvSnapshotStore {
public:
    static Status create(const std::string& dir, uint64_t max_bytes,
                         KvSnapshotStore** out);

    SnapshotSaveResult save(const SnapshotKey& key,
                            const std::vector<llama_token>& tokens,
                            const SnapshotPayload& payload);

    // Returns the payload for the caller to restore. `expect` is the log the
    // caller is about to continue; a mismatch is a miss, never a silent reuse.
    SnapshotLoadResult load(const SnapshotKey& key,
                            const std::vector<llama_token>& expect,
                            SnapshotPayload* payload);

    const std::string& dir() const { return dir_; }
    uint64_t max_bytes() const { return max_bytes_; }

    // Total size of snapshot files currently in the directory.
    uint64_t usage_bytes() const;

private:
    KvSnapshotStore(std::string dir, uint64_t max_bytes)
        : dir_(std::move(dir)), max_bytes_(max_bytes) {}
    std::string path_for(const SnapshotKey& key, const char* ext) const;
    void evict_if_needed(const SnapshotKey& keep);

    std::string dir_;
    uint64_t max_bytes_ = 0;
};

// Manifest fields, exposed so the key/manifest contract can be unit tested
// without a model or a llama context.
struct SnapshotManifest {
    std::string model_key;
    std::string policy_key;
    std::string kv_type;
    uint64_t token_hash = 0;
    uint64_t position_hash = 0;
    uint64_t state_hash = 0;
    uint64_t logits_hash = 0;
    uint64_t bytes = 0;
    uint64_t logits_bytes = 0;
    uint32_t n_vocab = 0;
    uint32_t n_tokens = 0;
    uint32_t n_ctx = 0;
    uint64_t created = 0;

    std::string serialize() const;
    static bool parse(const std::string& text, SnapshotManifest* out);
    bool matches(const SnapshotKey& key, uint64_t state_bytes,
                 uint64_t logits_bytes) const;
};

} // namespace edge_prefill
