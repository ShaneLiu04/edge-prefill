#include "edge_prefill/kv_snapshot.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "edge_prefill/log.h"

namespace fs = std::filesystem;

namespace edge_prefill {

namespace {

constexpr const char* kSnapshotVersion = "v1";
constexpr const char* kStateExt = ".kv";
constexpr const char* kMetaExt = ".meta";
constexpr const char* kLogitsExt = ".lg";
constexpr const char* kLogitsMagic = "LGF1";

std::string hex64(uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
    return std::string(buf);
}

// Dual-seed FNV-1a: two independent lanes give a 128-bit name, which is plenty
// to keep distinct sequences apart in a local cache directory.
void fnv_lanes(const std::string& text, uint64_t* a, uint64_t* b) {
    uint64_t h1 = 0xcbf29ce484222325ULL;
    uint64_t h2 = 0x9ae16a3b2f90404fULL;
    for (unsigned char c : text) {
        h1 ^= c;
        h1 *= 0x100000001b3ULL;
        h2 ^= static_cast<uint64_t>(c) + 0x9e3779b97f4a7c15ULL;
        h2 *= 0x100000001b3ULL;
        h2 ^= h2 >> 29;
    }
    *a = h1;
    *b = h2;
}

bool read_file(const std::string& path, std::vector<uint8_t>* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) return false;
    in.seekg(0, std::ios::beg);
    out->resize(static_cast<size_t>(size));
    if (size > 0) {
        in.read(reinterpret_cast<char*>(out->data()), size);
        if (!in) return false;
    }
    return true;
}

bool write_file_atomic(const std::string& path, const uint8_t* data, size_t size) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        if (size > 0) out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        out.flush();
        if (!out) return false;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

bool write_text_atomic(const std::string& path, const std::string& text) {
    return write_file_atomic(path, reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

double ms_since(const std::chrono::steady_clock::time_point& begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count();
}

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        std::string line = text.substr(start, end == std::string::npos ? std::string::npos
                                                                       : end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) lines.push_back(line);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return lines;
}

bool parse_u64(const std::string& text, uint64_t* out) {
    if (text.empty()) return false;
    uint64_t value = 0;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
        value = value * 10 + static_cast<uint64_t>(c - '0');
    }
    *out = value;
    return true;
}

bool parse_u64_hex(const std::string& text, uint64_t* out) {
    if (text.empty() || text.size() > 16) return false;
    uint64_t value = 0;
    for (char c : text) {
        uint64_t digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<uint64_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<uint64_t>(c - 'a' + 10);
        } else {
            return false;
        }
        value = (value << 4) | digit;
    }
    *out = value;
    return true;
}

} // namespace

uint64_t hash_bytes(const uint8_t* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t hash_tokens(const std::vector<llama_token>& tokens) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (llama_token token : tokens) {
        const uint32_t v = static_cast<uint32_t>(token);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>((v >> (8 * i)) & 0xffu);
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

uint64_t hash_positions(const std::vector<llama_pos>& positions) {
    uint64_t h = 0x9ae16a3b2f90404fULL;
    for (llama_pos position : positions) {
        const uint32_t v = static_cast<uint32_t>(position);
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<uint8_t>((v >> (8 * i)) & 0xffu);
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

std::string SnapshotKey::material() const {
    std::string out;
    out.reserve(model_key.size() + policy_key.size() + 64);
    out += kSnapshotVersion;
    out += "|model=" + model_key;
    out += "|policy=" + policy_key;
    out += "|kv=" + kv_type;
    out += "|ctx=" + std::to_string(n_ctx);
    out += "|n=" + std::to_string(n_tokens);
    out += "|t=" + hex64(token_hash);
    out += "|p=" + hex64(position_hash);
    return out;
}

std::string SnapshotKey::hex() const {
    uint64_t a = 0;
    uint64_t b = 0;
    fnv_lanes(material(), &a, &b);
    return hex64(a) + hex64(b);
}

std::string SnapshotManifest::serialize() const {
    std::string out;
    out += std::string("version=") + kSnapshotVersion + "\n";
    out += "model_key=" + model_key + "\n";
    out += "policy_key=" + policy_key + "\n";
    out += "kv_type=" + kv_type + "\n";
    out += "n_tokens=" + std::to_string(n_tokens) + "\n";
    out += "n_ctx=" + std::to_string(n_ctx) + "\n";
    out += "token_hash=" + hex64(token_hash) + "\n";
    out += "position_hash=" + hex64(position_hash) + "\n";
    out += "state_hash=" + hex64(state_hash) + "\n";
    out += "bytes=" + std::to_string(bytes) + "\n";
    out += "logits_hash=" + hex64(logits_hash) + "\n";
    out += "logits_bytes=" + std::to_string(logits_bytes) + "\n";
    out += "n_vocab=" + std::to_string(n_vocab) + "\n";
    out += "created=" + std::to_string(created) + "\n";
    return out;
}

bool SnapshotManifest::parse(const std::string& text, SnapshotManifest* out) {
    if (!out) return false;
    SnapshotManifest m;
    bool have_version = false;
    for (const std::string& line : split_lines(text)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) return false;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        uint64_t number = 0;
        if (key == "version") {
            if (value != kSnapshotVersion) return false;
            have_version = true;
        } else if (key == "model_key") {
            m.model_key = value;
        } else if (key == "policy_key") {
            m.policy_key = value;
        } else if (key == "kv_type") {
            m.kv_type = value;
        } else if (key == "n_tokens") {
            if (!parse_u64(value, &number)) return false;
            m.n_tokens = static_cast<uint32_t>(number);
        } else if (key == "n_ctx") {
            if (!parse_u64(value, &number)) return false;
            m.n_ctx = static_cast<uint32_t>(number);
        } else if (key == "token_hash") {
            if (!parse_u64_hex(value, &m.token_hash)) return false;
        } else if (key == "position_hash") {
            if (!parse_u64_hex(value, &m.position_hash)) return false;
        } else if (key == "state_hash") {
            if (!parse_u64_hex(value, &m.state_hash)) return false;
        } else if (key == "bytes") {
            if (!parse_u64(value, &m.bytes)) return false;
        } else if (key == "logits_hash") {
            if (!parse_u64_hex(value, &m.logits_hash)) return false;
        } else if (key == "logits_bytes") {
            if (!parse_u64(value, &m.logits_bytes)) return false;
        } else if (key == "n_vocab") {
            if (!parse_u64(value, &number)) return false;
            m.n_vocab = static_cast<uint32_t>(number);
        } else if (key == "created") {
            if (!parse_u64(value, &m.created)) return false;
        }
    }
    if (!have_version) return false;
    *out = m;
    return true;
}

bool SnapshotManifest::matches(const SnapshotKey& key, uint64_t state_bytes,
                               uint64_t logits_bytes_in) const {
    return model_key == key.model_key && policy_key == key.policy_key &&
           kv_type == key.kv_type && n_tokens == key.n_tokens && n_ctx == key.n_ctx &&
           token_hash == key.token_hash && position_hash == key.position_hash &&
           bytes == state_bytes && logits_bytes == logits_bytes_in;
}

Status KvSnapshotStore::create(const std::string& dir, uint64_t max_bytes,
                               KvSnapshotStore** out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null store output");
    if (dir.empty()) return Status::error(ErrorCode::InvalidArgument, "empty snapshot dir");
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec && !fs::is_directory(dir)) {
        return Status::error(ErrorCode::Io, "cannot create snapshot dir: " + dir);
    }
    *out = new KvSnapshotStore(dir, max_bytes);
    return Status::success();
}

std::string KvSnapshotStore::path_for(const SnapshotKey& key, const char* ext) const {
    return dir_ + "/" + key.hex() + ext;
}

uint64_t KvSnapshotStore::usage_bytes() const {
    uint64_t total = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.size() < 4) continue;
        if (name.compare(name.size() - 4, 4, ".tmp") == 0) continue;
        total += static_cast<uint64_t>(entry.file_size(ec));
    }
    return total;
}

void KvSnapshotStore::evict_if_needed(const SnapshotKey& keep) {
    if (max_bytes_ == 0) return;
    if (usage_bytes() <= max_bytes_) return;

    // Group by file stem so a manifest is never deleted without its state.
    struct Stem {
        std::string stem;
        std::filesystem::file_time_type mtime;
        uint64_t bytes = 0;
    };
    std::vector<Stem> stems;
    std::error_code ec;
    for (const auto& dir_entry : fs::directory_iterator(dir_, ec)) {
        if (ec) break;
        if (!dir_entry.is_regular_file(ec)) continue;
        const std::string name = dir_entry.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".tmp") == 0) continue;
        const std::string stem = dir_entry.path().stem().string();
        const uint64_t size = static_cast<uint64_t>(dir_entry.file_size(ec));
        auto it = std::find_if(stems.begin(), stems.end(),
                               [&](const Stem& s2) { return s2.stem == stem; });
        if (it == stems.end()) {
            Stem entry;
            entry.stem = stem;
            entry.mtime = dir_entry.last_write_time(ec);
            entry.bytes = size;
            stems.push_back(entry);
        } else {
            it->bytes += size;
        }
    }
    std::sort(stems.begin(), stems.end(),
              [](const Stem& a, const Stem& b) { return a.mtime < b.mtime; });

    uint64_t total = 0;
    for (const Stem& s2 : stems) total += s2.bytes;
    const std::string keep_stem = keep.hex();
    std::vector<std::string> doomed;
    for (const Stem& s2 : stems) {
        if (total <= max_bytes_) break;
        if (s2.stem == keep_stem) continue;
        total -= std::min(total, s2.bytes);
        doomed.push_back(dir_ + "/" + s2.stem + kStateExt);
        doomed.push_back(dir_ + "/" + s2.stem + kMetaExt);
        doomed.push_back(dir_ + "/" + s2.stem + kLogitsExt);
    }
    for (const std::string& path : doomed) {
        fs::remove(path, ec);
    }
    if (!doomed.empty()) {
        log_info("snapshot evicted " + std::to_string(doomed.size() / 3) +
                 " snapshot(s) to stay under " + std::to_string(max_bytes_) + " bytes");
    }
}

SnapshotSaveResult KvSnapshotStore::save(const SnapshotKey& key,
                                         const std::vector<llama_token>& tokens,
                                         const SnapshotPayload& payload) {
    SnapshotSaveResult result;
    const auto begin = std::chrono::steady_clock::now();
    if (payload.state.empty()) {
        result.reason = "empty-state";
        return result;
    }
    if (payload.logits.empty()) {
        result.reason = "empty-logits";
        return result;
    }
    if (tokens.size() != key.n_tokens) {
        result.reason = "token-count-mismatch";
        return result;
    }
    const std::string kv_path = path_for(key, kStateExt);
    if (!write_file_atomic(kv_path, payload.state.data(), payload.state.size())) {
        result.reason = "write-failed";
        return result;
    }

    // Logits blob: magic + vocabulary size + f32 values. Written after the KV
    // so a crash leaves a stale manifest pointing at a valid KV but missing
    // logits, which the size check turns into a miss.
    std::vector<uint8_t> blob;
    blob.reserve(8 + payload.logits.size() * sizeof(float));
    blob.insert(blob.end(), kLogitsMagic, kLogitsMagic + 4);
    const uint32_t n_vocab = static_cast<uint32_t>(payload.logits.size());
    for (int i = 0; i < 4; ++i) {
        blob.push_back(static_cast<uint8_t>((n_vocab >> (8 * i)) & 0xffu));
    }
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(payload.logits.data());
    blob.insert(blob.end(), raw, raw + payload.logits.size() * sizeof(float));
    const std::string logits_path = path_for(key, kLogitsExt);
    if (!write_file_atomic(logits_path, blob.data(), blob.size())) {
        std::error_code ec;
        fs::remove(kv_path, ec);
        result.reason = "logits-write-failed";
        return result;
    }

    SnapshotManifest manifest;
    manifest.model_key = key.model_key;
    manifest.policy_key = key.policy_key;
    manifest.kv_type = key.kv_type;
    manifest.token_hash = key.token_hash;
    manifest.position_hash = key.position_hash;
    manifest.state_hash = hash_bytes(payload.state.data(), payload.state.size());
    manifest.bytes = payload.state.size();
    manifest.logits_hash = hash_bytes(blob.data(), blob.size());
    manifest.logits_bytes = blob.size();
    manifest.n_vocab = n_vocab;
    manifest.n_tokens = key.n_tokens;
    manifest.n_ctx = key.n_ctx;
    manifest.created = static_cast<uint64_t>(std::time(nullptr));
    if (!write_text_atomic(path_for(key, kMetaExt), manifest.serialize())) {
        std::error_code ec;
        fs::remove(kv_path, ec);
        fs::remove(logits_path, ec);
        result.reason = "meta-write-failed";
        return result;
    }
    evict_if_needed(key);
    result.ok = true;
    result.bytes = payload.state.size() + blob.size();
    result.save_ms = ms_since(begin);
    return result;
}

SnapshotLoadResult KvSnapshotStore::load(const SnapshotKey& key,
                                         const std::vector<llama_token>& expect,
                                         SnapshotPayload* payload) {
    SnapshotLoadResult result;
    const auto begin = std::chrono::steady_clock::now();
    if (!payload) {
        result.state = "miss:no-output";
        return result;
    }
    const std::string kv_path = path_for(key, kStateExt);
    const std::string meta_path = path_for(key, kMetaExt);
    const std::string logits_path = path_for(key, kLogitsExt);
    std::error_code ec;
    if (!fs::exists(kv_path, ec) || !fs::exists(meta_path, ec) || !fs::exists(logits_path, ec)) {
        result.state = "miss:absent";
        return result;
    }
    std::vector<uint8_t> meta_bytes;
    if (!read_file(meta_path, &meta_bytes)) {
        result.state = "miss:manifest";
        return result;
    }
    SnapshotManifest manifest;
    if (!SnapshotManifest::parse(std::string(meta_bytes.begin(), meta_bytes.end()), &manifest)) {
        result.state = "miss:manifest";
        return result;
    }
    std::vector<uint8_t> state;
    std::vector<uint8_t> blob;
    if (!read_file(kv_path, &state) || !read_file(logits_path, &blob)) {
        result.state = "miss:absent";
        return result;
    }
    if (!manifest.matches(key, state.size(), blob.size())) {
        result.state = "miss:manifest";
        return result;
    }
    if (hash_bytes(state.data(), state.size()) != manifest.state_hash ||
        hash_bytes(blob.data(), blob.size()) != manifest.logits_hash) {
        result.state = "miss:corrupt";
        return result;
    }
    if (expect.size() != key.n_tokens || hash_tokens(expect) != key.token_hash) {
        result.state = "miss:tokens";
        return result;
    }
    if (blob.size() < 8 || std::memcmp(blob.data(), kLogitsMagic, 4) != 0) {
        result.state = "miss:corrupt";
        return result;
    }
    uint32_t n_vocab = 0;
    for (int i = 0; i < 4; ++i) {
        n_vocab |= static_cast<uint32_t>(blob[4 + i]) << (8 * i);
    }
    if (n_vocab != manifest.n_vocab || blob.size() != 8 + static_cast<size_t>(n_vocab) * sizeof(float)) {
        result.state = "miss:corrupt";
        return result;
    }
    payload->state = std::move(state);
    payload->logits.resize(n_vocab);
    std::memcpy(payload->logits.data(), blob.data() + 8,
                static_cast<size_t>(n_vocab) * sizeof(float));
    payload->n_vocab = n_vocab;
    result.hit = true;
    result.bytes = payload->state.size() + blob.size();
    result.load_ms = ms_since(begin);
    result.state = "warm";
    return result;
}

} // namespace edge_prefill
