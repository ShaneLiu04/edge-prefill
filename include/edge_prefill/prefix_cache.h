#pragma once

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <llama.h>

#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct CacheHit {
    uint32_t prefix_len = 0;
    std::vector<uint8_t> state;
    uint64_t age_ms = 0;
};

struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t reused_tokens = 0;
    uint64_t bytes = 0;
    uint64_t entries = 0;
    uint64_t evictions = 0;
};

class PrefixCache {
public:
    static Status create(uint64_t max_bytes, std::string model_key,
                         std::unique_ptr<PrefixCache>* out);

    std::optional<CacheHit> find(const std::vector<llama_token>& tokens,
                                 const KvSignature& sig);
    Status insert(const std::vector<llama_token>& tokens,
                  const KvSignature& sig,
                  std::vector<uint8_t> state);
    void clear();
    CacheStats stats() const;

private:
    PrefixCache(uint64_t max_bytes, std::string model_key);

    struct Entry {
        uint64_t id = 0;
        std::vector<llama_token> tokens;
        KvSignature sig;
        std::vector<uint8_t> state;
        uint64_t last_used = 0;
    };

    void evict_locked();

    mutable std::mutex mutex_;
    uint64_t max_bytes_;
    std::string model_key_;
    uint64_t used_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t next_id_ = 1;
    std::list<Entry> entries_;
    std::unordered_map<uint64_t, std::list<Entry>::iterator> by_id_;
    CacheStats stats_;
};

} // namespace edge_prefill
