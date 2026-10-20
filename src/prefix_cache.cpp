#include "edge_prefill/prefix_cache.h"

#include <algorithm>
#include <optional>

namespace edge_prefill {

PrefixCache::PrefixCache(uint64_t max_bytes, std::string model_key)
    : max_bytes_(max_bytes), model_key_(std::move(model_key)) {}

Status PrefixCache::create(uint64_t max_bytes, std::string model_key,
                           std::unique_ptr<PrefixCache>* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null cache output");
    if (model_key.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty model key");
    }
    *out = std::unique_ptr<PrefixCache>(new PrefixCache(max_bytes, std::move(model_key)));
    return Status::success();
}

std::optional<CacheHit> PrefixCache::find(const std::vector<llama_token>& tokens,
                                          const KvSignature& sig) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_bytes_ == 0 || sig.model_key != model_key_ || tokens.empty()) {
        ++stats_.misses;
        return std::nullopt;
    }

    Entry* best = nullptr;
    size_t best_prefix = 0;
    for (auto& entry : entries_) {
        if (!(entry.sig == sig)) continue;
        const size_t limit = std::min(tokens.size(), entry.tokens.size());
        size_t prefix = 0;
        while (prefix < limit && tokens[prefix] == entry.tokens[prefix]) ++prefix;
        if (prefix > best_prefix) {
            best_prefix = prefix;
            best = &entry;
        }
    }

    if (!best || best_prefix == 0) {
        ++stats_.misses;
        return std::nullopt;
    }

    ++clock_;
    const uint64_t age = clock_ - best->last_used;
    best->last_used = clock_;
    ++stats_.hits;
    stats_.reused_tokens += best_prefix;
    CacheHit hit;
    hit.prefix_len = static_cast<uint32_t>(best_prefix);
    hit.state = best->state;
    hit.age_ms = age;
    return hit;
}

Status PrefixCache::insert(const std::vector<llama_token>& tokens,
                           const KvSignature& sig,
                           std::vector<uint8_t> state) {
    if (tokens.empty()) return Status::error(ErrorCode::InvalidArgument, "empty tokens");
    if (state.empty()) return Status::error(ErrorCode::InvalidArgument, "empty state");
    std::lock_guard<std::mutex> lock(mutex_);
    if (max_bytes_ == 0) return Status::success();
    if (sig.model_key != model_key_) {
        return Status::error(ErrorCode::Cache, "model key mismatch");
    }
    if (state.size() > max_bytes_) {
        return Status::error(ErrorCode::Cache, "state larger than cache capacity");
    }

    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->tokens == tokens && it->sig == sig) {
            used_bytes_ -= it->state.size();
            it->state = std::move(state);
            used_bytes_ += it->state.size();
            ++clock_;
            it->last_used = clock_;
            return Status::success();
        }
    }

    ++clock_;
    Entry entry;
    entry.id = next_id_++;
    entry.tokens = tokens;
    entry.sig = sig;
    entry.state = std::move(state);
    entry.last_used = clock_;
    used_bytes_ += entry.state.size();
    entries_.push_back(std::move(entry));
    by_id_[entries_.back().id] = std::prev(entries_.end());

    while (used_bytes_ > max_bytes_ && entries_.size() > 1) {
        evict_locked();
    }
    return Status::success();
}

void PrefixCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    by_id_.clear();
    used_bytes_ = 0;
    stats_ = {};
}

CacheStats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats out = stats_;
    out.bytes = used_bytes_;
    out.entries = static_cast<uint64_t>(entries_.size());
    return out;
}

void PrefixCache::evict_locked() {
    if (entries_.empty()) return;
    auto victim = entries_.begin();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->last_used < victim->last_used) victim = it;
    }
    used_bytes_ -= victim->state.size();
    by_id_.erase(victim->id);
    entries_.erase(victim);
    ++stats_.evictions;
}

} // namespace edge_prefill
