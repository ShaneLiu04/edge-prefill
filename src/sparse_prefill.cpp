#include "edge_prefill/sparse_prefill.h"

#include <chrono>

#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

KvSignature signature_for(const LlamaBackend& backend) {
    KvSignature sig;
    sig.model_key = backend.fingerprint().to_key();
    sig.n_ctx = backend.n_ctx();
    sig.type_k = backend.context_options().type_k;
    sig.type_v = backend.context_options().type_v;
    return sig;
}

class PrefillModeGuard {
public:
    explicit PrefillModeGuard(LlamaBackend* backend) : backend_(backend) {
        if (backend_) backend_->set_prefill_mode(true);
    }
    ~PrefillModeGuard() {
        if (backend_) backend_->set_prefill_mode(false);
    }

private:
    LlamaBackend* backend_;
};

} // namespace

SparsePrefiller::SparsePrefiller(LlamaBackend* backend, PrefixCache* cache)
    : backend_(backend), cache_(cache) {}

Status SparsePrefiller::prefill(const std::vector<llama_token>& tokens,
                                const Selection& selection,
                                const Options& opts,
                                PrefillStats* stats) {
    if (!backend_ || !stats) {
        return Status::error(ErrorCode::InvalidArgument, "backend/stats is null");
    }
    if (tokens.empty() || selection.token_indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty token selection");
    }
    for (size_t i = 0; i < selection.token_indices.size(); ++i) {
        const uint32_t idx = selection.token_indices[i];
        if (idx >= tokens.size()) {
            return Status::error(ErrorCode::InvalidArgument, "selection index out of range");
        }
        if (i > 0 && idx <= selection.token_indices[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument, "selection indices not increasing");
        }
    }

    *stats = PrefillStats{};
    stats->prompt_tokens = static_cast<uint32_t>(tokens.size());
    stats->selected_tokens = static_cast<uint32_t>(selection.token_indices.size());
    stats->collect_chunk_stats = opts.collect_chunk_stats;

    const auto begin = std::chrono::steady_clock::now();
    uint32_t prefix_len = 0;
    if (cache_ && opts.enable_prefix_cache && selection.dense) {
        const KvSignature sig = signature_for(*backend_);
        const auto hit = cache_->find(tokens, sig);
        if (hit && hit->prefix_len > 0) {
            auto st = backend_->reset_sequence();
            if (!st.ok()) return st;
            st = backend_->load_state(hit->state);
            if (!st.ok()) {
                log_warn("prefix state restore failed; recomputing dense prefix");
                backend_->reset_sequence();
            } else {
                prefix_len = hit->prefix_len;
            }
        }
    }

    if (prefix_len == 0) {
        auto st = backend_->reset_sequence();
        if (!st.ok()) return st;
    }

    if (opts.enable_ffn_skip && opts.ffn_skip_mask != 0) {
        auto st = backend_->set_ffn_skip_mask(opts.ffn_skip_mask);
        if (!st.ok()) {
            log_warn(std::string("FFN skip disabled: ") + st.message);
        }
    } else {
        backend_->set_ffn_skip_mask(0);
    }

    std::vector<llama_token> active_tokens;
    std::vector<llama_pos> active_positions;
    active_tokens.reserve(selection.token_indices.size());
    active_positions.reserve(selection.token_indices.size());
    for (uint32_t idx : selection.token_indices) {
        if (idx < prefix_len) continue;
        active_tokens.push_back(tokens[idx]);
        active_positions.push_back(static_cast<llama_pos>(idx));
    }

    if (!active_tokens.empty()) {
        PrefillModeGuard guard(backend_);
        auto st = backend_->prefill(active_tokens, active_positions,
                                    opts.enable_chunking ? opts.chunk_size : 0, stats);
        if (!st.ok()) return st;
        stats->prompt_tokens = static_cast<uint32_t>(tokens.size());
        stats->selected_tokens = static_cast<uint32_t>(selection.token_indices.size());
    } else {
        stats->chunks = 0;
        stats->chunk_stats.clear();
    }

    const auto end = std::chrono::steady_clock::now();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    stats->total_us = elapsed + (prefix_len > 0 ? 0 : 0);
    stats->ttft_us = stats->total_us;
    stats->kv_pos_min = 0;
    stats->kv_pos_max = static_cast<int32_t>(tokens.size()) - 1;
    stats->kv_bytes = backend_->kv_bytes();

    if (cache_ && opts.enable_prefix_cache && selection.dense) {
        std::vector<uint8_t> state;
        auto st = backend_->save_state(&state);
        if (st.ok()) {
            const KvSignature sig = signature_for(*backend_);
            auto ins = cache_->insert(tokens, sig, std::move(state));
            if (!ins.ok()) log_warn("prefix cache insert failed: " + ins.message);
        }
    }
    return Status::success();
}

Status SparsePrefiller::decode_next(llama_token token, llama_pos pos,
                                    std::vector<float>* logits) {
    if (!backend_) return Status::error(ErrorCode::InvalidArgument, "backend is null");
    return backend_->decode(token, pos, logits);
}

Status SparsePrefiller::save_prefix_state(const std::vector<llama_token>& tokens,
                                          const KvSignature& sig) {
    if (!backend_ || !cache_) {
        return Status::error(ErrorCode::InvalidArgument, "backend/cache is null");
    }
    if (tokens.empty()) return Status::error(ErrorCode::InvalidArgument, "empty tokens");
    std::vector<uint8_t> state;
    auto st = backend_->save_state(&state);
    if (!st.ok()) return st;
    return cache_->insert(tokens, sig, std::move(state));
}

} // namespace edge_prefill
