#include "edge_prefill/sparse_prefill.h"

#include <algorithm>
#include <chrono>

#include "edge_prefill/attention_sparsity.h"
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

    // Validate gather indices before touching any model state so an invalid
    // request never leaves a partially written sequence behind.
    const bool attn_requested = opts.enable_attn_sparse && !selection.dense &&
                                !selection.attention_indices.empty();
    if (attn_requested) {
        auto vst = AttentionSparsityController::validate_indices(selection.attention_indices,
                                                                 selection);
        if (!vst.ok()) return vst;
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

    // Probe the gather patch once: a missing llama.cpp patch must degrade to
    // block-level sparsity instead of failing the request.
    bool attn_active = false;
    if (attn_requested) {
        stats->attn_indices = static_cast<uint32_t>(selection.attention_indices.size());
        auto probe = backend_->set_attn_top_k(1);
        if (!probe.ok()) {
            log_warn("attention top-k unavailable (" + probe.message +
                     "); falling back to block-level sparsity");
            stats->attn_fallback = true;
        } else {
            attn_active = true;
            stats->attn_top_k = static_cast<int32_t>(selection.attention_indices.size());
        }
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
        Status st = Status::success();
        if (attn_active) {
            st = attn_prefill(active_tokens, active_positions, selection.attention_indices,
                              opts, stats);
        } else {
            st = backend_->prefill(active_tokens, active_positions,
                                   opts.enable_chunking ? opts.chunk_size : 0, stats);
        }
        // Always restore the dense attention path so decode and subsequent
        // prefills never inherit stale gather settings.
        backend_->set_attn_top_k(0);
        backend_->set_attn_indices({});
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

Status SparsePrefiller::attn_prefill(const std::vector<llama_token>& tokens,
                                     const std::vector<llama_pos>& positions,
                                     const std::vector<int32_t>& attn_indices,
                                     const Options& opts,
                                     PrefillStats* stats) {
    // The llama.cpp gather patch indexes KV cells in decode order, not by
    // absolute position: with sparse selection the r-th cell holds the r-th
    // selected token, so position != row. Map every gather position to its
    // row up front; rows inherit the ascending order of the positions.
    std::vector<int32_t> index_rows;
    index_rows.reserve(attn_indices.size());
    for (int32_t pos : attn_indices) {
        auto it = std::lower_bound(positions.begin(), positions.end(), pos);
        if (it == positions.end() || *it != pos) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "attention index has no selected position");
        }
        index_rows.push_back(static_cast<int32_t>(it - positions.begin()));
    }

    // Chunked prefill writes KV cells progressively: when a chunk covering
    // rows [range.begin, range.end) executes, only rows < range.end exist.
    // Gathering beyond that reads unwritten cells, so each chunk gathers the
    // prefix of mapped rows already in the cache (empty prefix -> gather
    // disabled for the chunk, dense attention over the written rows, which
    // is causally safe).
    std::vector<Range> ranges;
    auto st = ChunkScheduler::plan(static_cast<uint32_t>(tokens.size()),
                                   opts.enable_chunking ? opts.chunk_size : 0,
                                   llama_n_ubatch(backend_->context()), &ranges);
    if (!st.ok()) return st;

    stats->chunks = static_cast<uint32_t>(ranges.size());
    stats->chunk_stats.clear();
    std::vector<int32_t> allowed;
    for (const auto& range : ranges) {
        std::vector<llama_token> chunk_tokens(tokens.begin() + range.begin,
                                              tokens.begin() + range.end);
        std::vector<llama_pos> chunk_positions(positions.begin() + range.begin,
                                                positions.begin() + range.end);
        const auto row_end =
            std::lower_bound(index_rows.begin(), index_rows.end(),
                             static_cast<int32_t>(range.end));
        allowed.assign(index_rows.begin(), row_end);
        st = backend_->set_attn_indices(allowed);
        if (!st.ok()) return st;
        st = backend_->set_attn_top_k(static_cast<int32_t>(allowed.size()));
        if (!st.ok()) return st;

        PrefillStats chunk_stats;
        chunk_stats.collect_chunk_stats = stats->collect_chunk_stats;
        st = backend_->prefill(chunk_tokens, chunk_positions, 0, &chunk_stats);
        if (!st.ok()) return st;
        for (ChunkStats cs : chunk_stats.chunk_stats) {
            cs.index = static_cast<uint32_t>(stats->chunk_stats.size());
            cs.attn_rows = static_cast<uint32_t>(allowed.size());
            stats->chunk_stats.push_back(cs);
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
