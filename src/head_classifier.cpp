#include "edge_prefill/head_classifier.h"

#include <algorithm>
#include <random>
#include <vector>

#include <llama.h>

namespace edge_prefill {
namespace {

Status invalid(const std::string& message) {
    return Status::error(ErrorCode::InvalidArgument, message);
}

// Deterministic probe sequence: rejection-sample non-control, non-EOG
// tokens from the vocabulary with a fixed-seed generator so the same model
// and parameters always produce the same sequence (and thus the same
// profile, byte for byte).
Status make_probe_tokens(const LlamaBackend& backend, const HeadProbeParams& params,
                         uint32_t n, std::vector<llama_token>* out) {
    const llama_vocab* vocab = backend.vocab();
    const int32_t n_vocab = backend.n_vocab();
    if (!vocab || n_vocab <= 0) return invalid("vocabulary is not available");
    std::mt19937_64 rng(params.seed);
    out->clear();
    out->reserve(n);
    // Guard against a pathological vocabulary where every token is rejected:
    // fail loudly instead of looping forever.
    uint32_t consecutive_rejections = 0;
    while (out->size() < n) {
        const llama_token tok =
                static_cast<llama_token>(rng() % static_cast<uint64_t>(n_vocab));
        if (llama_vocab_is_control(vocab, tok) || llama_vocab_is_eog(vocab, tok)) {
            if (++consecutive_rejections > 1000000u) {
                return invalid("vocabulary has no usable probe tokens");
            }
            continue;
        }
        consecutive_rejections = 0;
        out->push_back(tok);
    }
    return Status::success();
}

// Union of the sink rows (0..sink-1) and the recent rows (n-recent..n-1),
// ascending and deduplicated: the ranked pool the probed head gathers from.
// Overlap happens for short probes and must not produce duplicate rows.
std::vector<int32_t> probe_pool_rows(const HeadProbeParams& params, uint32_t n) {
    std::vector<int32_t> rows;
    for (uint32_t i = 0; i < params.sink && i < n; ++i) {
        rows.push_back(static_cast<int32_t>(i));
    }
    const uint32_t recent_begin = n > params.recent ? n - params.recent : 0;
    for (uint32_t i = recent_begin; i < n; ++i) {
        rows.push_back(static_cast<int32_t>(i));
    }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

// Positions whose next token is a repeat of the previous copy: inside copy
// c >= 2, position p = (c-1)*m + i predicts S[p+1] = S[(c-2)*m + i + 1].
std::vector<uint32_t> echo_positions(const HeadProbeParams& params) {
    std::vector<uint32_t> positions;
    const uint32_t m = params.block_len;
    for (uint32_t c = 2; c <= params.repeats; ++c) {
        for (uint32_t i = 0; i + 1 < m; ++i) {
            positions.push_back((c - 1) * m + i);
        }
    }
    return positions;
}

// The last token of copy c predicts the first token of copy c+1, which is
// S[0]: the classic induction probe.
std::vector<uint32_t> induction_positions(const HeadProbeParams& params) {
    std::vector<uint32_t> positions;
    for (uint32_t c = 1; c < params.repeats; ++c) {
        positions.push_back(c * params.block_len - 1);
    }
    return positions;
}

double flip_rate(const std::vector<int32_t>& probe, const std::vector<int32_t>& baseline,
                 const std::vector<uint32_t>& positions) {
    if (positions.empty()) return 0.0;
    uint32_t flips = 0;
    for (uint32_t p : positions) {
        if (probe[p] != baseline[p]) ++flips;
    }
    return static_cast<double>(flips) / static_cast<double>(positions.size());
}

} // namespace

HeadClassifier::HeadClassifier(double retrieval_threshold)
    : retrieval_threshold_(retrieval_threshold) {}

Status HeadClassifier::classify(LlamaBackend* backend, const HeadProbeParams& params,
                                const std::string& model_sha, HeadProfile* out) {
    if (!backend) return invalid("backend is null");
    if (!out) return invalid("out is null");
    if (params.block_len == 0) return invalid("probe block_len must be >= 1");
    if (params.repeats < 2) {
        return invalid("probe repeats must be >= 2: echo/induction need at least two copies");
    }
    if (params.probe_k == 0) return invalid("probe probe_k must be >= 1");
    if (params.sink + params.recent == 0) {
        return invalid("probe sink+recent must be >= 1: the ablated head needs a row pool");
    }

    const int32_t n_layer = backend->n_layer();
    const int32_t n_head = backend->n_head();
    if (n_layer <= 0 || n_head <= 0) {
        return invalid("model geometry unavailable (n_layer/n_head)");
    }

    const uint64_t probe_len =
            static_cast<uint64_t>(params.block_len) * static_cast<uint64_t>(params.repeats);
    const uint32_t n_ctx = backend->n_ctx();
    if (probe_len > n_ctx) {
        return invalid("probe length (" + std::to_string(probe_len) + ") exceeds n_ctx (" +
                       std::to_string(n_ctx) + "); lower --probe-repeats or --probe-block-len");
    }
    const uint32_t n = static_cast<uint32_t>(probe_len);

    std::vector<llama_token> tokens;
    auto st = make_probe_tokens(*backend, params, n, &tokens);
    if (!st.ok()) return st;

    const std::vector<uint32_t> echo = echo_positions(params);
    const std::vector<uint32_t> induction = induction_positions(params);

    // Dense baseline with the per-head path fully disabled.
    st = backend->set_attn_ranked_rows({});
    if (!st.ok()) return st;
    st = backend->set_attn_head_top_k({});
    if (!st.ok()) return st;
    st = backend->reset_sequence();
    if (!st.ok()) return st;
    std::vector<int32_t> baseline;
    st = backend->probe_argmax(tokens, &baseline);
    if (!st.ok()) return st;

    const std::vector<int32_t> pool = probe_pool_rows(params, n);
    std::vector<int32_t> matrix(static_cast<size_t>(n_layer) * static_cast<size_t>(n_head), -1);
    st = backend->set_attn_ranked_rows(pool);
    if (!st.ok()) return st;

    HeadProfile profile;
    profile.model_sha = model_sha;
    profile.n_layer = n_layer;
    profile.n_head = n_head;
    profile.probe = params;
    profile.heads.reserve(matrix.size());

    for (int32_t layer = 0; layer < n_layer; ++layer) {
        for (int32_t head = 0; head < n_head; ++head) {
            const size_t idx = static_cast<size_t>(layer) * static_cast<size_t>(n_head) +
                               static_cast<size_t>(head);
            matrix[idx] = static_cast<int32_t>(params.probe_k);
            st = backend->set_attn_head_top_k(matrix);
            if (!st.ok()) {
                backend->set_attn_head_top_k({});
                backend->set_attn_ranked_rows({});
                return st;
            }
            st = backend->reset_sequence();
            if (!st.ok()) {
                backend->set_attn_head_top_k({});
                backend->set_attn_ranked_rows({});
                return st;
            }
            std::vector<int32_t> probe_top1;
            st = backend->probe_argmax(tokens, &probe_top1);
            matrix[idx] = -1;
            if (!st.ok()) {
                backend->set_attn_head_top_k({});
                backend->set_attn_ranked_rows({});
                return st;
            }

            HeadProfileEntry entry;
            entry.layer = layer;
            entry.head = head;
            entry.echo_score = flip_rate(probe_top1, baseline, echo);
            entry.induction_score = flip_rate(probe_top1, baseline, induction);
            entry.retrieval =
                    std::max(entry.echo_score, entry.induction_score) > retrieval_threshold_;
            profile.heads.push_back(entry);
        }
    }

    // Restore dense semantics so the backend is left untouched for callers
    // that reuse it after classification (probe passes, generation, ...).
    st = backend->set_attn_head_top_k({});
    if (!st.ok()) return st;
    st = backend->set_attn_ranked_rows({});
    if (!st.ok()) return st;
    st = backend->reset_sequence();
    if (!st.ok()) return st;

    *out = std::move(profile);
    return Status::success();
}

} // namespace edge_prefill
