#include "edge_prefill/selector.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "edge_prefill/json_writer.h"
#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

std::string normalize_piece(const std::string& piece) {
    std::string out;
    out.reserve(piece.size());
    for (unsigned char c : piece) {
        if (c == 0xE2) continue; // UTF-8 leading byte of ▁ (we handle bytes below)
        if (c == 0x96 || c == 0x81) continue;
        if (c == 0xC4 || c == 0xA0) continue; // Ġ
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c >= 0x80) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back(' ');
        }
    }
    return out;
}

std::vector<std::string> terms_from_text(const std::string& text) {
    std::vector<std::string> terms;
    std::string current;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            current.push_back(static_cast<char>(std::tolower(c)));
        } else if (c >= 0x80) {
            if (!current.empty()) {
                terms.push_back(current);
                current.clear();
            }
            terms.emplace_back(1, static_cast<char>(c));
        } else {
            if (!current.empty()) {
                terms.push_back(current);
                current.clear();
            }
        }
    }
    if (!current.empty()) terms.push_back(current);
    return terms;
}

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    return hash;
}

using Vec64 = std::array<double, 64>;

Vec64 hashed_embedding(const std::string& text) {
    Vec64 v{};
    const uint64_t h = fnv1a(text);
    for (size_t i = 0; i < v.size(); ++i) {
        const uint64_t x = (h >> (i % 56)) ^ (0x9e3779b97f4a7c15ull * (i + 1));
        v[i] = static_cast<double>(x % 2001) / 1000.0 - 1.0;
    }
    double norm = 0.0;
    for (double x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (double& x : v) x /= norm;
    }
    return v;
}

double cosine(const Vec64& a, const Vec64& b) {
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return std::max(-1.0, std::min(1.0, dot));
}

void minmax(std::vector<double>* values) {
    if (values->empty()) return;
    const auto [lo, hi] = std::minmax_element(values->begin(), values->end());
    const double range = *hi - *lo;
    if (range < 1e-12) {
        for (double& v : *values) v = 0.5;
        return;
    }
    for (double& v : *values) v = (v - *lo) / range;
}

std::string reason_for(bool dense, double budget) {
    if (dense) return budget >= 0.95 ? "budget>=0.95" : "short-prompt";
    return "selected";
}

} // namespace

std::string Selection::to_json() const {
    std::string out;
    JsonWriter json(&out);
    json.begin_object();
    json.key("dense");
    json.value(dense);
    json.key("reason");
    json.value(reason);
    json.key("total_tokens");
    json.value(static_cast<uint64_t>(total_tokens));
    json.key("selected_tokens");
    json.value(static_cast<uint64_t>(selected_tokens));
    json.key("block_indices");
    json.begin_array();
    for (auto b : block_indices) json.value(static_cast<uint64_t>(b));
    json.end_array();
    json.key("attention_top_k");
    json.value(static_cast<uint64_t>(attention_top_k));
    json.key("attention_indices");
    json.begin_array();
    for (auto i : attention_indices) json.value(static_cast<uint64_t>(i));
    json.end_array();
    json.key("mandatory_blocks");
    json.begin_array();
    for (auto b : mandatory_blocks) json.value(static_cast<uint64_t>(b));
    json.end_array();
    json.key("signals");
    json.begin_array();
    for (const auto& s : signals) {
        json.begin_object();
        json.key("lexical");
        json.value(s.lexical);
        json.key("diversity");
        json.value(s.diversity);
        json.key("position");
        json.value(s.position);
        json.key("draft");
        json.value(s.draft);
        json.key("fused");
        json.value(s.fused);
        json.end_object();
    }
    json.end_array();
    json.end_object();
    return out;
}

Status BlockSelector::select(const std::vector<llama_token>& tokens,
                             const std::string& query_text,
                             const std::vector<std::string>& token_pieces,
                             const SelectionOptions& opts,
                             DraftScorer* draft,
                             Selection* out) const {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null selection output");
    if (tokens.empty()) return Status::error(ErrorCode::InvalidArgument, "empty token sequence");
    if (token_pieces.size() != tokens.size()) {
        return Status::error(ErrorCode::InvalidArgument, "token pieces size mismatch");
    }
    if (opts.budget <= 0.0 || opts.budget > 1.0) {
        return Status::error(ErrorCode::InvalidArgument, "budget must be in (0,1]");
    }
    if (opts.block_size < 8 || opts.block_size > 256) {
        return Status::error(ErrorCode::InvalidArgument, "block_size must be in [8,256]");
    }
    if (opts.min_dense_tokens < opts.block_size) {
        return Status::error(ErrorCode::InvalidArgument, "min_dense_tokens < block_size");
    }
    for (unsigned char c : query_text) {
        (void)c;
    }

    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());
    const uint32_t block_size = opts.block_size;
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;

    out->total_tokens = n_tokens;
    out->signals.assign(n_blocks, {});
    out->mandatory_blocks.clear();
    out->block_indices.clear();
    out->token_indices.clear();
    // Reset per call: a reused Selection must not leak the previous
    // request's score source into the audit trail.
    out->score_source = "lexical";

    if (n_tokens < opts.min_dense_tokens || opts.budget >= 0.95) {
        out->dense = true;
        out->reason = reason_for(true, opts.budget);
        out->selected_tokens = n_tokens;
        out->token_indices.resize(n_tokens);
        std::iota(out->token_indices.begin(), out->token_indices.end(), 0);
        out->block_indices.resize(n_blocks);
        std::iota(out->block_indices.begin(), out->block_indices.end(), 0);
        return Status::success();
    }
    out->dense = false;
    out->reason = "selected";

    std::vector<std::vector<std::string>> block_terms(n_blocks);
    std::vector<std::unordered_set<std::string>> block_terms_unique(n_blocks);
    std::vector<std::vector<std::string>> token_terms(n_tokens);
    std::vector<Vec64> token_embeddings(tokens.size());
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const std::string norm = normalize_piece(token_pieces[i]);
        auto terms = terms_from_text(norm);
        for (const auto& term : terms) {
            if (term.size() >= 2 || static_cast<unsigned char>(term[0]) >= 0x80) {
                block_terms[i / block_size].push_back(term);
                block_terms_unique[i / block_size].insert(term);
                token_terms[i].push_back(term);
            }
        }
        token_embeddings[i] = hashed_embedding(norm);
    }

    std::unordered_set<std::string> query_terms;
    for (const auto& term : terms_from_text(query_text)) {
        if (term.size() >= 2 || (!term.empty() && static_cast<unsigned char>(term[0]) >= 0x80)) {
            query_terms.insert(term);
        }
    }

    std::unordered_map<std::string, uint32_t> df;
    if (!query_terms.empty()) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            for (const auto& term : block_terms_unique[b]) {
                if (query_terms.count(term)) ++df[term];
            }
        }
    }

    std::vector<double> lexical(n_blocks, 0.0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        double score = 0.0;
        for (const auto& term : block_terms_unique[b]) {
            if (!query_terms.count(term)) continue;
            const double idf = std::log((static_cast<double>(n_blocks) + 1.0) /
                                        (static_cast<double>(df[term]) + 1.0)) + 1.0;
            score += idf;
        }
        const uint32_t size = std::min(block_size, n_tokens - b * block_size);
        lexical[b] = score / std::sqrt(static_cast<double>(std::max<uint32_t>(1, size)));
    }

    // AR004 FR-3: per-position lexical relevance (idf of the token's own
    // query-term hits, deduped within the token). Retained for the
    // query-chunk gating planner; superseded by draft scores below when a
    // draft scorer is active.
    std::vector<double> per_pos(n_tokens, 0.0);
    {
        std::vector<std::string> uniq;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            uniq = token_terms[i];
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            double score = 0.0;
            for (const auto& term : uniq) {
                if (!query_terms.count(term)) continue;
                score += std::log((static_cast<double>(n_blocks) + 1.0) /
                                  (static_cast<double>(df[term]) + 1.0)) + 1.0;
            }
            per_pos[i] = score;
        }
    }

    Vec64 global{};
    for (const auto& e : token_embeddings) {
        for (size_t k = 0; k < global.size(); ++k) global[k] += e[k];
    }
    double global_norm = 0.0;
    for (double x : global) global_norm += x * x;
    global_norm = std::sqrt(global_norm);
    if (global_norm > 1e-12) {
        for (double& x : global) x /= global_norm;
    }

    std::vector<double> diversity(n_blocks, 0.0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        const uint32_t begin = b * block_size;
        const uint32_t end = std::min(n_tokens, begin + block_size);
        Vec64 centroid{};
        for (uint32_t i = begin; i < end; ++i) {
            for (size_t k = 0; k < centroid.size(); ++k) centroid[k] += token_embeddings[i][k];
        }
        double norm = 0.0;
        for (double x : centroid) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-12) {
            for (double& x : centroid) x /= norm;
        }
        double intra = 0.0;
        for (uint32_t i = begin; i < end; ++i) intra += cosine(token_embeddings[i], centroid);
        intra /= std::max<uint32_t>(1, end - begin);
        diversity[b] = 0.5 * (1.0 - cosine(centroid, global)) + 0.5 * (1.0 - intra);
    }

    std::vector<double> position(n_blocks, 0.0);
    const double lambda = std::max(1.0, static_cast<double>(n_blocks) / 4.0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (b < opts.sink_blocks || b + opts.recent_blocks >= n_blocks) {
            position[b] = 1.0;
        } else {
            const double d_sink = static_cast<double>(b);
            const double d_recent = static_cast<double>(n_blocks - 1 - b);
            position[b] = std::exp(-std::min(d_sink, d_recent) / lambda);
        }
    }

    std::vector<double> draft_scores(n_blocks, 0.0);
    std::vector<double> draft_per_token;
    bool draft_enabled = false;
    if (draft) {
        auto st = draft->score(tokens, query_text, &draft_per_token);
        // A failed draft pass (missing model, vocab mismatch, oversized
        // prompt) must degrade to lexical-only selection rather than fail
        // the request; never fabricate semantic scores.
        if (st.ok() && draft_per_token.size() == tokens.size()) {
            draft_enabled = true;
            out->score_source = draft->source();
            for (uint32_t b = 0; b < n_blocks; ++b) {
                const uint32_t begin = b * block_size;
                const uint32_t end = std::min(n_tokens, begin + block_size);
                double sum = 0.0;
                for (uint32_t i = begin; i < end; ++i) sum += draft_per_token[i];
                draft_scores[b] = sum / std::max<uint32_t>(1, end - begin);
            }
        } else if (!st.ok()) {
            out->reason = "draft-disabled";
            log_info("draft scorer disabled: " + st.message);
        } else {
            out->reason = "draft-disabled";
            log_info("draft scorer disabled: score size mismatch");
        }
    }

    minmax(&lexical);
    minmax(&diversity);
    minmax(&position);
    if (draft_enabled) minmax(&draft_scores);

    // FR-3 score source: semantic draft scores win over the lexical
    // per-position fallback when the draft scorer produced a full-length
    // vector.
    if (draft_enabled && draft_per_token.size() == n_tokens) {
        per_pos = draft_per_token;
    }
    out->per_pos_scores = std::move(per_pos);

    SelectorWeights weights = opts.weights;
    if (!draft_enabled) {
        const double used = weights.lexical + weights.diversity + weights.position;
        if (used > 1e-12) {
            weights.lexical /= used;
            weights.diversity /= used;
            weights.position /= used;
        }
        weights.draft = 0.0;
    } else {
        const double total = weights.lexical + weights.diversity + weights.position + weights.draft;
        if (total > 1e-12) {
            weights.lexical /= total;
            weights.diversity /= total;
            weights.position /= total;
            weights.draft /= total;
        }
    }

    std::vector<double> fused(n_blocks, 0.0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        out->signals[b].lexical = lexical[b];
        out->signals[b].diversity = diversity[b];
        out->signals[b].position = position[b];
        out->signals[b].draft = draft_enabled ? draft_scores[b] : 0.0;
        fused[b] = weights.lexical * lexical[b] + weights.diversity * diversity[b] +
                   weights.position * position[b] + weights.draft * draft_scores[b];
        out->signals[b].fused = fused[b];
    }

    std::set<uint32_t> mandatory;
    mandatory.insert(0);
    mandatory.insert(n_blocks - 1);
    if (!query_terms.empty()) {
        // The prompt suffix contains the question itself, so its lexical
        // score is a self-match. Restrict the best-lexical anchor to context
        // blocks that are not already guaranteed by sink/recent coverage.
        uint32_t best_lex = 0;
        bool best_lex_found = false;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            if (b < opts.sink_blocks || b + opts.recent_blocks >= n_blocks) continue;
            if (!best_lex_found || lexical[b] > lexical[best_lex]) {
                best_lex = b;
                best_lex_found = true;
            }
        }
        if (best_lex_found) {
            mandatory.insert(best_lex);
            // Facts often straddle a block boundary; keep the neighbors of the
            // strongest lexical match so the answer span is never split away.
            for (int32_t delta = -2; delta <= 2; ++delta) {
                const int32_t b = static_cast<int32_t>(best_lex) + delta;
                if (b >= 0 && b < static_cast<int32_t>(n_blocks)) {
                    mandatory.insert(static_cast<uint32_t>(b));
                }
            }
        }
    }
    const uint32_t stride = std::max<uint32_t>(1, opts.coverage_stride);
    for (uint32_t start = 0; start < n_blocks; start += stride) {
        const uint32_t end = std::min(n_blocks, start + stride);
        uint32_t best = start;
        for (uint32_t b = start + 1; b < end; ++b) {
            if (fused[b] > fused[best]) best = b;
        }
        mandatory.insert(best);
    }

    const uint32_t target = static_cast<uint32_t>(
        std::ceil(static_cast<double>(n_tokens) * opts.budget));
    std::set<uint32_t> selected_blocks;
    uint32_t selected_tokens = 0;
    for (uint32_t b : mandatory) {
        selected_blocks.insert(b);
        selected_tokens += std::min(block_size, n_tokens - b * block_size);
    }

    std::vector<uint32_t> order(n_blocks);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (std::abs(fused[a] - fused[b]) > 1e-12) return fused[a] > fused[b];
        return a < b;
    });
    for (uint32_t b : order) {
        if (selected_tokens >= target) break;
        if (selected_blocks.count(b)) continue;
        selected_blocks.insert(b);
        selected_tokens += std::min(block_size, n_tokens - b * block_size);
    }

    const uint32_t limit = static_cast<uint32_t>(std::ceil(target * 1.05));
    if (selected_tokens > limit) {
        std::vector<uint32_t> removable;
        for (uint32_t b : selected_blocks) {
            if (!mandatory.count(b)) removable.push_back(b);
        }
        std::sort(removable.begin(), removable.end(), [&](uint32_t a, uint32_t b) {
            if (std::abs(fused[a] - fused[b]) > 1e-12) return fused[a] < fused[b];
            return a > b;
        });
        for (uint32_t b : removable) {
            if (selected_tokens <= limit) break;
            selected_blocks.erase(b);
            selected_tokens -= std::min(block_size, n_tokens - b * block_size);
        }
    }

    out->mandatory_blocks.assign(mandatory.begin(), mandatory.end());
    out->block_indices.assign(selected_blocks.begin(), selected_blocks.end());
    for (uint32_t b : out->block_indices) {
        const uint32_t begin = b * block_size;
        const uint32_t end = std::min(n_tokens, begin + block_size);
        for (uint32_t i = begin; i < end; ++i) out->token_indices.push_back(i);
    }

    if (out->token_indices.size() > limit) {
        std::set<uint32_t> mandatory_tokens;
        for (uint32_t b : mandatory) {
            const uint32_t begin = b * block_size;
            const uint32_t end = std::min(n_tokens, begin + block_size);
            for (uint32_t i = begin; i < end; ++i) mandatory_tokens.insert(i);
        }
        std::vector<uint32_t> trimmed;
        trimmed.reserve(limit);
        for (uint32_t i : out->token_indices) {
            if (mandatory_tokens.count(i)) trimmed.push_back(i);
        }
        for (uint32_t i : out->token_indices) {
            if (trimmed.size() >= limit) break;
            if (!mandatory_tokens.count(i)) trimmed.push_back(i);
        }
        std::sort(trimmed.begin(), trimmed.end());
        out->token_indices = std::move(trimmed);
    }

    // The last prompt token must be present to produce the first-token logits.
    if (!out->token_indices.empty() &&
        out->token_indices.back() != n_tokens - 1) {
        out->token_indices.push_back(n_tokens - 1);
        std::sort(out->token_indices.begin(), out->token_indices.end());
        out->token_indices.erase(
            std::unique(out->token_indices.begin(), out->token_indices.end()),
            out->token_indices.end());
    }
    out->selected_tokens = static_cast<uint32_t>(out->token_indices.size());
    return Status::success();
}

SelectorWeights semantic_priority_weights() {
    // AR005: with the embedding engine active, semantic similarity is the
    // strongest available relevance signal (bge query/block cosine vs the
    // hashed-embedding proxies); 0.50 lets it decide while the remaining
    // channels break ties and cover engine blind spots.
    SelectorWeights w;
    w.lexical = 0.20;
    w.diversity = 0.15;
    w.position = 0.15;
    w.draft = 0.50;
    return w;
}

} // namespace edge_prefill
