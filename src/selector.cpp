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
    const auto [lo_it, hi_it] = std::minmax_element(values->begin(), values->end());
    // The bounds must be copied *before* the loop: they are iterators into the
    // very vector that is being overwritten. Reading `*lo_it` inside the loop
    // returned 0 as soon as the minimum element had been normalised, so every
    // element after it was mapped to `v / range` instead of `(v - lo) / range`
    // and the signal was no longer in [0, 1] (measured: diversity up to 3.2,
    // position up to 1.16 at 32K). The over-ranged priors then outranked the
    // lexical evidence and the 32K admission spent 62% of its budget on
    // zero-evidence filler blocks.
    const double lo = *lo_it;
    const double hi = *hi_it;
    const double range = hi - lo;
    if (range < 1e-12) {
        for (double& v : *values) v = 0.5;
        return;
    }
    for (double& v : *values) v = (v - lo) / range;
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
    json.key("phrase_blocks");
    json.value(static_cast<uint64_t>(phrase_blocks));
    json.key("specific_terms");
    json.value(static_cast<uint64_t>(specific_terms));
    json.key("coverage_queries");
    json.value(static_cast<uint64_t>(coverage_queries));
    json.key("evidence_blocks");
    json.value(static_cast<uint64_t>(evidence_blocks));
    json.key("stop_reason");
    json.value(stop_reason);
    json.key("anchor_blocks");
    json.value(static_cast<uint64_t>(anchor_blocks));
    json.key("mmr_penalized");
    json.value(static_cast<uint64_t>(mmr_penalized));
    json.key("structure_tokens");
    json.value(static_cast<uint64_t>(structure_tokens));
    json.key("evidence_tokens");
    json.value(static_cast<uint64_t>(evidence_tokens));
    json.key("span_tokens");
    json.value(static_cast<uint64_t>(span_tokens));
    json.key("uncovered_facts");
    json.value(static_cast<uint64_t>(uncovered_facts));
    json.key("lexical_concentration");
    json.value(lexical_concentration);
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
        json.key("phrase");
        json.value(s.phrase);
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
        out->stop_reason = "dense";
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
    // AR008 FR-1: the unfiltered word sequence per block, in token order.
    // The filtered `block_terms` drops one-character terms (a lone digit),
    // which is exactly the discriminator that separates "vault 5" from
    // "vault 6"; positional phrases must see it.
    const bool phrase_enabled = opts.phrase_weight > 0.0;
    // The unfiltered per-block word sequence feeds both the phrase signal and
    // the AR009 coverage sub-queries, so it is always built (one string per
    // term, no extra tokenisation).
    std::vector<std::vector<std::string>> block_seq(n_blocks);
    std::vector<std::string> flat_seq;
    std::vector<uint32_t> flat_tok;
    if (phrase_enabled) {
        flat_seq.reserve(n_tokens * 2);
        flat_tok.reserve(n_tokens * 2);
    }
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const std::string norm = normalize_piece(token_pieces[i]);
        auto terms = terms_from_text(norm);
        for (const auto& term : terms) {
            block_seq[i / block_size].push_back(term);
            if (phrase_enabled) {
                flat_seq.push_back(term);
                // A piece can hold several words, so a flat-sequence index is
                // not a token index. The phrase co-occurrence below needs the
                // token each term came from (the previous code indexed per_pos
                // with the flat index and read past the end whenever a piece
                // held more than one word).
                flat_tok.push_back(i);
            }
            if (term.size() >= 2 || static_cast<unsigned char>(term[0]) >= 0x80) {
                block_terms[i / block_size].push_back(term);
                block_terms_unique[i / block_size].insert(term);
                token_terms[i].push_back(term);
            }
        }
        token_embeddings[i] = hashed_embedding(norm);
    }

    std::unordered_set<std::string> query_terms;
    // The raw query sequence always exists: the portfolio profile counts
    // discriminative terms even when the phrase signal itself is disabled
    // (phrase_weight=0), otherwise routing would silently degrade to the
    // "flat" bucket.
    std::vector<std::string> query_seq;
    for (const auto& term : terms_from_text(query_text)) {
        query_seq.push_back(term);
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

    // AR008 FR-1: positional phrase admission. A query bigram such as
    // "vault 5" is one discriminative unit: the document only contains it
    // once, while each of its words occurs in every sibling sentence. The
    // block level IDF makes a phrase hit decisive without any model call.
    std::vector<std::string> query_phrases;
    std::vector<std::unordered_set<std::string>> block_bigrams;
    std::vector<double> phrase(n_blocks, 0.0);
    std::unordered_map<std::string, uint32_t> phrase_df;
    if (phrase_enabled && query_seq.size() >= 2) {
        auto discriminative = [](const std::string& term) {
            if (term.empty()) return false;
            bool has_digit = false;
            for (char c : term) {
                if (c >= '0' && c <= '9') has_digit = true;
            }
            return has_digit || term.size() >= 5 ||
                   static_cast<unsigned char>(term[0]) >= 0x80;
        };
        std::unordered_set<std::string> seen;
        for (size_t i = 0; i + 1 < query_seq.size(); ++i) {
            const std::string& a = query_seq[i];
            const std::string& b = query_seq[i + 1];
            if (a.empty() || b.empty()) continue;
            if (!discriminative(a) && !discriminative(b)) continue;
            const std::string key = a + " " + b;
            if (seen.insert(key).second) query_phrases.push_back(key);
        }
        // Deterministic order for reproducible tie-breaking.
        std::sort(query_phrases.begin(), query_phrases.end());
        if (!query_phrases.empty()) {
            block_bigrams.assign(n_blocks, {});
            for (uint32_t b = 0; b < n_blocks; ++b) {
                const auto& seq = block_seq[b];
                for (size_t i = 0; i + 1 < seq.size(); ++i) {
                    if (seq[i].empty() || seq[i + 1].empty()) continue;
                    block_bigrams[b].insert(seq[i] + " " + seq[i + 1]);
                }
            }
            // A phrase that straddles a block boundary ("... for vault" | "3
            // is ...") is invisible to per-block bigrams, and that is exactly
            // how a needle can land on the wrong side of a 32-token cut. Add
            // the crossing pair to both neighbours: they are adjacent in the
            // text even though they live in different blocks.
            for (uint32_t b = 0; b + 1 < n_blocks; ++b) {
                if (block_seq[b].empty() || block_seq[b + 1].empty()) continue;
                const std::string cross = block_seq[b].back() + " " + block_seq[b + 1].front();
                if (cross.empty() || cross == " ") continue;
                bool wanted = false;
                for (const auto& p : query_phrases) {
                    if (p == cross) {
                        wanted = true;
                        break;
                    }
                }
                if (!wanted) continue;
                block_bigrams[b].insert(cross);
                block_bigrams[b + 1].insert(cross);
            }
            for (uint32_t b = 0; b < n_blocks; ++b) {
                uint32_t hits = 0;
                for (const auto& p : query_phrases) {
                    if (block_bigrams[b].count(p)) ++hits;
                }
                if (hits > 0) ++out->phrase_blocks;
            }
            for (const auto& p : query_phrases) {
                uint32_t n = 0;
                for (uint32_t b = 0; b < n_blocks; ++b) {
                    if (block_bigrams[b].count(p)) ++n;
                }
                phrase_df[p] = n;
            }
            for (uint32_t b = 0; b < n_blocks; ++b) {
                double score = 0.0;
                for (const auto& p : query_phrases) {
                    if (!block_bigrams[b].count(p)) continue;
                    score += std::log((static_cast<double>(n_blocks) + 1.0) /
                                      (static_cast<double>(phrase_df[p]) + 1.0)) + 1.0;
                }
                const uint32_t size = std::min(block_size, n_tokens - b * block_size);
                phrase[b] = score / std::sqrt(static_cast<double>(std::max<uint32_t>(1, size)));
            }
        }
    }
    {
        auto discriminative = [](const std::string& term) {
            if (term.empty()) return false;
            bool has_digit = false;
            for (char c : term) {
                if (c >= '0' && c <= '9') has_digit = true;
            }
            return has_digit || term.size() >= 5 ||
                   static_cast<unsigned char>(term[0]) >= 0x80;
        };
        std::unordered_set<std::string> specific;
        for (const auto& term : query_seq) {
            if (discriminative(term)) specific.insert(term);
        }
        out->specific_terms = static_cast<uint32_t>(specific.size());
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
        // AR008 FR-1: phrase hits also land in the per-position vector so
        // the query-chunk (AR004) and in-graph page (AR006) paths see the
        // same discriminator, not just the block selector.
        if (!query_phrases.empty()) {
            const std::vector<std::string>& flat = flat_seq;
            std::unordered_map<std::string, double> phrase_weight;
            for (const auto& p : query_phrases) {
                phrase_weight[p] = std::log((static_cast<double>(n_blocks) + 1.0) /
                                            (static_cast<double>(phrase_df[p]) + 1.0)) + 1.0;
            }
            for (size_t i = 0; i + 1 < flat.size(); ++i) {
                if (flat[i].empty() || flat[i + 1].empty()) continue;
                const auto it = phrase_weight.find(flat[i] + " " + flat[i + 1]);
                if (it == phrase_weight.end()) continue;
                per_pos[flat_tok[i]] += it->second;
                per_pos[flat_tok[i + 1]] += it->second;
            }
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
    // Only normalise the phrase signal when something matched: minmax maps a
    // constant vector to 0.5, which would add a uniform (ranking-neutral but
    // misleading) boost and report phrase_blocks=0 with a non-zero signal.
    if (std::any_of(phrase.begin(), phrase.end(), [](double v) { return v > 1e-12; })) {
        minmax(&phrase);
    }
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

    // AR010 FR-1: split the block score into two kinds of quantity.
    //
    //   evidence = does this block contain what the question asks for?
    //   prior    = is it a good place to put attention (sink/recent, dense text)?
    //
    // Piling both into one additive score lets a prior *create* admission: at
    // 32K the diversity/position terms alone pushed 30 zero-evidence filler
    // blocks into a 50-block selection (62% of the budget). Priors now only
    // order blocks that already carry evidence, with a saturating gain so that
    // "almost no evidence" buys "almost no prior".
    constexpr double kPriorSat = 0.25;
    std::vector<double> evidence(n_blocks, 0.0);
    std::vector<double> prior(n_blocks, 0.0);
    double evidence_max = 0.0;
    for (uint32_t b = 0; b < n_blocks; ++b) {
        evidence[b] = weights.lexical * lexical[b] +
                      (draft_enabled ? weights.draft * draft_scores[b] : 0.0) +
                      opts.phrase_weight * phrase[b];
        prior[b] = weights.diversity * diversity[b] + weights.position * position[b];
        evidence_max = std::max(evidence_max, evidence[b]);
    }
    const double sat_ref = std::max(1e-12, kPriorSat * evidence_max);
    std::vector<double> fused(n_blocks, 0.0);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        out->signals[b].lexical = lexical[b];
        out->signals[b].diversity = diversity[b];
        out->signals[b].position = position[b];
        out->signals[b].draft = draft_enabled ? draft_scores[b] : 0.0;
        out->signals[b].phrase = phrase[b];
        const double gate = opts.prior_gate > 0.5
                                    ? std::min(1.0, evidence[b] / sat_ref)
                                    : 1.0;
        fused[b] = evidence[b] + gate * prior[b];
        out->signals[b].fused = fused[b];
    }

    {
        double sum = 0.0;
        double top = 0.0;
        for (double v : lexical) {
            sum += v;
            top = std::max(top, v);
        }
        out->lexical_concentration = sum > 1e-12 ? top / sum : 0.0;
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
            // AR011 FR-1: the window radius is configurable because +-2 blocks
            // is 160 tokens at block_size=32 - a fifth of a 4K budget spent on
            // neighbourhood protection alone.
            const int32_t radius = opts.anchor_span_radius > 0
                                           ? static_cast<int32_t>(
                                                     std::min<uint32_t>(opts.anchor_span_radius, 2))
                                           : 2;
            for (int32_t delta = -radius; delta <= radius; ++delta) {
                const int32_t b = static_cast<int32_t>(best_lex) + delta;
                if (b >= 0 && b < static_cast<int32_t>(n_blocks)) {
                    mandatory.insert(static_cast<uint32_t>(b));
                }
            }
        }
    }

    // AR007 FR-2: budget-aware coverage sampling. The sample count scales
    // with the token budget (a quarter of the selected blocks, at least 4),
    // so a small budget still places a coverage probe in every region of
    // the prompt instead of one probe per fixed 8-block bucket.
    const uint32_t target = static_cast<uint32_t>(
        std::ceil(static_cast<double>(n_tokens) * opts.budget));
    const uint32_t target_blocks = std::max<uint32_t>(
        1, static_cast<uint32_t>(std::ceil(static_cast<double>(target) / block_size)));
    uint32_t n_samples = std::max<uint32_t>(
        4, static_cast<uint32_t>(std::lround(0.25 * static_cast<double>(target_blocks) *
                                             std::max(0.0, opts.anchor_stride_scale))));
    n_samples = std::min<uint32_t>(n_samples, n_blocks);
    // An explicit coverage_stride still wins when the user narrowed it below
    // the automatic density (0 means "auto" in AR007).
    if (opts.coverage_stride > 1) {
        n_samples = std::min<uint32_t>(
            n_samples, (n_blocks + opts.coverage_stride - 1) / opts.coverage_stride);
    }
    const uint32_t stride = std::max<uint32_t>(1, (n_blocks + n_samples - 1) / n_samples);
    std::set<uint32_t> stride_anchors;
    for (uint32_t start = 0; start < n_blocks; start += stride) {
        const uint32_t end = std::min(n_blocks, start + stride);
        uint32_t best = start;
        for (uint32_t b = start + 1; b < end; ++b) {
            if (fused[b] > fused[best]) best = b;
        }
        stride_anchors.insert(best);
    }

    // AR009 FR-2: sub-query coverage. Bounded so a question that repeats many
    // discriminative terms cannot turn the greedy into an O(n^2 * P) scan.
    constexpr size_t kMaxCoverageSubqueries = 8;
    const double coverage_weight = opts.coverage_weight;
    // AR009 FR-2: coverage sub-queries. A question can ask for several facts
    // at once ("list the passcodes for vault 1, 2 and 3"). A top-k fill then
    // spends the whole budget on the single strongest match and the sibling
    // facts are never admitted. Submodular facility location fixes that: each
    // discriminative query term is a sub-query and contributes the gain of its
    // best *uncovered* block, so the greedy step is forced to pay for every
    // fact in turn. With fewer than two sub-queries every gain is zero and the
    // ranking stays bit-for-bit the AR008 one.
    std::vector<double> sub_idf;
    std::vector<std::vector<char>> sub_hit;  // [sub-query][block]
    std::vector<double> covered;
    double sub_idf_max = 0.0;
    {
        // Sub-queries are the query *terms* that are rare in the prompt, not
        // the bigrams: an enumerated question ("the vaults one, two and
        // three") never contains "vault two" as an adjacent pair, while the
        // word "two" still identifies its block. Rarity in the prompt is the
        // criterion, not token length: a length heuristic silently drops
        // short keys ("one", "two") and keeps template words ("passcodes").
        std::unordered_set<std::string> seen;
        std::vector<std::string> subs;
        for (const auto& term : query_seq) {
            if (term.size() < 2) continue;
            if (!seen.insert(term).second) continue;
            subs.push_back(term);
        }
        std::sort(subs.begin(), subs.end());
        std::vector<std::pair<double, const std::string*>> ranked;
        std::vector<std::vector<char>> hits_by_sub;
        for (const auto& term : subs) {
            std::vector<char> hits(n_blocks, 0);
            uint32_t n_hits = 0;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                for (const auto& t : block_seq[b]) {
                    if (t != term) continue;
                    hits[b] = 1;
                    ++n_hits;
                    break;
                }
            }
            // Only *specific* terms can identify a fact. A term that occurs in
            // most fact blocks ("vault", "secret", "passcode") is satisfied by
            // any of them, so giving it its own anchor wastes the budget on
            // the first fact and dilutes the coverage gain. Two blocks is the
            // tolerance for a sentence that straddles a block boundary.
            if (n_hits == 0 || n_hits > 2) continue;
            const double idf = std::log((static_cast<double>(n_blocks) + 1.0) /
                                        (static_cast<double>(n_hits) + 1.0)) + 1.0;
            ranked.emplace_back(idf, &term);
            hits_by_sub.push_back(std::move(hits));
        }
        std::vector<size_t> order(ranked.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (std::abs(ranked[a].first - ranked[b].first) > 1e-12) {
                return ranked[a].first > ranked[b].first;
            }
            return *ranked[a].second < *ranked[b].second;
        });
        if (order.size() > kMaxCoverageSubqueries) order.resize(kMaxCoverageSubqueries);
        for (size_t i : order) {
            sub_idf.push_back(ranked[i].first);
            sub_idf_max = std::max(sub_idf_max, ranked[i].first);
            sub_hit.push_back(std::move(hits_by_sub[i]));
        }
        if (sub_idf.size() < 2 || coverage_weight <= 0.0) {
            sub_idf.clear();
            sub_hit.clear();
            sub_idf_max = 0.0;
        } else {
            covered.assign(sub_idf.size(), 0.0);
        }
    }
    // The chat template repeats the question in the final block, so that block
    // contains every sub-query at once. Treating it as evidence would mark the
    // whole question "covered" before the first real fact is admitted (the
    // first measurement round did exactly that: coverage stayed at 0 and the
    // selection was identical to AR008). A block that echoes most of the
    // question is never evidence, so it is dropped from the sub-query hits.
    if (!sub_idf.empty()) {
        const size_t echo_threshold = std::max<size_t>(2, (sub_idf.size() + 1) / 2);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            size_t n_hit = 0;
            for (size_t p = 0; p < sub_idf.size(); ++p) {
                if (sub_hit[p][b]) ++n_hit;
            }
            if (n_hit < echo_threshold) continue;
            for (size_t p = 0; p < sub_idf.size(); ++p) sub_hit[p][b] = 0;
        }
    }
    // Coverage gain of block b: how much of the still-uncovered sub-query IDF
    // it would add, normalised to [0,1] so it composes with the fused score.
    auto block_covers = [&](uint32_t b) {
        if (sub_idf_max <= 0.0) return 0.0;
        double gain = 0.0;
        for (size_t p = 0; p < sub_idf.size(); ++p) {
            if (sub_hit[p][b]) gain += std::max(0.0, sub_idf[p] - covered[p]);
        }
        return gain / sub_idf_max;
    };
    auto mark_covered = [&](uint32_t b) {
        for (size_t p = 0; p < sub_idf.size(); ++p) {
            if (sub_hit[p][b]) covered[p] = sub_idf[p];
        }
    };

    // AR007 FR-2: entity anchors. Distinct question terms that look specific
    // (digits, long words, CJK) get their best context block pinned, so a
    // multi-fact question keeps one anchor per fact instead of spending the
    // whole budget on the single strongest lexical match.
    std::set<uint32_t> entity_anchors;
    // AR009: when sub-query coverage is active it *replaces* the AR007 entity
    // heuristic. Both do "one anchor per fact", but the heuristic keys on token
    // length (so the short keys "one"/"two" never qualify) and has no budget
    // awareness: on an 8K prompt with 6 planted facts it pinned 10 blocks
    // (320 tokens of a 429-token budget), which starved the relevance fill.
    if (!query_terms.empty() && sub_idf.empty()) {
        const uint32_t df_cap = std::max<uint32_t>(2, n_blocks / 50);
        std::vector<const std::string*> specific;
        for (const auto& term : query_terms) {
            if (df.find(term) == df.end()) continue;
            if (df[term] > df_cap) continue;
            bool looks_specific = term.size() >= 6;
            for (char c : term) {
                if (c >= '0' && c <= '9') looks_specific = true;
            }
            if (!looks_specific && static_cast<unsigned char>(term[0]) < 0x80) continue;
            specific.push_back(&term);
        }
        std::sort(specific.begin(), specific.end(),
                  [](const std::string* a, const std::string* b) { return *a < *b; });
        for (const std::string* term : specific) {
            uint32_t best = 0;
            double best_score = 0.0;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (!block_terms_unique[b].count(*term)) continue;
                if (lexical[b] > best_score) {
                    best_score = lexical[b];
                    best = b;
                }
            }
            if (best_score > 0.0) entity_anchors.insert(best);
        }
    }

    // AR008 FR-1: the best phrase block is the single most reliable piece of
    // evidence (a discriminative bigram usually occurs once in the prompt),
    // so it is pinned ahead of entity/stride anchors and protected from
    // trimming.
    std::set<uint32_t> phrase_anchors;
    if (!query_phrases.empty()) {
        // Pin every block that contains the *most discriminative* phrase,
        // i.e. the one with the highest block-level IDF ("vault 5" rather
        // than the template "for vault" that all sibling facts share).
        // Anchoring on the arg-max block score instead lets a sibling fact
        // win by a tokenisation accident, and anchoring on every template
        // hit drags competing answers into the context, which measurably
        // confuses the model ("vault 3" answered as "vault 1").
        if (!sub_idf.empty()) {
            // Multi-fact question: every requested fact gets its own anchor.
            // Pinning all blocks of one term would represent a single fact and
            // leave the rest of the answer outside the context.
            const uint32_t phrase_budget = target / 2;
            uint32_t used = 0;
            for (size_t p = 0; p < sub_idf.size(); ++p) {
                uint32_t best = n_blocks;
                double best_score = 0.0;
                for (uint32_t b = 0; b < n_blocks; ++b) {
                    if (!sub_hit[p][b]) continue;
                    // Pure facility location: sub-query p scores its own block.
                    // Summing the other sub-queries' IDF made every sub pick
                    // the same "richest" block, so the remaining facts fell
                    // through to no anchor at all. Ties go to the block that is
                    // most relevant to the whole question.
                    const double score = sub_idf[p] + lexical[b];
                    if (best == n_blocks || score > best_score) {
                        best_score = score;
                        best = b;
                    }
                }
                if (best == n_blocks || phrase_anchors.count(best)) continue;
                const uint32_t size = std::min(block_size, n_tokens - best * block_size);
                if (used + size > phrase_budget && !phrase_anchors.empty()) continue;
                phrase_anchors.insert(best);
                used += size;
            }
        }
        std::string best_phrase;
        uint32_t best_df = 0;
        for (const auto& p : query_phrases) {
            if (sub_idf.size() >= 2) break;
            const auto it = phrase_df.find(p);
            if (it == phrase_df.end() || it->second == 0) continue;
            if (best_phrase.empty() || it->second < best_df) {
                best_phrase = p;
                best_df = it->second;
            }
        }
        if (!best_phrase.empty()) {
            const uint32_t phrase_budget = target / 2;
            uint32_t used = 0;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (!block_bigrams[b].count(best_phrase)) continue;
                const uint32_t size = std::min(block_size, n_tokens - b * block_size);
                if (used + size > phrase_budget && !phrase_anchors.empty()) break;
                phrase_anchors.insert(b);
                used += size;
            }
        }
    }

    // Anchor budget cap: anchors must not crowd out the relevance fill.
    // Trim on the priority order stride > entity > phrase > recent, never
    // touching the sink page or the final block (they carry the attention
    // sink and the first-token logits).
    std::set<uint32_t> anchors;
    for (uint32_t b = 0; b < std::min<uint32_t>(opts.sink_blocks, n_blocks); ++b) anchors.insert(b);
    anchors.insert(n_blocks - 1);
    for (uint32_t b : mandatory) anchors.insert(b);
    for (uint32_t b : phrase_anchors) anchors.insert(b);
    for (uint32_t b : entity_anchors) anchors.insert(b);
    for (uint32_t b : stride_anchors) anchors.insert(b);
    auto anchor_tokens = [&](const std::set<uint32_t>& set) {
        uint32_t n = 0;
        for (uint32_t b : set) n += std::min(block_size, n_tokens - b * block_size);
        return n;
    };
    // AR011 FR-1: separate the *structural* anchors (sink + recent, never
    // trimmed) from the evidence anchors. Measured before this change: a 4K
    // prompt at b=0.08 selected 310 tokens of which 9/10 blocks were mandatory,
    // i.e. anchor_ratio=0.35 was already breached by the untrimmable phrase
    // anchors and the best-lexical +-2 window. The budget was a quota for
    // structure, not for evidence.
    std::set<uint32_t> structure_blocks;
    for (uint32_t b = 0; b < std::min<uint32_t>(opts.sink_blocks, n_blocks); ++b) {
        structure_blocks.insert(b);
    }
    structure_blocks.insert(n_blocks - 1);
    const uint32_t structure_tokens = anchor_tokens(structure_blocks);
    // The cap only has to be enforced when the anchors actually dominate the
    // budget. At 16K the legacy anchor set is 288 tokens against a 781-token
    // budget: the fill is evidence-limited there and structural accounting just
    // admits *more* evidence (measured 280 -> 320 tokens for no quality gain).
    // Where the budget is tight (4K/8K, multi-fact) the anchors are 70-90% of
    // the budget and the accounting is what decides the context.
    const uint32_t anchor_tokens_raw = anchor_tokens(anchors);
    const bool structure_applies = opts.structure_budget > 0 &&
                                   anchor_tokens_raw > target / 2;
    uint32_t anchor_cap = 0;
    if (structure_applies) {
        // Structure gets an absolute allowance; anchor_ratio then applies to the
        // remaining evidence budget instead of the whole budget.
        uint32_t struct_budget = std::max<uint32_t>(structure_tokens, opts.structure_budget);
        const uint32_t struct_max = std::max<uint32_t>(structure_tokens, target / 2);
        if (struct_budget > struct_max) {
            log_warn("structure_budget " + std::to_string(opts.structure_budget) +
                     " exceeds half the token budget; clamping to " +
                     std::to_string(struct_max));
            struct_budget = struct_max;
        }
        const uint32_t evidence_budget = std::max<uint32_t>(
            block_size, target > struct_budget ? target - struct_budget : block_size);
        anchor_cap = struct_budget + static_cast<uint32_t>(std::ceil(
            static_cast<double>(evidence_budget) *
            std::clamp(opts.anchor_ratio, 0.05, 1.0)));
    } else {
        anchor_cap = std::max<uint32_t>(block_size, static_cast<uint32_t>(
            std::ceil(static_cast<double>(target) * std::clamp(opts.anchor_ratio, 0.05, 1.0))));
    }
    for (auto it = stride_anchors.rbegin(); it != stride_anchors.rend() &&
                                             anchor_tokens(anchors) > anchor_cap; ++it) {
        // Phrase anchors are the only discriminative evidence in the prompt;
        // they outrank every positional sample, so the stride trim must not
        // take them (this was how a pinned "vault 3" block got evicted by a
        // coverage sample in the first measurement round).
        if (anchors.count(*it) && *it != 0 && *it != n_blocks - 1 && !mandatory.count(*it) &&
            !phrase_anchors.count(*it)) {
            anchors.erase(*it);
        }
    }
    for (auto it = entity_anchors.rbegin(); it != entity_anchors.rend() &&
                                             anchor_tokens(anchors) > anchor_cap; ++it) {
        if (anchors.count(*it) && *it != 0 && *it != n_blocks - 1 && !mandatory.count(*it) &&
            !phrase_anchors.count(*it)) {
            anchors.erase(*it);
        }
    }
    if (structure_applies) {
        // AR011: the cap must also bind the anchors the legacy path protected.
        // Drop phrase anchors (weakest evidence first, by fused score) and then
        // shrink the best-lexical window, keeping the peak block itself.
        std::vector<uint32_t> droppable;
        for (uint32_t b : anchors) {
            if (structure_blocks.count(b) || !phrase_anchors.count(b)) continue;
            droppable.push_back(b);
        }
        std::sort(droppable.begin(), droppable.end(), [&](uint32_t a, uint32_t b) {
            if (std::abs(fused[a] - fused[b]) > 1e-12) return fused[a] < fused[b];
            return a > b;
        });
        for (uint32_t b : droppable) {
            if (anchor_tokens(anchors) <= anchor_cap) break;
            anchors.erase(b);
        }
        // Shrink best_lex +- radius: drop the ring furthest from the peak first.
        if (anchor_tokens(anchors) > anchor_cap) {
            uint32_t peak = n_blocks;
            double peak_score = -1.0;
            for (uint32_t b : anchors) {
                if (structure_blocks.count(b)) continue;
                if (fused[b] > peak_score) {
                    peak_score = fused[b];
                    peak = b;
                }
            }
            if (peak != n_blocks) {
                std::vector<uint32_t> ring;
                for (uint32_t b : anchors) {
                    if (structure_blocks.count(b) || b == peak) continue;
                    ring.push_back(b);
                }
                std::sort(ring.begin(), ring.end(), [&](uint32_t a, uint32_t b) {
                    const uint32_t da = a > peak ? a - peak : peak - a;
                    const uint32_t db = b > peak ? b - peak : peak - b;
                    if (da != db) return da > db;
                    return a > b;
                });
                for (uint32_t b : ring) {
                    if (anchor_tokens(anchors) <= anchor_cap) break;
                    anchors.erase(b);
                }
            }
        }
    }
    mandatory = anchors;
    out->anchor_blocks = static_cast<uint32_t>(anchors.size());

    // AR007 FR-3: MMR-guided fill. Blocks near-duplicating an already
    // selected block are down-weighted instead of being dropped, so repeated
    // boilerplate cannot eat the budget while genuine evidence still wins.
    auto jaccard = [&](uint32_t a, uint32_t b) {
        const auto& sa = block_terms_unique[a];
        const auto& sb = block_terms_unique[b];
        if (sa.empty() || sb.empty()) return 0.0;
        size_t inter = 0;
        const auto& small = sa.size() <= sb.size() ? sa : sb;
        const auto& large = sa.size() <= sb.size() ? sb : sa;
        for (const auto& t : small) {
            if (large.count(t)) ++inter;
        }
        const size_t uni = sa.size() + sb.size() - inter;
        return uni == 0 ? 0.0 : static_cast<double>(inter) / static_cast<double>(uni);
    };
    const double mmr_threshold = std::clamp(opts.mmr_threshold, 0.0, 1.0);
    const double mmr_lambda = std::clamp(opts.mmr_lambda, 0.0, 1.0);

    std::set<uint32_t> selected_blocks;
    uint32_t selected_tokens = 0;
    for (uint32_t b : mandatory) {
        selected_blocks.insert(b);
        selected_tokens += std::min(block_size, n_tokens - b * block_size);
    }
    for (uint32_t b : selected_blocks) mark_covered(b);

    // AR011 FR-2: coverage reservation. The AR009 facility-location gain only
    // *bribes* the greedy; it does not promise anything. Measured on the
    // six-fact family the fill stopped with q_min=0.087 - two of the asked
    // facts had no admitted evidence at all. Reserve one block per uncovered
    // sub-query before the greedy starts, in IDF order, and report the
    // shortfall honestly when the budget cannot hold them all.
    if (opts.coverage_reserve > 0 && !sub_idf.empty()) {
        std::vector<size_t> order(sub_idf.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (std::abs(sub_idf[a] - sub_idf[b]) > 1e-12) return sub_idf[a] > sub_idf[b];
            return a < b;
        });
        for (size_t p : order) {
            if (covered[p] > 0.0) continue;
            bool hittable = false;
            for (uint32_t b = 0; b < n_blocks && !hittable; ++b) {
                if (sub_hit[p][b]) hittable = true;
            }
            // A sub-query with no non-echo block anywhere cannot be covered by
            // any policy; counting it as a shortfall would make single-fact
            // prompts look broken (measured: 5 "uncovered" on a 1-fact prompt
            // whose echo blocks are excluded by design).
            if (!hittable) continue;
            uint32_t best = n_blocks;
            double best_score = -1.0;
            for (uint32_t b = 0; b < n_blocks; ++b) {
                if (selected_blocks.count(b) || !sub_hit[p][b]) continue;
                if (best_score >= 0.0 && fused[b] <= best_score) continue;
                const uint32_t size = std::min(block_size, n_tokens - b * block_size);
                if (selected_tokens + size > target) continue;
                best = b;
                best_score = fused[b];
            }
            if (best == n_blocks) {
                ++out->uncovered_facts;
                continue;
            }
            selected_blocks.insert(best);
            selected_tokens += std::min(block_size, n_tokens - best * block_size);
            mark_covered(best);
        }
        if (out->uncovered_facts == 0) out->stop_reason = "facts-covered";
    }


    // Incremental MMR: max_sim[b] tracks the highest similarity between b
    // and any selected block and is refreshed only when a block is added, so
    // the fill is O(n_blocks^2) set operations instead of O(n_blocks^3).
    std::vector<double> max_sim(n_blocks, 0.0);
    std::vector<double> adjusted(fused);
    std::vector<char> mmr_hit(n_blocks, 0);
    auto apply_mmr = [&](uint32_t b) {
        double score = fused[b];
        const double sim = max_sim[b];
        if (sim > mmr_threshold && mmr_threshold < 1.0) {
            const double excess = (sim - mmr_threshold) / (1.0 - mmr_threshold);
            score *= std::max(0.0, 1.0 - mmr_lambda * excess);
            if (!mmr_hit[b]) {
                mmr_hit[b] = 1;
            }
        }
        adjusted[b] = score;
    };
    for (uint32_t b : selected_blocks) {
        for (uint32_t c = 0; c < n_blocks; ++c) {
            if (c == b) continue;
            const double j = jaccard(c, b);
            if (j > max_sim[c]) max_sim[c] = j;
        }
    }
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (!selected_blocks.count(b)) apply_mmr(b);
    }
    // AR007: span growing. Evidence usually spans a block boundary (the
    // needle sentence starts mid-block), so a selected block makes its
    // immediate neighbours more attractive. Without this bias the fill
    // scatters isolated blocks across the prompt and the answer span gets
    // split away — measured as a quality drop when the budget grows.
    constexpr double kSpanGrowWeight = 0.5;
    std::vector<char> adjacent(n_blocks, 0);
    auto mark_adjacent = [&](uint32_t b) {
        if (b > 0) adjacent[b - 1] = 1;
        if (b + 1 < n_blocks) adjacent[b + 1] = 1;
    };
    for (uint32_t b : selected_blocks) mark_adjacent(b);

    // AR010 FR-2: the fill has an explicit stopping rule instead of relying on
    // "the best score happens to be exactly 0". "Fill the budget" used to be a
    // side effect of numerically broken priors; with correct normalisation the
    // same loop stopped at 210/443 tokens and quality collapsed (4K 1.000 ->
    // 0.500, multi-fact 0.878 -> 0.579).
    const bool fill_priors = opts.fill_priors > 0.5;
    while (selected_tokens < target) {
        uint32_t best = n_blocks;
        double best_score = 0.0;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            if (selected_blocks.count(b)) continue;
            const double score = adjusted[b] + (adjacent[b] ? kSpanGrowWeight : 0.0) +
                                 coverage_weight * block_covers(b);
            if (score > best_score || (score == best_score && best != n_blocks && b < best)) {
                best_score = score;
                best = b;
            }
        }
        if (best == n_blocks) {
            out->stop_reason = "no-signal";
            break;
        }
        if (evidence[best] <= 0.0 && !fill_priors) {
            out->stop_reason = "evidence-exhausted";
            break;
        }
        selected_blocks.insert(best);
        selected_tokens += std::min(block_size, n_tokens - best * block_size);
        mark_adjacent(best);
        mark_covered(best);
        for (uint32_t c = 0; c < n_blocks; ++c) {
            if (selected_blocks.count(c)) continue;
            const double j = jaccard(c, best);
            if (j > max_sim[c]) {
                max_sim[c] = j;
                apply_mmr(c);
            }
        }
    }
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (mmr_hit[b]) ++out->mmr_penalized;
    }
    for (size_t p = 0; p < sub_idf.size(); ++p) {
        if (covered[p] > 0.0) ++out->coverage_queries;
    }
    for (uint32_t b : selected_blocks) {
        if (evidence[b] > 0.0) ++out->evidence_blocks;
    }
    if (selected_tokens >= target) out->stop_reason = "budget";

    const uint32_t limit = static_cast<uint32_t>(std::ceil(target * 1.05));
    if (selected_tokens > limit) {
        std::vector<uint32_t> removable;
        for (uint32_t b : selected_blocks) {
            if (!mandatory.count(b)) removable.push_back(b);
        }
        std::sort(removable.begin(), removable.end(), [&](uint32_t a, uint32_t b) {
            if (std::abs(adjusted[a] - adjusted[b]) > 1e-12) return adjusted[a] < adjusted[b];
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

    // AR011 FR-1 audit: split the admitted tokens into the structural anchors
    // (sink/recent, never trimmed) and everything the evidence bought.
    out->structure_tokens = anchor_tokens(structure_blocks);
    out->evidence_tokens = out->token_indices.size() > out->structure_tokens
                                   ? static_cast<uint32_t>(out->token_indices.size()) -
                                             out->structure_tokens
                                   : 0;

    // AR011 FR-3: span admission. The admission unit and the execution unit are
    // different concerns: the attention pool addresses 32-token blocks, but a
    // needle sentence is ~20 tokens, so a whole-block admission pays for the
    // filler around it. Keep a token span centred on the evidence peak inside
    // each non-structural block. Blocks with no positive evidence score stay
    // whole (never guess), and neighbours that are already selected extend the
    // span so a cross-block answer is not cut in half.
    //
    // Single-fact only: with several asked facts the selection has to carry one
    // complete sentence per fact, and trimming measured q=0.576 -> 0.405
    // (multifact8k) and q_min 0.087 -> 0.000 (hard). Multi-fact prompts
    // therefore keep whole blocks; the anchor budget (FR-1) is what saves them.
    if (opts.span_admission > 0 && sub_idf.empty() && !out->token_indices.empty() &&
        out->per_pos_scores.size() == n_tokens) {
        const uint32_t span_min = std::max<uint32_t>(
            block_size / 4, std::min<uint32_t>(opts.span_min_tokens, block_size));
        // Sentence boundaries: a span centred on the evidence peak must still
        // contain the *answer*, which usually follows the matched key in the
        // same sentence ("...passcode for vault 1 is MAGENTA-FALCON-7000.").
        // A blind +-12-token window measured q=0.625 on len4k: the window
        // covered "vault 1" and dropped the code. So grow the span to the
        // enclosing sentence first, then to span_min if it is still shorter.
        auto ends_sentence = [&](uint32_t i) {
            const std::string& piece = token_pieces[i];
            if (piece.empty()) return false;
            for (char c : piece) {
                if (c == '.' || c == '!' || c == '?' || c == '\n' || c == ';') return true;
            }
            return false;
        };
        std::vector<char> keep(n_tokens, 0);
        for (uint32_t b : out->block_indices) {
            const uint32_t begin = b * block_size;
            const uint32_t end = std::min(n_tokens, begin + block_size);
            const bool whole = structure_blocks.count(b) != 0;
            uint32_t peak = begin;
            double peak_score = 0.0;
            for (uint32_t i = begin; i < end; ++i) {
                if (out->per_pos_scores[i] > peak_score) {
                    peak_score = out->per_pos_scores[i];
                    peak = i;
                }
            }
            if (whole || peak_score <= 0.0) {
                for (uint32_t i = begin; i < end; ++i) keep[i] = 1;
                continue;
            }
            uint32_t lo = begin;
            for (uint32_t i = peak; i > begin; --i) {
                if (ends_sentence(i - 1)) {
                    lo = i;
                    break;
                }
            }
            uint32_t hi = end;
            for (uint32_t i = peak; i < end; ++i) {
                if (ends_sentence(i)) {
                    hi = i + 1;
                    break;
                }
            }
            // Pad a short sentence to span_min, symmetric around the peak.
            while (hi - lo < span_min) {
                bool grew = false;
                if (hi < end) {
                    ++hi;
                    grew = true;
                }
                if (hi - lo < span_min && lo > begin) {
                    --lo;
                    grew = true;
                }
                if (!grew) break;
            }
            // Extend across a neighbouring selected block so a sentence that
            // straddles the block boundary is not cut in half.
            while (lo > 0 && keep[lo - 1]) --lo;
            while (hi < n_tokens && keep[hi]) ++hi;
            for (uint32_t i = lo; i < hi; ++i) keep[i] = 1;
        }
        std::vector<uint32_t> trimmed;
        trimmed.reserve(out->token_indices.size());
        for (uint32_t i : out->token_indices) {
            if (keep[i]) trimmed.push_back(i);
        }
        out->span_tokens = static_cast<uint32_t>(out->token_indices.size() - trimmed.size());
        if (!trimmed.empty()) {
            out->token_indices = std::move(trimmed);
        }
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
