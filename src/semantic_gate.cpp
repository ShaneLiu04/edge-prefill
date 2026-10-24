#include "edge_prefill/semantic_gate.h"

#include <algorithm>
#include <utility>

namespace edge_prefill {
namespace {

Status invalid(const std::string& message) {
    return Status::error(ErrorCode::InvalidArgument, message);
}

} // namespace

Status broadcast_block_scores(const std::vector<double>& block_scores,
                              uint32_t block_size,
                              size_t n_tokens,
                              std::vector<double>* per_token_scores) {
    if (!per_token_scores) return invalid("null per-token score output");
    if (block_size == 0) return invalid("block size is 0");
    if (block_scores.empty()) return invalid("empty block scores");
    if (n_tokens == 0) return invalid("n_tokens is 0");
    const size_t n_blocks_needed = (n_tokens + block_size - 1) / block_size;
    if (block_scores.size() != n_blocks_needed) {
        return invalid("block score count " + std::to_string(block_scores.size()) +
                       " does not match the blocking of " +
                       std::to_string(n_tokens) + " tokens (" +
                       std::to_string(n_blocks_needed) + " blocks)");
    }
    per_token_scores->clear();
    per_token_scores->resize(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        // Bit-identical copy within a block: the broadcast property the
        // downstream gating relies on.
        (*per_token_scores)[i] = block_scores[i / block_size];
    }
    return Status::success();
}

EmbeddingSelectionScorer::EmbeddingSelectionScorer(
        EmbeddingScorer* scorer,
        const std::vector<std::string>* token_pieces,
        uint32_t block_size)
        : scorer_(scorer), token_pieces_(token_pieces), block_size_(block_size) {}

Status EmbeddingSelectionScorer::score(const std::vector<llama_token>& tokens,
                                       const std::string& query,
                                       std::vector<double>* per_token_scores) {
    if (!per_token_scores) return invalid("null per-token score output");
    if (!scorer_) return invalid("embedding scorer is null");
    if (!scorer_->loaded()) return invalid("embedding scorer is not loaded");
    if (!token_pieces_) return invalid("token pieces pointer is null");
    if (block_size_ == 0) return invalid("block size is 0");
    if (tokens.empty()) return invalid("empty token sequence");
    if (query.empty()) return invalid("empty query");
    if (token_pieces_->size() < tokens.size()) {
        return invalid("token pieces size " + std::to_string(token_pieces_->size()) +
                       " < token count " + std::to_string(tokens.size()));
    }

    const size_t n = tokens.size();
    const size_t n_blocks = (n + block_size_ - 1) / block_size_;
    std::vector<std::string> block_texts(n_blocks);
    for (size_t b = 0; b < n_blocks; ++b) {
        const size_t begin = b * block_size_;
        const size_t end = std::min(n, begin + block_size_);
        for (size_t i = begin; i < end; ++i) {
            if (!block_texts[b].empty()) block_texts[b].push_back(' ');
            block_texts[b].append((*token_pieces_)[i]);
        }
    }

    std::vector<double> block_scores;
    auto st = scorer_->score_query_blocks(query, block_texts, &block_scores);
    if (!st.ok()) return st;
    return broadcast_block_scores(std::move(block_scores), block_size_, n,
                                  per_token_scores);
}

} // namespace edge_prefill
