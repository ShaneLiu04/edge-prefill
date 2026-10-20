#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct BlockSignal {
    double lexical = 0.0;
    double diversity = 0.0;
    double position = 0.0;
    double draft = 0.0;
    double fused = 0.0;
};

struct Selection {
    std::vector<uint32_t> block_indices;
    std::vector<uint32_t> token_indices;
    std::vector<BlockSignal> signals;
    std::vector<uint32_t> mandatory_blocks;
    uint32_t total_tokens = 0;
    uint32_t selected_tokens = 0;
    bool dense = false;
    std::string reason;

    std::string to_json() const;
};

struct SelectionOptions {
    uint32_t block_size = 32;
    double budget = 0.5;
    uint32_t min_dense_tokens = 512;
    uint32_t sink_blocks = 1;
    uint32_t recent_blocks = 2;
    uint32_t coverage_stride = 8;
    SelectorWeights weights;
};

class DraftScorer {
public:
    virtual ~DraftScorer() = default;
    virtual Status score(const std::vector<llama_token>& tokens,
                         const std::string& query,
                         std::vector<double>* per_token_scores) = 0;
};

class BlockSelector {
public:
    Status select(const std::vector<llama_token>& tokens,
                  const std::string& query_text,
                  const std::vector<std::string>& token_pieces,
                  const SelectionOptions& opts,
                  DraftScorer* draft,
                  Selection* out) const;
};

} // namespace edge_prefill
