#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

// AR008 FR-2: measured execution-combination portfolio.
//
// The repository has several complete ways to run a prompt (selection-level
// sparsity, the in-graph page gate with two page budgets, dense). Which one
// is best depends on cheap prompt features, not on taste: on single-needle
// retrieval the selector is both faster and as accurate, on prompts with no
// discriminative term the gate recovers facts the selector loses. The table
// below is generated from measurements (`benchmarks/scripts/portfolio_matrix.py`),
// fingerprinted to the model that produced it, and consulted per prompt.
//
// File format (strict, line oriented, `#` comments allowed):
//
//     model_sha256 = <sha256 of the GGUF>
//     default      = selection
//     <key>        = <combo>
//
// <key> = <discriminability>:<length_bucket> where discriminability is
// `low` (<= 1 discriminative query term) or `high`, length bucket is
// `lt4k` / `8k` / `16k` / `32k`, and the special key `flat:any` matches
// prompts whose evidence is neither phrased nor concentrated.
// <combo> in {selection, gate-p16, gate-p24-local32, dense}.
struct PortfolioProfile {
    uint32_t n_tokens = 0;
    uint32_t specific_terms = 0;
    uint32_t phrase_blocks = 0;
    double lexical_concentration = 0.0;
};

struct PortfolioRow {
    std::string combo = "selection";
    std::string key;
};

class PortfolioTable {
public:
    bool load(const std::string& path);
    bool empty() const { return rows_.empty(); }
    const std::vector<std::pair<std::string, std::string>>& rows() const { return rows_; }
    const std::string& model_sha256() const { return model_sha256_; }
    const std::string& default_combo() const { return default_combo_; }

    // Pick a combination for the profile. Returns false when nothing
    // matched (caller keeps its own default and records the fallback).
    bool recommend(const PortfolioProfile& profile, PortfolioRow* row) const;

    // True when the table was produced for this model (empty fingerprint
    // means "unverified", which the runner treats as a mismatch).
    bool matches_model(const std::string& sha256) const;

    // Exposed for tests and for the router's audit trail.
    static std::string length_bucket(uint32_t n_tokens);
    static std::string discriminability(uint32_t specific_terms);

private:
    std::vector<std::pair<std::string, std::string>> rows_;
    std::string model_sha256_;
    std::string default_combo_ = "selection";
    std::string path_;
};

// Loading goes through a small mtime-keyed cache so a bench run with
// thousands of samples does not re-read the table per case.
Status load_portfolio_cached(const std::string& path, PortfolioTable* out);

// AR008 FR-2: the bucket key for a profile.
std::string portfolio_key(const PortfolioProfile& profile);

} // namespace edge_prefill
