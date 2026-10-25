#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

// AR007 FR-3: measured budget-vs-quality curve.
//
// The file is line oriented (`key = value`), matching the repository's
// config format so the generator script and the loader cannot drift:
//
//   # comment
//   model_sha256 = 6a1a2eb6...
//   quality_floor = 0.95
//   4000 = 0.15
//   8000 = 0.12
//   16000 = 0.12
//   32000 = 0.13
//
// Numeric keys are prompt lengths (tokens), values are the recommended
// admission budget that reached quality_floor on that model. The loader is
// strict: an unparsable value is an error, never a silent default.
struct BudgetCurve {
    std::string model_sha256;
    double quality_floor = 0.95;
    // Sorted ascending by length.
    std::vector<std::pair<uint32_t, double>> points;

    // Largest length not exceeding n_tokens (or the smallest point when the
    // prompt is shorter than every measured length). Returns false when the
    // curve is empty.
    bool recommend(uint32_t n_tokens, double* budget) const;
    // True when the curve records the same model fingerprint. An empty
    // recorded hash means the file predates fingerprinting and is rejected.
    bool matches_model(const std::string& sha256) const;
};

Status load_budget_curve(const std::string& path, BudgetCurve* out);

} // namespace edge_prefill
