#pragma once

#include <cstdint>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

struct CostSample {
    uint32_t tokens = 0;
    double ttft_ms = 0.0;
};

// AR016 FR-2: affine cost model TTFT ~= a + b * tokens, fitted on real
// measurements. The model is only allowed to drive planning when it explains
// the samples well (r2 and sample-count gates); a bad fit is an error, never
// a silent guess.
struct CostModel {
    double intercept_ms = 0.0;
    double ms_per_token = 0.0;
    double r2 = 0.0;
    double max_rel_error = 0.0;
    uint32_t n_samples = 0;
    bool valid = false;
};

// Fits the model. Fails when fewer than 5 samples, all token counts equal, or
// r2 < min_r2 (default 0.90).
Status fit_cost_model(const std::vector<CostSample>& samples, CostModel* out,
                      double min_r2 = 0.90);

// Conservative upper bound: (a + b * tokens) * (1 + safety).
double predict_ttft_ms(const CostModel& model, uint32_t tokens, double safety = 0.15);

// Largest token count whose conservative prediction stays within target_ms,
// clamped to [lo, hi]. Returns lo when even lo cannot fit.
uint32_t max_tokens_within(const CostModel& model, double target_ms, uint32_t lo,
                           uint32_t hi, double safety = 0.15);

} // namespace edge_prefill
