#include "edge_prefill/cost_model.h"

#include <cmath>
#include <limits>

namespace edge_prefill {

Status fit_cost_model(const std::vector<CostSample>& samples, CostModel* out,
                      double min_r2) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null cost model output");
    *out = CostModel{};
    if (samples.size() < 5) {
        return Status::error(ErrorCode::InvalidArgument,
                             "cost model needs at least 5 samples");
    }
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto& s : samples) {
        if (s.tokens == 0) {
            return Status::error(ErrorCode::InvalidArgument, "cost sample with 0 tokens");
        }
        if (!(s.ttft_ms > 0.0)) {
            return Status::error(ErrorCode::InvalidArgument, "cost sample with non-positive ttft");
        }
        sum_x += static_cast<double>(s.tokens);
        sum_y += s.ttft_ms;
    }
    const double n = static_cast<double>(samples.size());
    const double mean_x = sum_x / n;
    const double mean_y = sum_y / n;
    double sxx = 0.0, sxy = 0.0, syy = 0.0;
    for (const auto& s : samples) {
        const double dx = static_cast<double>(s.tokens) - mean_x;
        const double dy = s.ttft_ms - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if (!(sxx > 0.0)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "cost model needs at least two distinct token counts");
    }
    const double b = sxy / sxx;
    if (!(b > 0.0)) {
        return Status::error(ErrorCode::Internal, "cost model slope is not positive");
    }
    const double a = mean_y - b * mean_x;
    // Prefill is sublinear in the token count (a fixed graph/setup cost is
    // amortised over the batch), so an affine fit over the sampled range can
    // legitimately land on a negative intercept. It is only a problem if it
    // makes a *sampled* point non-positive, because then the relative error and
    // the "conservative" wording stop meaning anything.
    for (const auto& s : samples) {
        if (a + b * static_cast<double>(s.tokens) <= 0.0) {
            return Status::error(ErrorCode::Internal,
                                 "cost model is not positive over the sampled range");
        }
    }
    double sse = 0.0;
    double max_rel = 0.0;
    for (const auto& s : samples) {
        const double pred = a + b * static_cast<double>(s.tokens);
        const double err = pred - s.ttft_ms;
        sse += err * err;
        const double rel = std::fabs(err) / s.ttft_ms;
        if (rel > max_rel) max_rel = rel;
    }
    const double r2 = syy > 0.0 ? 1.0 - sse / syy : 0.0;
    if (!(r2 >= min_r2)) {
        return Status::error(ErrorCode::Internal,
                             "cost model fit below r2 gate (" + std::to_string(r2) + ")");
    }
    out->intercept_ms = a;
    out->ms_per_token = b;
    out->r2 = r2;
    out->max_rel_error = max_rel;
    out->n_samples = static_cast<uint32_t>(samples.size());
    out->valid = true;
    return Status::success();
}

double predict_ttft_ms(const CostModel& model, uint32_t tokens, double safety) {
    double base = model.intercept_ms + model.ms_per_token * static_cast<double>(tokens);
    if (base < 0.0) base = 0.0;
    return base * (1.0 + (safety > 0.0 ? safety : 0.0));
}

uint32_t max_tokens_within(const CostModel& model, double target_ms, uint32_t lo,
                           uint32_t hi, double safety) {
    if (hi < lo) hi = lo;
    if (!model.valid || !(model.ms_per_token > 0.0)) return lo;
    const double scale = 1.0 + (safety > 0.0 ? safety : 0.0);
    const double budget = target_ms / scale - model.intercept_ms;
    if (!(budget > 0.0)) return lo;
    const double tokens = budget / model.ms_per_token;
    if (tokens >= static_cast<double>(hi)) return hi;
    if (tokens <= static_cast<double>(lo)) return lo;
    return static_cast<uint32_t>(tokens);
}

} // namespace edge_prefill
