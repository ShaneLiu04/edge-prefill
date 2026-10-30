#include "edge_prefill/cost_model.h"

#include "test_framework.h"

using namespace edge_prefill;

namespace {

std::vector<CostSample> linear_samples() {
    // ttft = 200 + 5 * tokens, exactly.
    std::vector<CostSample> s;
    for (uint32_t t : {64u, 128u, 256u, 512u, 1024u}) {
        s.push_back(CostSample{t, 200.0 + 5.0 * static_cast<double>(t)});
    }
    return s;
}

} // namespace

TEST_CASE(cost_model_fits_affine_relation) {
    CostModel model;
    const Status st = fit_cost_model(linear_samples(), &model);
    CHECK(st.ok());
    CHECK(model.valid);
    CHECK_NEAR(model.intercept_ms, 200.0, 1e-6);
    CHECK_NEAR(model.ms_per_token, 5.0, 1e-6);
    CHECK_NEAR(model.r2, 1.0, 1e-9);
    CHECK_EQ(model.n_samples, 5u);
}

TEST_CASE(cost_model_requires_enough_samples) {
    const std::vector<CostSample> all = linear_samples();
    std::vector<CostSample> few(all.begin(), all.begin() + 4);
    CostModel model;
    CHECK(!fit_cost_model(few, &model).ok());
    CHECK(!model.valid);
}

TEST_CASE(cost_model_rejects_single_token_count) {
    std::vector<CostSample> same;
    for (int i = 0; i < 6; ++i) same.push_back(CostSample{256u, 1500.0 + i});
    CostModel model;
    CHECK(!fit_cost_model(same, &model).ok());
}

TEST_CASE(cost_model_rejects_noisy_fit) {
    std::vector<CostSample> noisy = linear_samples();
    noisy[0].ttft_ms = 2000.0;
    noisy[2].ttft_ms = 400.0;
    noisy[4].ttft_ms = 9000.0;
    CostModel model;
    const Status st = fit_cost_model(noisy, &model);
    CHECK(!st.ok());
}

TEST_CASE(cost_model_prediction_is_conservative) {
    CostModel model;
    CHECK(fit_cost_model(linear_samples(), &model).ok());
    const double plain = model.intercept_ms + model.ms_per_token * 512.0;
    CHECK(predict_ttft_ms(model, 512, 0.15) > plain);
    CHECK_NEAR(predict_ttft_ms(model, 512, 0.0), plain, 1e-9);
}

TEST_CASE(cost_model_max_tokens_respects_target_and_clamps) {
    CostModel model;
    CHECK(fit_cost_model(linear_samples(), &model).ok());
    // target 1200 ms with 15% safety -> (1200/1.15 - 200)/5 ~= 168
    const uint32_t fit = max_tokens_within(model, 1200.0, 1u, 4096u, 0.15);
    CHECK(fit > 150u);
    CHECK(fit < 180u);
    CHECK(predict_ttft_ms(model, fit, 0.15) <= 1200.0);

    // Impossible target collapses to the lower clamp.
    CHECK_EQ(max_tokens_within(model, 1.0, 7u, 4096u, 0.15), 7u);
    // Huge target collapses to the upper clamp.
    CHECK_EQ(max_tokens_within(model, 1e9, 1u, 999u, 0.15), 999u);
    // Hi < lo is normalised.
    CHECK_EQ(max_tokens_within(model, 1e9, 500u, 100u, 0.15), 500u);
}

TEST_CASE(cost_model_invalid_model_predicts_lo) {
    CostModel model; // not fitted
    CHECK_EQ(max_tokens_within(model, 1000.0, 32u, 4096u, 0.15), 32u);
}
