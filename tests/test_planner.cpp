#include "test_framework.h"

#include <algorithm>
#include <string>

#include "edge_prefill/planner.h"

using namespace edge_prefill;

namespace {

PlannerInput input(uint32_t n, Mode mode = Mode::Balanced) {
    PlannerInput in;
    in.n_tokens = n;
    in.mode = mode;
    in.arch_supports_ffn_skip = true;
    in.n_layer = 28;
    in.n_ubatch = 512;
    return in;
}

Config config() {
    Config cfg;
    cfg.block_size = 32;
    cfg.chunk_size = 512;
    cfg.budget = 0.5;
    return cfg;
}

bool has(const Plan& plan, const char* value) {
    return std::find(plan.reasons.begin(), plan.reasons.end(), value) != plan.reasons.end();
}

} // namespace

TEST_CASE(planner_short_prompt_dense) {
    Config cfg = config();
    Plan plan;
    CHECK_OK(Planner::make_plan(input(256), cfg, &plan));
    CHECK(plan.dense_fallback);
    CHECK_NEAR(plan.budget, 1.0, 1e-12);
    CHECK(has(plan, "short-prompt"));
}

TEST_CASE(planner_balanced_8k) {
    Config cfg = config();
    Plan plan;
    CHECK_OK(Planner::make_plan(input(8192), cfg, &plan));
    CHECK(!plan.dense_fallback);
    CHECK_NEAR(plan.budget, 0.5, 1e-12);
    CHECK_EQ(plan.skip_ffn_mask, 0u);
    CHECK(plan.kv_type == KvPrecision::Q8_0);
    CHECK(std::find(plan.passes.begin(), plan.passes.end(), "sparse_prefill") !=
          plan.passes.end());
    CHECK(std::find(plan.passes.begin(), plan.passes.end(), "kv_precision") !=
          plan.passes.end());
}

TEST_CASE(planner_latency_and_quality_modes) {
    Config cfg = config();
    Plan latency;
    CHECK_OK(Planner::make_plan(input(8192, Mode::Latency), cfg, &latency));
    CHECK_NEAR(latency.budget, 0.4, 1e-12);
    CHECK(latency.skip_ffn_mask != 0);

    Plan quality;
    CHECK_OK(Planner::make_plan(input(8192, Mode::Quality), cfg, &quality));
    CHECK_NEAR(quality.budget, 0.7, 1e-12);
    CHECK_EQ(quality.skip_ffn_mask, 0u);
    CHECK(quality.kv_type == KvPrecision::F16);
}

TEST_CASE(planner_dense_mode) {
    Config cfg = config();
    Plan plan;
    CHECK_OK(Planner::make_plan(input(8192, Mode::Dense), cfg, &plan));
    CHECK(plan.dense_fallback);
    CHECK_NEAR(plan.budget, 1.0, 1e-12);
    CHECK(has(plan, "mode-dense"));
}

TEST_CASE(planner_clamps_16k) {
    Config cfg = config();
    Plan plan;
    CHECK_OK(Planner::make_plan(input(16384), cfg, &plan));
    CHECK_NEAR(plan.budget, 0.25, 1e-12);
    CHECK(has(plan, "budget-clamped"));
}

TEST_CASE(planner_negative_gain_falls_back) {
    Config cfg = config();
    auto in = input(1024);
    in.costs.dense_ms_per_token = 1e-6;
    Plan plan;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK(plan.dense_fallback);
    CHECK(has(plan, "negative-estimated-gain"));
}

TEST_CASE(planner_prefix_reuse) {
    Config cfg = config();
    auto in = input(8192);
    in.prefix_hit = 6000;
    Plan plan;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_NEAR(plan.budget, 0.7, 1e-12);
    CHECK(has(plan, "prefix-reuse"));
    CHECK(std::find(plan.passes.begin(), plan.passes.end(), "prefix_reuse") !=
          plan.passes.end());
}

TEST_CASE(planner_overrides) {
    Config cfg = config();
    cfg.budget_overridden = true;
    cfg.budget = 0.6;
    cfg.kv_overridden = true;
    cfg.kv_type = KvPrecision::Q4_0;
    cfg.skip_ffn_overridden = true;
    cfg.skip_ffn_mask = 0x3;
    Plan plan;
    CHECK_OK(Planner::make_plan(input(8192), cfg, &plan));
    CHECK_NEAR(plan.budget, 0.6, 1e-12);
    CHECK(plan.kv_type == KvPrecision::Q4_0);
    CHECK_EQ(plan.skip_ffn_mask, 0x3u);
    CHECK(has(plan, "cli-override"));
}

TEST_CASE(planner_ffn_unsupported_is_disabled) {
    Config cfg = config();
    auto in = input(8192, Mode::Latency);
    in.arch_supports_ffn_skip = false;
    Plan plan;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_EQ(plan.skip_ffn_mask, 0u);
    CHECK(has(plan, "pafs-disabled"));
}

TEST_CASE(planner_invalid_inputs) {
    Config cfg = config();
    Plan plan;
    auto in = input(8192);
    in.n_ubatch = 0;
    CHECK_ERR(Planner::make_plan(in, cfg, &plan), ErrorCode::InvalidArgument);

    in = input(8192);
    in.prefix_hit = 9000;
    CHECK_ERR(Planner::make_plan(in, cfg, &plan), ErrorCode::InvalidArgument);

    in = input(8192);
    in.mode = static_cast<Mode>(999);
    CHECK_ERR(Planner::make_plan(in, cfg, &plan), ErrorCode::InvalidArgument);

    CHECK_ERR(Planner::make_plan(input(8192), cfg, nullptr), ErrorCode::InvalidArgument);
}
