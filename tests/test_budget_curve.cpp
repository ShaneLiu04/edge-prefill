#include "test_framework.h"

#include <cstdio>
#include <fstream>
#include <string>

#include "edge_prefill/budget_curve.h"
#include "edge_prefill/config.h"
#include "edge_prefill/planner.h"

using namespace edge_prefill;

namespace {

std::string write_temp(const std::string& name, const std::string& text) {
    std::string path = "/tmp/edge_test_" + name;
    std::ofstream out(path);
    out << text;
    out.close();
    return path;
}

} // namespace

// AR007 FR-3: strict loader — valid file, comments, and a hard error for a
// malformed value (never a silent default).
TEST_CASE(budget_curve_load_and_recommend) {
    const std::string path = write_temp("curve_ok.txt",
                                        "# test curve\n"
                                        "model_sha256 = abc123\n"
                                        "quality_floor = 0.95\n"
                                        "4000 = 0.15\n"
                                        "8000 = 0.12\n"
                                        "16000 = 0.12\n"
                                        "32000 = 0.13\n");
    BudgetCurve curve;
    CHECK_OK(load_budget_curve(path, &curve));
    CHECK_EQ(curve.points.size(), 4u);
    CHECK(curve.matches_model("abc123"));
    CHECK(!curve.matches_model("def456"));
    CHECK(!curve.matches_model(""));

    double budget = 0.0;
    CHECK(curve.recommend(1000, &budget));
    CHECK_EQ(budget, 0.15);
    CHECK(curve.recommend(4000, &budget));
    CHECK_EQ(budget, 0.15);
    CHECK(curve.recommend(5000, &budget));
    CHECK_EQ(budget, 0.12);
    CHECK(curve.recommend(8000, &budget));
    CHECK_EQ(budget, 0.12);
    CHECK(curve.recommend(20000, &budget));
    CHECK_EQ(budget, 0.13);
    CHECK(curve.recommend(64000, &budget));
    CHECK_EQ(budget, 0.13);
    std::remove(path.c_str());
}

TEST_CASE(budget_curve_rejects_bad_values) {
    BudgetCurve curve;
    const std::string bad_value = write_temp("curve_bad.txt", "8000 = fast\n");
    CHECK_ERR(load_budget_curve(bad_value, &curve), ErrorCode::InvalidArgument);
    std::remove(bad_value.c_str());

    const std::string bad_key = write_temp("curve_bad2.txt", "eight_k = 0.1\n");
    CHECK_ERR(load_budget_curve(bad_key, &curve), ErrorCode::InvalidArgument);
    std::remove(bad_key.c_str());

    const std::string empty = write_temp("curve_empty.txt", "# nothing\n");
    CHECK_ERR(load_budget_curve(empty, &curve), ErrorCode::InvalidArgument);
    std::remove(empty.c_str());

    const std::string missing = "/tmp/edge_test_does_not_exist.txt";
    CHECK_ERR(load_budget_curve(missing, &curve), ErrorCode::Io);
}

// AR007 FR-3: the planner consumes the curve only when the user did not
// override the budget, and records where the number came from.
TEST_CASE(planner_budget_source_and_curve) {
    const std::string path = write_temp("curve_planner.txt",
                                        "model_sha256 = abc123\n"
                                        "4000 = 0.15\n"
                                        "8000 = 0.14\n"
                                        "16000 = 0.12\n");
    PlannerInput in;
    in.n_tokens = 6000;
    in.n_layer = 28;
    in.mode = Mode::Balanced;

    Config cfg;
    Plan plan;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_EQ(plan.budget_source, "default");
    CHECK_EQ(plan.budget, 0.5);

    cfg.budget_curve = path;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_EQ(plan.budget_source, "model");
    CHECK_EQ(plan.budget, 0.14);
    CHECK(std::find(plan.reasons.begin(), plan.reasons.end(), "budget-curve") !=
          plan.reasons.end());

    // Explicit budget always wins over the curve.
    cfg.budget = 0.3;
    cfg.budget_overridden = true;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_EQ(plan.budget_source, "override");
    CHECK_EQ(plan.budget, 0.3);

    // Missing curve file degrades to the default with a reason, no failure.
    Config broken;
    broken.budget_curve = "/tmp/edge_test_missing_curve.txt";
    CHECK_OK(Planner::make_plan(in, broken, &plan));
    CHECK_EQ(plan.budget_source, "default");
    CHECK(std::find(plan.reasons.begin(), plan.reasons.end(), "budget-curve-unavailable") !=
          plan.reasons.end());
    std::remove(path.c_str());
}

// AR007 FR-3: long prompts keep the legacy 4096-token clamp for explicit
// budgets, while a curve-derived budget may use the measured 8192 ceiling.
TEST_CASE(planner_long_prompt_clamp_semantics) {
    const std::string path = write_temp("curve_clamp.txt",
                                        "model_sha256 = abc123\n"
                                        "32000 = 0.26\n");
    PlannerInput in;
    in.n_tokens = 31000;
    in.n_layer = 28;
    in.mode = Mode::Balanced;

    Config cfg;
    Plan plan;
    cfg.budget = 0.5;
    cfg.budget_overridden = true;
    CHECK_OK(Planner::make_plan(in, cfg, &plan));
    CHECK_EQ(plan.budget_source, "clamp");
    CHECK(plan.budget < 0.14);  // 4096 / 31000

    Config curve_cfg;
    curve_cfg.budget_curve = path;
    CHECK_OK(Planner::make_plan(in, curve_cfg, &plan));
    CHECK_EQ(plan.budget_source, "model");
    CHECK(plan.budget > 0.25);  // the curve may admit up to 8192 tokens
    std::remove(path.c_str());
}
