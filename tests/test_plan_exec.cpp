#include "test_framework.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "edge_prefill/benchmark.h"
#include "edge_prefill/cost_model.h"
#include "edge_prefill/exec_plan.h"

using namespace edge_prefill;

namespace {

std::string plan_test_model() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

// Builds a tiny on-disk dataset with a single long-ish context so the bench can
// run end to end without the checked-in benchmark data.
bool plan_dataset(const std::filesystem::path& dir, uint32_t sentences) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string text = "The vault passcode is MAGENTA-FALCON-7000. ";
    for (uint32_t i = 0; i < sentences; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    {
        std::ofstream ctx((dir / "ctx.txt").string(), std::ios::binary);
        if (!ctx) return false;
        ctx << text;
    }
    std::ofstream index((dir / "index.tsv").string(), std::ios::binary);
    if (!index) return false;
    index << "id\ttask\tcontext_path\tquestion\tanswer\n";
    index << "niah-01\tniah\tctx.txt\tWhat is the vault passcode?\tMAGENTA-FALCON-7000\n";
    return true;
}

} // namespace

// AR016 FR-1: the bench must honour the plan instead of silently using
// hardware_concurrency() and zero GPU layers. Skipped without a real model.
TEST_CASE(plan_threads_and_placement_are_audited) {
    const std::string model = plan_test_model();
    if (model.empty()) {
        std::printf("SKIP plan_threads_and_placement_are_audited (no EDGE_PREFILL_TEST_MODEL)\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() / "edge_prefill_plan_test";
    CHECK(plan_dataset(dir, 40));

    BenchmarkConfig bcfg;
    bcfg.model = model;
    bcfg.data_dir = dir.string();
    bcfg.modes = {Mode::Latency};
    bcfg.budgets = {0.5};
    bcfg.lengths = {0};
    bcfg.repeats = 1;
    bcfg.warmup = 0;
    bcfg.max_new_tokens = 4;
    bcfg.n_threads = 3;
    bcfg.n_threads_batch = 3;
    bcfg.n_ubatch = 128;
    bcfg.n_gpu_layers = 0;
    bcfg.plan_source = "cli";
    bcfg.out_dir = dir.string();

    std::vector<Sample> samples;
    std::vector<std::string> warnings;
    CHECK_OK(BenchmarkRunner().run(bcfg, &samples, &warnings));
    CHECK_EQ(samples.size(), 1u);
    if (samples.size() != 1u) return;
    // The audit columns must reflect the plan, not the hardware default.
    CHECK_EQ(samples[0].n_threads, 3);
    CHECK_EQ(samples[0].n_threads_batch, 3);
    CHECK_EQ(samples[0].n_ubatch, 128u);
    CHECK_EQ(samples[0].gpu_layers_requested, 0);
    CHECK_EQ(std::string(samples[0].device_state), std::string("cpu"));
    CHECK_EQ(std::string(samples[0].plan_source), std::string("cli"));
}

// AR016 FR-1: asking for GPU layers on a machine that cannot provide them must
// fail loudly. On a CUDA build with a device this either offloads (gpu state)
// or is a hard error; it must never come back as a silent "cpu" success.
TEST_CASE(plan_gpu_request_never_silently_degrades) {
    const std::string model = plan_test_model();
    if (model.empty()) {
        std::printf("SKIP plan_gpu_request_never_silently_degrades (no model)\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() / "edge_prefill_plan_test";
    CHECK(plan_dataset(dir, 20));

    BenchmarkConfig bcfg;
    bcfg.model = model;
    bcfg.data_dir = dir.string();
    bcfg.modes = {Mode::Latency};
    bcfg.budgets = {0.5};
    bcfg.lengths = {0};
    bcfg.repeats = 1;
    bcfg.warmup = 0;
    bcfg.max_new_tokens = 2;
    bcfg.n_threads = 4;
    bcfg.n_gpu_layers = 99;
    bcfg.allow_cpu_fallback = false;
    bcfg.out_dir = dir.string();

    std::vector<Sample> samples;
    std::vector<std::string> warnings;
    const Status st = BenchmarkRunner().run(bcfg, &samples, &warnings);
    if (st.ok()) {
        // A real GPU was used: the audit must say so.
        CHECK_EQ(samples.size(), 1u);
        if (!samples.empty()) {
            CHECK_EQ(samples[0].gpu_layers_requested, 99);
            CHECK(samples[0].gpu_layers_offloaded > 0);
            CHECK_EQ(std::string(samples[0].device_state), std::string("gpu"));
        }
    } else {
        CHECK(st.code == ErrorCode::Unavailable || st.code == ErrorCode::Unsupported);
    }
}

// AR016 FR-3: the SLA planner only shrinks the budget and only when a valid
// cost model says the first selection would overshoot. Model-free check of the
// solver maths through the public cost-model helpers.
TEST_CASE(plan_sla_shape_is_monotone) {
    CostModel model;
    model.intercept_ms = 200.0;
    model.ms_per_token = 5.5;
    model.r2 = 0.99;
    model.n_samples = 5;
    model.valid = true;
    uint32_t prev = 0;
    for (double target : {500.0, 1000.0, 2000.0, 4000.0}) {
        const uint32_t fit = max_tokens_within(model, target, 8u, 4096u, 0.15);
        CHECK(fit >= prev);
        prev = fit;
    }
    // 2000 ms target with 15% safety can afford well over 200 tokens.
    CHECK(max_tokens_within(model, 2000.0, 8u, 4096u, 0.15) > 200u);
}
