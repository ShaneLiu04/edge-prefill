#include "edge_prefill/exec_plan.h"

#include "test_framework.h"

using namespace edge_prefill;

TEST_CASE(exec_plan_merge_applies_field_level_precedence) {
    ExecPlan conf;
    conf.has_n_threads = true;
    conf.n_threads = 8;
    conf.has_n_gpu_layers = true;
    conf.n_gpu_layers = 0;
    conf.source = PlanSource::Conf;

    ExecPlan cli;
    cli.has_n_threads = true;
    cli.n_threads = 12;
    cli.source = PlanSource::Cli;

    const ExecPlan merged = ExecPlan::merge(conf, cli);
    // CLI wins on the fields it sets ...
    CHECK(merged.has_n_threads);
    CHECK_EQ(merged.n_threads, 12);
    // ... and the conf value survives on the fields it does not set.
    CHECK(merged.has_n_gpu_layers);
    CHECK_EQ(merged.n_gpu_layers, 0);
    CHECK_EQ(static_cast<int>(merged.source), static_cast<int>(PlanSource::Cli));
}

TEST_CASE(exec_plan_merge_keeps_all_layers_sentinel) {
    ExecPlan probe;
    probe.has_n_gpu_layers = true;
    probe.n_gpu_layers = -1;
    probe.source = PlanSource::Probe;
    const ExecPlan merged = ExecPlan::merge(ExecPlan{}, probe);
    CHECK_EQ(merged.n_gpu_layers, -1);
}

TEST_CASE(exec_plan_validate_rejects_bad_geometry) {
    ExecPlan plan;
    plan.has_n_ubatch = true;
    plan.n_ubatch = 0;
    CHECK(!plan.validate().ok());

    ExecPlan bad_batch;
    bad_batch.has_n_ubatch = true;
    bad_batch.n_ubatch = 1024;
    bad_batch.has_n_batch = true;
    bad_batch.n_batch = 512;
    CHECK(!bad_batch.validate().ok());

    ExecPlan negative_threads;
    negative_threads.has_n_threads = true;
    negative_threads.n_threads = -2;
    CHECK(!negative_threads.validate().ok());

    CHECK(ExecPlan{}.validate().ok());
}

TEST_CASE(exec_plan_to_conf_roundtrips_only_set_fields) {
    ExecPlan plan;
    plan.has_n_threads = true;
    plan.n_threads = 12;
    plan.has_n_ubatch = true;
    plan.n_ubatch = 256;
    const std::string conf = plan.to_conf();
    CHECK(conf.find("n_threads = 12") != std::string::npos);
    CHECK(conf.find("n_ubatch = 256") != std::string::npos);
    CHECK(conf.find("n_gpu_layers") == std::string::npos);

    ExecPlan empty;
    CHECK(empty.to_conf().empty());
}

TEST_CASE(placement_report_requires_gpu_when_requested) {
    PlacementReport report;
    report.requested_layers = 99;
    report.build_supports_gpu = true;
    report.gpu_device_present = true;
    report.offloaded_layers = 29;
    report.total_layers = 29;
    CHECK(report.resolve(false).ok());
    CHECK_EQ(static_cast<int>(report.state), static_cast<int>(DeviceState::Gpu));

    PlacementReport silent;
    silent.requested_layers = 99;
    silent.build_supports_gpu = true;
    silent.gpu_device_present = true;
    silent.offloaded_layers = 0;
    // Zero layers offloaded while 99 were requested is a hard failure: this is
    // exactly the silent-degradation bug that produced the AR015 mislabel.
    const Status st = silent.resolve(false);
    CHECK(!st.ok());
    CHECK_EQ(static_cast<int>(st.code), static_cast<int>(ErrorCode::Unavailable));

    PlacementReport fallback;
    fallback.requested_layers = 99;
    fallback.build_supports_gpu = true;
    fallback.gpu_device_present = true;
    fallback.offloaded_layers = 0;
    CHECK(fallback.resolve(true).ok());
    CHECK_EQ(static_cast<int>(fallback.state), static_cast<int>(DeviceState::CpuFallback));
}

TEST_CASE(placement_report_marks_unverified_instead_of_guessing) {
    PlacementReport report;
    report.requested_layers = 99;
    report.build_supports_gpu = true;
    report.gpu_device_present = true;
    report.offloaded_layers = -1; // llama.cpp did not report
    CHECK(report.resolve(false).ok());
    CHECK_EQ(static_cast<int>(report.state), static_cast<int>(DeviceState::Unverified));
}

TEST_CASE(placement_report_cpu_request_is_always_cpu) {
    PlacementReport report;
    report.requested_layers = 0;
    report.offloaded_layers = 0;
    CHECK(report.resolve(false).ok());
    CHECK_EQ(static_cast<int>(report.state), static_cast<int>(DeviceState::Cpu));
    CHECK_EQ(std::string(device_state_name(report.state)), std::string("cpu"));
}

TEST_CASE(placement_report_no_gpu_build_is_unsupported) {
    PlacementReport report;
    report.requested_layers = 8;
    report.build_supports_gpu = false;
    report.gpu_device_present = false;
    const Status st = report.resolve(false);
    CHECK(!st.ok());
    CHECK_EQ(static_cast<int>(st.code), static_cast<int>(ErrorCode::Unsupported));
}
