#pragma once

#include <cstdint>
#include <string>

#include "edge_prefill/status.h"

namespace edge_prefill {

// Where a field value came from. Higher priority wins in ExecPlan::merge.
enum class PlanSource : int {
    Default = 0,
    Probe = 1,
    Conf = 2,
    Cli = 3,
};

const char* plan_source_name(PlanSource source);

// Whether the requested device placement was actually observed after load.
enum class DeviceState : int {
    Cpu = 0,
    Gpu = 1,
    CpuFallback = 2,
    Unverified = 3,
};

const char* device_state_name(DeviceState state);

// AR016 FR-1: the execution plan is a first-class, auditable object. Each
// field carries its own "explicitly set" flag so merge() can apply
// field-level precedence (CLI > conf > probe > default) instead of
// overwriting a whole struct.
struct ExecPlan {
    bool has_n_gpu_layers = false;
    int32_t n_gpu_layers = 0; // -1 = all layers

    bool has_n_threads = false;
    int32_t n_threads = 0; // 0 = backend auto

    bool has_n_threads_batch = false;
    int32_t n_threads_batch = 0; // 0 = follow n_threads

    bool has_n_ubatch = false;
    uint32_t n_ubatch = 512;

    bool has_n_batch = false;
    uint32_t n_batch = 2048;

    bool has_flash_attn = false;
    bool flash_attn = false;

    PlanSource source = PlanSource::Default;

    static ExecPlan merge(const ExecPlan& lower, const ExecPlan& higher);

    Status validate() const;

    // Renders the plan as `key = value` lines so it can be appended to a run
    // config and re-parsed by the existing config reader (one config syntax
    // for both probe output and hand-written confs).
    std::string to_conf() const;
};

// AR016 FR-1: what the backend observed after loading the model.
struct PlacementReport {
    int32_t requested_layers = 0;
    int32_t offloaded_layers = -1; // -1 = llama.cpp did not report
    int32_t total_layers = 0;
    bool gpu_device_present = false;
    bool build_supports_gpu = false;
    std::string device_name; // e.g. "CUDA0 (Quadro RTX 5000)" or "CPU"
    DeviceState state = DeviceState::Cpu;

    // Decides the state from the request and the observation. Returns
    // ErrorCode::Unavailable when a GPU was requested but nothing was
    // offloaded and fallback is not allowed.
    Status resolve(bool allow_cpu_fallback);
};

} // namespace edge_prefill
