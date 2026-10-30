#include "edge_prefill/exec_plan.h"

#include <cstdio>

namespace edge_prefill {

const char* plan_source_name(PlanSource source) {
    switch (source) {
        case PlanSource::Default: return "default";
        case PlanSource::Probe: return "probe";
        case PlanSource::Conf: return "conf";
        case PlanSource::Cli: return "cli";
    }
    return "unknown";
}

const char* device_state_name(DeviceState state) {
    switch (state) {
        case DeviceState::Cpu: return "cpu";
        case DeviceState::Gpu: return "gpu";
        case DeviceState::CpuFallback: return "cpu(fallback)";
        case DeviceState::Unverified: return "unverified";
    }
    return "unknown";
}

ExecPlan ExecPlan::merge(const ExecPlan& lower, const ExecPlan& higher) {
    ExecPlan out = lower;
    if (higher.has_n_gpu_layers) {
        out.has_n_gpu_layers = true;
        out.n_gpu_layers = higher.n_gpu_layers;
    }
    if (higher.has_n_threads) {
        out.has_n_threads = true;
        out.n_threads = higher.n_threads;
    }
    if (higher.has_n_threads_batch) {
        out.has_n_threads_batch = true;
        out.n_threads_batch = higher.n_threads_batch;
    }
    if (higher.has_n_ubatch) {
        out.has_n_ubatch = true;
        out.n_ubatch = higher.n_ubatch;
    }
    if (higher.has_n_batch) {
        out.has_n_batch = true;
        out.n_batch = higher.n_batch;
    }
    if (higher.has_flash_attn) {
        out.has_flash_attn = true;
        out.flash_attn = higher.flash_attn;
    }
    out.source = higher.source;
    return out;
}

Status ExecPlan::validate() const {
    if (has_n_threads && n_threads < 0) {
        return Status::error(ErrorCode::InvalidArgument, "plan: negative n_threads");
    }
    if (has_n_threads_batch && n_threads_batch < 0) {
        return Status::error(ErrorCode::InvalidArgument, "plan: negative n_threads_batch");
    }
    if (n_ubatch == 0) {
        return Status::error(ErrorCode::InvalidArgument, "plan: n_ubatch must be > 0");
    }
    if (n_batch < n_ubatch) {
        return Status::error(ErrorCode::InvalidArgument, "plan: n_batch < n_ubatch");
    }
    return Status::success();
}

std::string ExecPlan::to_conf() const {
    std::string out;
    char buf[128];
    if (has_n_gpu_layers) {
        std::snprintf(buf, sizeof(buf), "n_gpu_layers = %d\n", n_gpu_layers);
        out += buf;
    }
    if (has_n_threads) {
        std::snprintf(buf, sizeof(buf), "n_threads = %d\n", n_threads);
        out += buf;
    }
    if (has_n_threads_batch) {
        std::snprintf(buf, sizeof(buf), "n_threads_batch = %d\n", n_threads_batch);
        out += buf;
    }
    if (has_n_ubatch) {
        std::snprintf(buf, sizeof(buf), "n_ubatch = %u\n", n_ubatch);
        out += buf;
    }
    if (has_n_batch) {
        std::snprintf(buf, sizeof(buf), "n_batch = %u\n", n_batch);
        out += buf;
    }
    if (has_flash_attn) {
        std::snprintf(buf, sizeof(buf), "flash_attn = %d\n", flash_attn ? 1 : 0);
        out += buf;
    }
    return out;
}

Status PlacementReport::resolve(bool allow_cpu_fallback) {
    state = DeviceState::Cpu;
    if (requested_layers == 0) {
        return Status::success();
    }
    if (!build_supports_gpu || !gpu_device_present) {
        if (allow_cpu_fallback) {
            state = DeviceState::CpuFallback;
            return Status::success();
        }
        return Status::error(ErrorCode::Unsupported,
                             "plan requests " + std::to_string(requested_layers) +
                                 " GPU layers but this build/device has no GPU offload");
    }
    if (offloaded_layers < 0) {
        // llama.cpp did not report the placement; do not pretend it worked.
        state = DeviceState::Unverified;
        return Status::success();
    }
    if (offloaded_layers > 0) {
        state = DeviceState::Gpu;
        return Status::success();
    }
    if (allow_cpu_fallback) {
        state = DeviceState::CpuFallback;
        return Status::success();
    }
    return Status::error(ErrorCode::Unavailable,
                         "plan requests " + std::to_string(requested_layers) +
                             " GPU layers but 0 were offloaded");
}

} // namespace edge_prefill
