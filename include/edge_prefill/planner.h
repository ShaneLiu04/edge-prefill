#pragma once

#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

class Planner {
public:
    static Status make_plan(const PlannerInput& in, const Config& cfg, Plan* out);
};

} // namespace edge_prefill
