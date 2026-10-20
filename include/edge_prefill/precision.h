#pragma once

#include <string>

#include "edge_prefill/backend.h"
#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

class PrecisionController {
public:
    static Status apply(ContextOptions* opts, const PhaseKVPlan& plan,
                        std::string* reason);
};

} // namespace edge_prefill
