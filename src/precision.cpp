#include "edge_prefill/precision.h"

namespace edge_prefill {

Status PrecisionController::apply(ContextOptions* opts, const PhaseKVPlan& plan,
                                  std::string* reason) {
    if (!opts) return Status::error(ErrorCode::InvalidArgument, "null context options");
    if (reason) reason->clear();
    if (plan.prefill_type != plan.decode_type) {
        if (reason) *reason = "kv-phase-fallback";
        return Status::error(ErrorCode::Unsupported,
                             "prefill/decode KV type mismatch is not supported in this version");
    }
    opts->type_k = plan.prefill_type;
    opts->type_v = plan.prefill_type;
    return Status::success();
}

} // namespace edge_prefill
