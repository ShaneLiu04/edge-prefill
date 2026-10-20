#pragma once

#include <cstdint>
#include <string>

#include "edge_prefill/backend.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

class FfnSkipController {
public:
    static Status validate(const LlamaBackend& backend, uint64_t requested,
                           uint64_t* effective, std::string* warning);
    static Status auto_mask(int32_t n_layer, double ratio, uint64_t* out);
    static Status parse(const std::string& spec, int32_t n_layer, uint64_t* out,
                        std::string* warning);
};

} // namespace edge_prefill
