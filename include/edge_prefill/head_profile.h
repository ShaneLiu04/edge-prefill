#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "edge_prefill/status.h"

namespace edge_prefill {

// Parameters of the offline head-classification probe (classify-heads).
// Mirrored into head_profile.json so a profile records how it was produced.
struct HeadProbeParams {
    uint32_t block_len = 32;
    uint32_t repeats = 8;
    uint32_t sink = 4;
    uint32_t recent = 8;
    uint32_t probe_k = 8;
    uint64_t seed = 42;
};

// Per-(layer, head) sensitivity measured by the offline probe. `retrieval`
// marks heads that must stay dense during adaptive sparse prefill.
struct HeadProfileEntry {
    int32_t layer = 0;
    int32_t head = 0;
    double echo_score = 0.0;
    double induction_score = 0.0;
    bool retrieval = false;
};

struct HeadProfile {
    std::string model_sha;
    int32_t n_layer = 0;
    int32_t n_head = 0;
    HeadProbeParams probe;
    std::vector<HeadProfileEntry> heads;
};

// Loads a head profile from a JSON file. Missing/unreadable file -> Io;
// malformed JSON or structurally invalid profile -> InvalidArgument with the
// offending field name (never silently ignored).
Status load_head_profile(const std::string& path, HeadProfile* out);

// Checks that a profile matches the loaded model geometry. Mismatched
// n_layer/n_head or out-of-range entries -> InvalidArgument; the caller must
// not use a rejected profile.
Status validate_head_profile(const HeadProfile& profile, int32_t n_layer,
                             int32_t n_head);

} // namespace edge_prefill
