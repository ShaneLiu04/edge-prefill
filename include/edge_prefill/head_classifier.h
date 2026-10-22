#pragma once

#include <string>

#include "edge_prefill/backend.h"
#include "edge_prefill/head_profile.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

// Offline ablation-based head classifier (the classify-heads subcommand).
//
// For every (layer, head) it runs one real forward of a fixed-seed repeated
// random-token probe sequence with only that head restricted to the
// sink+recent row pool (all other heads dense), and measures the top-1 flip
// rate versus a dense baseline at two kinds of probe positions:
//   - echo positions: inside repeat copies, where the next token equals the
//     corresponding token of the previous copy;
//   - induction positions: the last token of each copy, which predicts the
//     first token of the next copy.
// A head whose max(echo, induction) flip rate exceeds the threshold is
// classified as a retrieval head: it cannot tolerate sink+recent-only
// attention, so adaptive sparse prefill must keep it dense.
class HeadClassifier {
public:
    explicit HeadClassifier(double retrieval_threshold = 0.10);

    Status classify(LlamaBackend* backend, const HeadProbeParams& params,
                    const std::string& model_sha, HeadProfile* out);

private:
    double retrieval_threshold_;
};

} // namespace edge_prefill
