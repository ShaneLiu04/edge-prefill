#pragma once

#include <string>

namespace edge_prefill {

// SHA256 of a file, used for model/data fingerprints in the environment
// snapshot, expected-result tables and the AR007 budget-curve guard.
//
// Implementation note: this shells out to `sha256sum` because hashing a
// multi-GB GGUF is IO bound and the coreutils implementation is faster than
// anything we would hand-roll here. Returns "unavailable" (never throws or
// returns an empty string) when the tool or the file is missing, so callers
// can compare against "" and fail closed.
std::string sha256_of_file(const std::string& path);

} // namespace edge_prefill
