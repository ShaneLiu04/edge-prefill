#pragma once

#include <string>

namespace edge_prefill {

enum class ErrorCode : int {
    Ok = 0,
    InvalidArgument = 2,
    ModelLoad = 3,
    Tokenize = 4,
    ContextInit = 5,
    Prefill = 6,
    Decode = 7,
    Cache = 8,
    Unsupported = 9,
    OutOfMemory = 10,
    Io = 11,
    Internal = 12,
    // AR016 FR-1: the requested execution plan could not be honoured (e.g. GPU
    // layers requested but none offloaded). Never silently downgraded.
    Unavailable = 13,
};

struct Status {
    ErrorCode code = ErrorCode::Ok;
    std::string message;

    bool ok() const noexcept { return code == ErrorCode::Ok; }
    static Status success() { return {}; }
    static Status error(ErrorCode code, std::string message) {
        return Status{code, std::move(message)};
    }
};

const char* error_code_name(ErrorCode code);

} // namespace edge_prefill
