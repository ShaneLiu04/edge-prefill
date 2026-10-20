#include "edge_prefill/status.h"

namespace edge_prefill {

const char* error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::Ok: return "Ok";
        case ErrorCode::InvalidArgument: return "InvalidArgument";
        case ErrorCode::ModelLoad: return "ModelLoad";
        case ErrorCode::Tokenize: return "Tokenize";
        case ErrorCode::ContextInit: return "ContextInit";
        case ErrorCode::Prefill: return "Prefill";
        case ErrorCode::Decode: return "Decode";
        case ErrorCode::Cache: return "Cache";
        case ErrorCode::Unsupported: return "Unsupported";
        case ErrorCode::OutOfMemory: return "OutOfMemory";
        case ErrorCode::Io: return "Io";
        case ErrorCode::Internal: return "Internal";
    }
    return "Unknown";
}

} // namespace edge_prefill
