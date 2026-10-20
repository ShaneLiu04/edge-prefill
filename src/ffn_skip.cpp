#include "edge_prefill/ffn_skip.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <vector>

namespace edge_prefill {
namespace {

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : text) {
        if (c == sep) {
            parts.push_back(current);
            current.clear();
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            current.push_back(c);
        }
    }
    parts.push_back(current);
    return parts;
}

bool parse_int(const std::string& text, int32_t* out) {
    try {
        size_t pos = 0;
        const long value = std::stol(text, &pos, 10);
        if (pos != text.size()) return false;
        *out = static_cast<int32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

Status FfnSkipController::auto_mask(int32_t n_layer, double ratio, uint64_t* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null mask output");
    if (n_layer <= 0 || n_layer > 64) {
        return Status::error(ErrorCode::InvalidArgument, "n_layer must be in (0,64]");
    }
    if (!(ratio >= 0.0 && ratio <= 1.0)) {
        return Status::error(ErrorCode::InvalidArgument, "ratio must be in [0,1]");
    }
    const int32_t count = static_cast<int32_t>(std::llround(ratio * n_layer));
    uint64_t mask = 0;
    if (count > 0) {
        const double step = static_cast<double>(n_layer) / count;
        for (int32_t i = 0; i < count; ++i) {
            const int32_t layer = std::min(n_layer - 1, static_cast<int32_t>(i * step));
            mask |= (uint64_t(1) << layer);
        }
    }
    *out = mask;
    return Status::success();
}

Status FfnSkipController::parse(const std::string& spec, int32_t n_layer, uint64_t* out,
                                std::string* warning) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null mask output");
    if (n_layer <= 0 || n_layer > 64) {
        return Status::error(ErrorCode::InvalidArgument, "n_layer must be in (0,64]");
    }
    if (spec.empty()) return Status::error(ErrorCode::InvalidArgument, "empty mask spec");
    if (warning) warning->clear();

    if (!spec.empty() && spec.back() == '%') {
        try {
            const double ratio = std::stod(spec.substr(0, spec.size() - 1)) / 100.0;
            return auto_mask(n_layer, ratio, out);
        } catch (...) {
            return Status::error(ErrorCode::InvalidArgument, "invalid percentage mask");
        }
    }

    uint64_t mask = 0;
    for (const auto& part : split(spec, ',')) {
        if (part.empty()) {
            return Status::error(ErrorCode::InvalidArgument, "empty mask component");
        }
        const auto dash = part.find('-');
        int32_t begin = 0;
        int32_t end = 0;
        if (dash == std::string::npos) {
            if (!parse_int(part, &begin)) {
                return Status::error(ErrorCode::InvalidArgument, "invalid mask token: " + part);
            }
            end = begin;
        } else {
            if (!parse_int(part.substr(0, dash), &begin) ||
                !parse_int(part.substr(dash + 1), &end)) {
                return Status::error(ErrorCode::InvalidArgument, "invalid mask range: " + part);
            }
            if (end < begin) std::swap(begin, end);
        }
        if (begin < 0) {
            return Status::error(ErrorCode::InvalidArgument, "negative layer index");
        }
        for (int32_t layer = begin; layer <= end; ++layer) {
            if (layer >= n_layer) {
                if (warning) *warning = "mask-bit-out-of-range";
                continue;
            }
            if (layer >= 64) {
                if (warning) *warning = "mask-bit-out-of-range";
                continue;
            }
            mask |= (uint64_t(1) << layer);
        }
    }
    *out = mask;
    return Status::success();
}

Status FfnSkipController::validate(const LlamaBackend& backend, uint64_t requested,
                                   uint64_t* effective, std::string* warning) {
    if (!effective) return Status::error(ErrorCode::InvalidArgument, "null effective mask");
    if (warning) warning->clear();
#if !defined(EDGE_PREFILL_HAS_FFN_SKIP)
    return Status::error(ErrorCode::Unsupported, "llama.cpp patch is not available");
#else
    const std::string arch = backend.arch();
    if (arch != "llama" && arch != "qwen2" && arch != "qwen3") {
        return Status::error(ErrorCode::Unsupported, "FFN skip is not validated for arch: " + arch);
    }
    const int32_t n_layer = backend.n_layer();
    if (n_layer <= 0) {
        return Status::error(ErrorCode::Unsupported, "invalid n_layer");
    }
    if (n_layer > 64) {
        return Status::error(ErrorCode::Unsupported, "n_layer > 64 is not supported");
    }
    const uint64_t all = (uint64_t(1) << n_layer) - 1;
    *effective = requested & all;
    if (*effective != requested && warning) {
        *warning = "mask-bit-out-of-range";
    }
    return Status::success();
#endif
}

} // namespace edge_prefill
