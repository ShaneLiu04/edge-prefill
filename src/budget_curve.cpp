#include "edge_prefill/budget_curve.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace edge_prefill {
namespace {

std::string trim(const std::string& text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

bool parse_u32(const std::string& text, uint32_t* out) {
    if (text.empty()) return false;
    size_t pos = 0;
    try {
        const unsigned long value = std::stoul(text, &pos, 10);
        if (pos != text.size() || value == 0) return false;
        *out = static_cast<uint32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double_strict(const std::string& text, double* out) {
    if (text.empty()) return false;
    size_t pos = 0;
    try {
        const double value = std::stod(text, &pos);
        if (pos != text.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

bool BudgetCurve::recommend(uint32_t n_tokens, double* budget) const {
    if (!budget || points.empty()) return false;
    // The curve is measured per dataset length, not on a dense grid, so a
    // prompt usually lands between two points. Take the next longer point:
    // its measurement already covers the shorter prompt inside its context,
    // while the previous point was measured without that extra content.
    // Below the first point and above the last we clamp to the nearest end.
    const auto* best = &points.back();
    for (const auto& point : points) {
        if (point.first >= n_tokens) {
            best = &point;
            break;
        }
    }
    *budget = best->second;
    return true;
}

bool BudgetCurve::matches_model(const std::string& sha256) const {
    if (model_sha256.empty() || sha256.empty()) return false;
    return model_sha256 == sha256;
}

Status load_budget_curve(const std::string& path, BudgetCurve* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null budget curve output");
    if (path.empty()) return Status::error(ErrorCode::InvalidArgument, "empty budget curve path");
    std::ifstream in(path);
    if (!in) {
        return Status::error(ErrorCode::Io, "cannot read budget curve: " + path);
    }
    BudgetCurve curve;
    std::string line;
    uint32_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string text = trim(line);
        if (text.empty() || text[0] == '#') continue;
        const size_t eq = text.find('=');
        if (eq == std::string::npos) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "budget curve line " + std::to_string(line_no) +
                                     " is not 'key = value'");
        }
        const std::string key = trim(text.substr(0, eq));
        const std::string value = trim(text.substr(eq + 1));
        if (key.empty()) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "budget curve line " + std::to_string(line_no) +
                                     " has an empty key");
        }
        if (key == "model_sha256") {
            curve.model_sha256 = value;
        } else if (key == "quality_floor") {
            double parsed = 0.0;
            if (!parse_double_strict(value, &parsed) || parsed <= 0.0 || parsed > 1.0) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "budget curve quality_floor must be in (0,1]");
            }
            curve.quality_floor = parsed;
        } else {
            uint32_t length = 0;
            double budget = 0.0;
            if (!parse_u32(key, &length)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "budget curve key '" + key + "' is not a prompt length");
            }
            if (!parse_double_strict(value, &budget) || budget <= 0.0 || budget > 1.0) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "budget curve budget for length " + key +
                                         " must be in (0,1]");
            }
            curve.points.emplace_back(length, budget);
        }
    }
    if (curve.points.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "budget curve has no length points: " + path);
    }
    std::sort(curve.points.begin(), curve.points.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    *out = std::move(curve);
    return Status::success();
}

} // namespace edge_prefill
