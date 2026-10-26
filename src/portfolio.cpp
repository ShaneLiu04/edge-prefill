#include "edge_prefill/portfolio.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>

#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

std::string trim(const std::string& text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
    return text.substr(begin, end - begin);
}

bool known_combo(const std::string& combo) {
    return combo == "selection" || combo == "gate-p16" ||
           combo == "gate-p24-local32" || combo == "dense";
}

struct CacheEntry {
    std::filesystem::file_time_type mtime;
    PortfolioTable table;
};

std::mutex g_mutex;
std::vector<std::pair<std::string, CacheEntry>> g_cache;

} // namespace

std::string PortfolioTable::length_bucket(uint32_t n_tokens) {
    if (n_tokens < 5000) return "lt4k";
    if (n_tokens < 12000) return "8k";
    if (n_tokens < 24000) return "16k";
    return "32k";
}

std::string PortfolioTable::discriminability(uint32_t specific_terms) {
    return specific_terms >= 2 ? "high" : "low";
}

std::string portfolio_key(const PortfolioProfile& profile) {
    // "flat" means the prompt has neither a discriminative phrase hit nor
    // concentrated lexical evidence: the selector has nothing to anchor on,
    // which is where the in-graph gate earns its cost.
    if (profile.phrase_blocks == 0 && profile.lexical_concentration < 0.30) {
        return "flat:any";
    }
    return PortfolioTable::discriminability(profile.specific_terms) + ":" +
           PortfolioTable::length_bucket(profile.n_tokens);
}

bool PortfolioTable::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    rows_.clear();
    model_sha256_.clear();
    default_combo_ = "selection";
    path_ = path;

    std::string line;
    uint32_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string text = trim(line);
        if (text.empty() || text[0] == '#') continue;
        const size_t eq = text.find('=');
        if (eq == std::string::npos) {
            log_warn("portfolio: ignoring malformed line " + std::to_string(line_no));
            continue;
        }
        const std::string key = trim(text.substr(0, eq));
        const std::string value = trim(text.substr(eq + 1));
        if (key.empty() || value.empty()) {
            log_warn("portfolio: ignoring empty key/value at line " +
                     std::to_string(line_no));
            continue;
        }
        if (key == "model_sha256") {
            model_sha256_ = value;
            continue;
        }
        if (key == "default") {
            if (!known_combo(value)) {
                log_warn("portfolio: unknown default combo '" + value + "', keeping selection");
                continue;
            }
            default_combo_ = value;
            continue;
        }
        if (!known_combo(value)) {
            log_warn("portfolio: unknown combo '" + value + "' at line " +
                     std::to_string(line_no));
            continue;
        }
        rows_.emplace_back(key, value);
    }
    return !rows_.empty();
}

bool PortfolioTable::matches_model(const std::string& sha256) const {
    if (model_sha256_.empty() || sha256.empty()) return false;
    return model_sha256_ == sha256;
}

bool PortfolioTable::recommend(const PortfolioProfile& profile, PortfolioRow* row) const {
    if (!row || rows_.empty()) return false;
    const std::string key = portfolio_key(profile);
    for (const auto& entry : rows_) {
        if (entry.first == key) {
            row->combo = entry.second;
            row->key = key;
            return true;
        }
    }
    if (profile.phrase_blocks == 0 && profile.lexical_concentration < 0.30) {
        for (const auto& entry : rows_) {
            if (entry.first == "flat:any") {
                row->combo = entry.second;
                row->key = "flat:any";
                return true;
            }
        }
    }
    for (const auto& entry : rows_) {
        if (entry.first == "default") {
            row->combo = entry.second;
            row->key = "default";
            return true;
        }
    }
    row->combo = default_combo_;
    row->key = "default";
    return true;
}

Status load_portfolio_cached(const std::string& path, PortfolioTable* out) {
    if (path.empty() || out == nullptr) {
        return Status::error(ErrorCode::InvalidArgument, "portfolio: empty path");
    }
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) {
        return Status::error(ErrorCode::Io, "portfolio: cannot stat " + path);
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const auto& entry : g_cache) {
            if (entry.first == path && entry.second.mtime == mtime) {
                *out = entry.second.table;
                return Status::success();
            }
        }
    }
    PortfolioTable table;
    if (!table.load(path)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "portfolio: no usable rows in " + path);
    }
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_cache.erase(std::remove_if(g_cache.begin(), g_cache.end(),
                                     [&](const std::pair<std::string, CacheEntry>& e) {
                                         return e.first == path;
                                     }),
                      g_cache.end());
        g_cache.push_back({path, CacheEntry{mtime, table}});
    }
    *out = table;
    return Status::success();
}

} // namespace edge_prefill
