#include "edge_prefill/log.h"

#include <iostream>

namespace edge_prefill {
namespace {
LogLevel g_level = LogLevel::Warn;

void emit(const char* level, const std::string& message) {
    std::cerr << "[edge-prefill][" << level << "] " << message << '\n';
}
} // namespace

void set_log_level(LogLevel level) { g_level = level; }
LogLevel log_level() { return g_level; }

void log_error(const std::string& message) { emit("error", message); }
void log_warn(const std::string& message) {
    if (g_level >= LogLevel::Warn) emit("warn", message);
}
void log_info(const std::string& message) {
    if (g_level >= LogLevel::Info) emit("info", message);
}
void log_debug(const std::string& message) {
    if (g_level >= LogLevel::Debug) emit("debug", message);
}

} // namespace edge_prefill
