#pragma once

#include <string>

namespace edge_prefill {

enum class LogLevel { Error = 0, Warn = 1, Info = 2, Debug = 3 };

void set_log_level(LogLevel level);
LogLevel log_level();
void log_error(const std::string& message);
void log_warn(const std::string& message);
void log_info(const std::string& message);
void log_debug(const std::string& message);

} // namespace edge_prefill
