#include "edge_prefill/hash_util.h"

#include <cstdio>
#include <string>

namespace edge_prefill {

std::string sha256_of_file(const std::string& path) {
    if (path.empty()) return "unavailable";
    const std::string command = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "unavailable";
    char buf[128] = {};
    const char* result = fgets(buf, sizeof(buf), pipe);
    pclose(pipe);
    if (!result) return "unavailable";
    std::string text(buf);
    const auto space = text.find(' ');
    return space == std::string::npos ? text : text.substr(0, space);
}

} // namespace edge_prefill
