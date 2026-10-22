#include "test_framework.h"

#include <string>

namespace edge_test {

std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

int& failures() {
    static int count = 0;
    return count;
}

} // namespace edge_test

int main(int argc, char** argv) {
    // Optional substring filter (argv[1]) so a single case can be run or
    // timed in isolation; without arguments every case runs as before.
    const std::string filter = argc > 1 ? argv[1] : "";
    for (const auto& c : edge_test::registry()) {
        if (!filter.empty() && std::string(c.name).find(filter) == std::string::npos) {
            continue;
        }
        const int before = edge_test::failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            edge_test::fail(__FILE__, __LINE__, std::string("exception: ") + e.what());
        } catch (...) {
            edge_test::fail(__FILE__, __LINE__, "unknown exception");
        }
        std::cout << (edge_test::failures() == before ? "PASS " : "FAIL ") << c.name << '\n';
    }
    if (edge_test::failures() != 0) {
        std::cerr << edge_test::failures() << " failure(s)\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
