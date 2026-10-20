#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace edge_test {

struct Case {
    std::string name;
    std::function<void()> fn;
};

std::vector<Case>& registry();
int& failures();

struct Registrar {
    Registrar(const std::string& name, std::function<void()> fn) {
        registry().push_back(Case{name, std::move(fn)});
    }
};

inline void fail(const char* file, int line, const std::string& message) {
    ++failures();
    std::cerr << "FAIL " << file << ":" << line << " " << message << '\n';
}

} // namespace edge_test

#define TEST_CASE(name)                                                        \
    static void test_##name();                                                 \
    static ::edge_test::Registrar registrar_##name(#name, test_##name);        \
    static void test_##name()

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) ::edge_test::fail(__FILE__, __LINE__, "CHECK failed: " #expr); \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        const auto va = (a);                                                   \
        const auto vb = (b);                                                   \
        if (!(va == vb)) {                                                     \
            ::edge_test::fail(__FILE__, __LINE__,                              \
                              std::string("CHECK_EQ failed: ") + #a + " != " + #b); \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        const double va = static_cast<double>(a);                              \
        const double vb = static_cast<double>(b);                              \
        if (std::fabs(va - vb) > (eps)) {                                      \
            ::edge_test::fail(__FILE__, __LINE__,                              \
                              std::string("CHECK_NEAR failed: ") + #a + " vs " + #b); \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr)                                                         \
    do {                                                                       \
        const auto st__ = (expr);                                              \
        if (!st__.ok()) {                                                      \
            ::edge_test::fail(__FILE__, __LINE__,                              \
                              std::string("expected Ok, got ") +                 \
                                  ::edge_prefill::error_code_name(st__.code) +   \
                                  ": " + st__.message);                        \
        }                                                                      \
    } while (0)

#define CHECK_ERR(expr, expected)                                              \
    do {                                                                       \
        const auto st__ = (expr);                                              \
        if (st__.code != (expected)) {                                         \
            ::edge_test::fail(__FILE__, __LINE__,                              \
                              std::string("expected ") +                       \
                                  ::edge_prefill::error_code_name(expected) +    \
                                  ", got " +                                   \
                                  ::edge_prefill::error_code_name(st__.code));   \
        }                                                                      \
    } while (0)
