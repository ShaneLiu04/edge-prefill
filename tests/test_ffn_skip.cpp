#include "test_framework.h"

#include <string>

#include "edge_prefill/ffn_skip.h"

using namespace edge_prefill;

TEST_CASE(ffn_auto_mask_ratio) {
    uint64_t mask = 0;
    CHECK_OK(FfnSkipController::auto_mask(28, 0.25, &mask));
    uint64_t expected = 0;
    for (int layer : {0, 4, 8, 12, 16, 20, 24}) expected |= (uint64_t(1) << layer);
    CHECK_EQ(mask, expected);

    CHECK_OK(FfnSkipController::auto_mask(28, 0.0, &mask));
    CHECK_EQ(mask, 0u);

    CHECK_OK(FfnSkipController::auto_mask(28, 1.0, &mask));
    CHECK_EQ(mask, (uint64_t(1) << 28) - 1);
}

TEST_CASE(ffn_parse_list_and_range) {
    uint64_t mask = 0;
    std::string warning;
    CHECK_OK(FfnSkipController::parse("0,3-5", 28, &mask, &warning));
    CHECK(warning.empty());
    CHECK_EQ(mask, (uint64_t(1) << 0) | (uint64_t(1) << 3) | (uint64_t(1) << 4) |
                       (uint64_t(1) << 5));

    CHECK_OK(FfnSkipController::parse(" 2 , 2 ", 28, &mask, &warning));
    CHECK_EQ(mask, (uint64_t(1) << 2));
}

TEST_CASE(ffn_parse_percentage) {
    uint64_t mask = 0;
    std::string warning;
    CHECK_OK(FfnSkipController::parse("50%", 28, &mask, &warning));
    uint64_t expected = 0;
    for (int i = 0; i < 14; ++i) expected |= (uint64_t(1) << (i * 2));
    CHECK_EQ(mask, expected);
}

TEST_CASE(ffn_parse_out_of_range_warns) {
    uint64_t mask = 0;
    std::string warning;
    CHECK_OK(FfnSkipController::parse("0-100", 28, &mask, &warning));
    CHECK_EQ(mask, (uint64_t(1) << 28) - 1);
    CHECK_EQ(warning, std::string("mask-bit-out-of-range"));
}

TEST_CASE(ffn_parse_invalid) {
    uint64_t mask = 0;
    CHECK_ERR(FfnSkipController::parse("", 28, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("abc", 28, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("1,,2", 28, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("-1", 28, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("x%", 28, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("0", 0, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("0", 65, &mask, nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::parse("0", 28, nullptr, nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(ffn_auto_mask_invalid) {
    uint64_t mask = 0;
    CHECK_ERR(FfnSkipController::auto_mask(0, 0.25, &mask), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::auto_mask(65, 0.25, &mask), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::auto_mask(28, -0.1, &mask), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::auto_mask(28, 1.1, &mask), ErrorCode::InvalidArgument);
    CHECK_ERR(FfnSkipController::auto_mask(28, 0.25, nullptr), ErrorCode::InvalidArgument);
}
