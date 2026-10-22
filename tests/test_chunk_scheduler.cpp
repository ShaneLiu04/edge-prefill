#include "test_framework.h"

#include <vector>

#include "edge_prefill/chunk_scheduler.h"

using namespace edge_prefill;

TEST_CASE(chunk_exact_multiple) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(8192, 512, 512, &ranges));
    CHECK_EQ(ranges.size(), 16u);
    CHECK_EQ(ranges.front().begin, 0u);
    CHECK_EQ(ranges.front().end, 512u);
    CHECK_EQ(ranges.back().begin, 7680u);
    CHECK_EQ(ranges.back().end, 8192u);
    for (size_t i = 1; i < ranges.size(); ++i) {
        CHECK_EQ(ranges[i].begin, ranges[i - 1].end);
        CHECK(ranges[i].size() <= 512u);
    }
}

TEST_CASE(chunk_remainder) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(8000, 512, 512, &ranges));
    CHECK_EQ(ranges.size(), 16u);
    CHECK_EQ(ranges.back().size(), 320u);
    CHECK_EQ(ranges.back().end, 8000u);
}

TEST_CASE(chunk_ubatch_smaller_than_chunk) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(1000, 512, 128, &ranges));
    CHECK_EQ(ranges.size(), 8u);
    for (const auto& r : ranges) CHECK(r.size() <= 128u);
    CHECK_EQ(ranges.back().end, 1000u);
}

TEST_CASE(chunk_zero_uses_default) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(1000, 0, 128, &ranges));
    CHECK_EQ(ranges.size(), 8u);
    for (const auto& r : ranges) CHECK(r.size() <= 128u);
}

// AR004 FR-1: chunk_size == 0 means "auto": align to n_ubatch so every
// chunk executes as a single llama_decode ubatch (no 512 fallback).
TEST_CASE(chunk_zero_aligns_ubatch) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(1000, 0, 2048, &ranges));
    CHECK_EQ(ranges.size(), 1u);
    CHECK_EQ(ranges.front().begin, 0u);
    CHECK_EQ(ranges.front().end, 1000u);

    CHECK_OK(ChunkScheduler::plan(5000, 0, 2048, &ranges));
    CHECK_EQ(ranges.size(), 3u);
    CHECK_EQ(ranges[1].size(), 2048u);
    CHECK_EQ(ranges.back().size(), 904u);
    for (size_t i = 1; i < ranges.size(); ++i) {
        CHECK_EQ(ranges[i].begin, ranges[i - 1].end);
    }
}

// An explicit chunk equal to the token count yields one range (the same
// geometry enable_chunking=false must use instead of the 0 sentinel).
TEST_CASE(chunk_equals_n_single_range) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(1000, 1000, 2048, &ranges));
    CHECK_EQ(ranges.size(), 1u);
    CHECK_EQ(ranges.front().begin, 0u);
    CHECK_EQ(ranges.front().end, 1000u);

    // The enable_chunking=false geometry: chunk == n_tokens even when it
    // exceeds n_ubatch must stay a single range (llama_decode splits
    // internally; the pre-AR004 0-sentinel silently chunked at 512).
    CHECK_OK(ChunkScheduler::plan(1000, 1000, 128, &ranges));
    CHECK_EQ(ranges.size(), 1u);
    CHECK_EQ(ranges.front().end, 1000u);
}

TEST_CASE(chunk_single_token) {
    std::vector<Range> ranges;
    CHECK_OK(ChunkScheduler::plan(1, 512, 128, &ranges));
    CHECK_EQ(ranges.size(), 1u);
    CHECK_EQ(ranges.front().begin, 0u);
    CHECK_EQ(ranges.front().end, 1u);
}

TEST_CASE(chunk_invalid_inputs) {
    std::vector<Range> ranges;
    CHECK_ERR(ChunkScheduler::plan(0, 512, 512, &ranges), ErrorCode::InvalidArgument);
    CHECK_ERR(ChunkScheduler::plan(100, 512, 0, &ranges), ErrorCode::InvalidArgument);
    CHECK_ERR(ChunkScheduler::plan(100, 512, 512, nullptr), ErrorCode::InvalidArgument);
}
