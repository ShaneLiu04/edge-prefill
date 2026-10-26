cmake_minimum_required(VERSION 3.20)
project(edge_prefill VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EDGE_PREFILL_BUILD_TESTS "Build unit tests" ON)
option(EDGE_PREFILL_WITH_CUDA "Build llama.cpp with CUDA" OFF)
set(EDGE_PREFILL_LLAMA_DIR "" CACHE PATH "Use an existing llama.cpp source tree")
set(EDGE_PREFILL_LLAMA_COMMIT "3bb63cbcbfb62056c84fe9e521b0f3d756d29c32" CACHE STRING "Pinned llama.cpp commit")

if(NOT EDGE_PREFILL_LLAMA_DIR)
    include(FetchContent)
    FetchContent_Declare(llama
        URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${EDGE_PREFILL_LLAMA_COMMIT}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply_llama_patch.sh
    )
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
    set(GGML_CUDA ${EDGE_PREFILL_WITH_CUDA} CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(llama)
else()
    set(llama_SOURCE_DIR ${EDGE_PREFILL_LLAMA_DIR})
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${EDGE_PREFILL_WITH_CUDA} CACHE BOOL "" FORCE)
    add_subdirectory(${EDGE_PREFILL_LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
endif()

add_library(edge_prefill
    src/status.cpp
    src/log.cpp
    src/json_writer.cpp
    src/config.cpp
    src/backend.cpp
    src/selector.cpp
    src/draft_scorer.cpp
        src/embedding_scorer.cpp
        src/semantic_gate.cpp
    src/attention_sparsity.cpp
    src/prefix_cache.cpp
    src/chunk_scheduler.cpp
    src/ffn_skip.cpp
    src/precision.cpp
    src/head_profile.cpp
    src/head_classifier.cpp
    src/planner.cpp
    src/budget_curve.cpp
    src/portfolio.cpp
    src/hash_util.cpp
    src/sparse_prefill.cpp
    src/metrics.cpp
    src/benchmark.cpp
)

target_include_directories(edge_prefill
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include
)
target_link_libraries(edge_prefill PUBLIC llama ggml)
target_compile_definitions(edge_prefill PUBLIC
    EDGE_PREFILL_HAS_FFN_SKIP=1
    EDGE_PREFILL_LLAMA_COMMIT="${EDGE_PREFILL_LLAMA_COMMIT}"
)
if(MSVC)
    target_compile_options(edge_prefill PRIVATE /W4 /WX)
else()
    target_compile_options(edge_prefill PRIVATE -Wall -Wextra -Werror)
endif()

add_executable(edge-prefill src/main.cpp)
target_link_libraries(edge-prefill PRIVATE edge_prefill)
target_include_directories(edge-prefill PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include)

if(EDGE_PREFILL_BUILD_TESTS)
    enable_testing()
    add_executable(edge_prefill_tests
        tests/test_main.cpp
        tests/test_selector.cpp
        tests/test_draft_scorer.cpp
        tests/test_attention_sparsity.cpp
        tests/test_prefix_cache.cpp
        tests/test_chunk_scheduler.cpp
        tests/test_planner.cpp
        tests/test_budget_curve.cpp
        tests/test_portfolio.cpp
        tests/test_ffn_skip.cpp
        tests/test_head_profile.cpp
        tests/test_head_classifier.cpp
        tests/test_integration.cpp
        tests/test_sparse_attn_exec.cpp
        tests/test_embedding_scorer.cpp
    )
    target_link_libraries(edge_prefill_tests PRIVATE edge_prefill)
    target_include_directories(edge_prefill_tests PRIVATE ${llama_SOURCE_DIR}/include ${llama_SOURCE_DIR}/ggml/include)
    add_test(NAME edge_prefill_tests COMMAND edge_prefill_tests)
endif()
