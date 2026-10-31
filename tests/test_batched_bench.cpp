#include "test_framework.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "edge_prefill/benchmark.h"

using namespace edge_prefill;

namespace {

std::string bench_batch_model() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

bool bench_batch_dataset(const std::filesystem::path& dir, uint32_t sentences,
                         uint32_t cases) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string text = "The vault passcode is MAGENTA-FALCON-7000. ";
    for (uint32_t i = 0; i < sentences; ++i) {
        text += "The quick brown fox jumps over the lazy dog. Rain falls on the old roof. ";
    }
    {
        std::ofstream ctx((dir / "ctx.txt").string(), std::ios::binary);
        if (!ctx) return false;
        ctx << text;
    }
    std::ofstream index((dir / "index.tsv").string(), std::ios::binary);
    if (!index) return false;
    index << "id\ttask\tcontext_path\tquestion\tanswer\n";
    for (uint32_t i = 0; i < cases; ++i) {
        index << "niah-0" << (i + 1) << "\tniah\tctx.txt\tWhat is the vault passcode?\t"
              << "MAGENTA-FALCON-7000\n";
    }
    return true;
}

} // namespace

// AR017 FR-2/FR-3: batching must not change a single byte of an answer. The
// sample carries a hash of the generated text, so this compares answers exactly
// rather than through f1.
TEST_CASE(batched_bench_matches_serial) {
    const std::string model = bench_batch_model();
    if (model.empty()) {
        std::printf("SKIP batched_bench_matches_serial (no EDGE_PREFILL_TEST_MODEL)\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() / "edge_prefill_batch_test";
    CHECK(bench_batch_dataset(dir, 400, 2));

    auto run = [&](uint32_t k, std::vector<Sample>* out) -> Status {
        BenchmarkConfig bcfg;
        bcfg.model = model;
        bcfg.data_dir = dir.string();
        bcfg.modes = {Mode::Latency};
        // a small budget keeps the admitted span inside the batch KV share
        bcfg.budgets = {0.05};
        bcfg.lengths = {0};
        bcfg.repeats = 1;
        bcfg.warmup = 0;
        bcfg.max_new_tokens = 8;

        bcfg.n_threads = 4;
        bcfg.n_threads_batch = 4;
        bcfg.n_ubatch = 128;
        bcfg.decode_batch_size = k;
        // compacted positions keep the admitted span near the budget, which is
        // what makes the case fit a per-sequence KV share
        bcfg.position_compaction = 1;
        bcfg.decode_batch_ctx = k * 3072;
        bcfg.plan_source = "cli";
        bcfg.out_dir = dir.string();
        std::vector<std::string> warnings;
        return BenchmarkRunner().run(bcfg, out, &warnings);
    };

    std::vector<Sample> serial;
    CHECK_OK(run(1, &serial));
    std::vector<Sample> batched;
    CHECK_OK(run(2, &batched));
    CHECK_EQ(serial.size(), 2u);
    CHECK_EQ(batched.size(), 2u);
    if (serial.size() != 2u || batched.size() != 2u) return;

    CHECK_EQ(serial[0].generated_hash, batched[0].generated_hash);
    CHECK_EQ(serial[1].generated_hash, batched[1].generated_hash);
    CHECK_NEAR(serial[0].quality, batched[0].quality, 1e-9);
    CHECK_NEAR(serial[1].quality, batched[1].quality, 1e-9);
    // The batched arm must report that it really batched, and the serial arm
    // must keep saying 1 (no silent switch).
    CHECK_EQ(serial[0].batch_size, 1u);
    CHECK_EQ(batched[0].batch_size, 2u);
    CHECK(batched[0].batch_steps > 0u);
    CHECK(batched[0].seq_id != batched[1].seq_id);
}
