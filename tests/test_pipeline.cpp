#include "test_framework.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "edge_prefill/benchmark.h"

using namespace edge_prefill;

namespace {

std::string pipeline_model() {
    const char* path = std::getenv("EDGE_PREFILL_TEST_MODEL");
    return path ? path : "";
}

bool pipeline_dataset(const std::filesystem::path& dir, uint32_t sentences,
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

// AR018: the pipeline moves the prefilled sequence to another context as KV
// state. If that handoff were lossy the answers would drift, so the only
// acceptable result is the serial arm's answer, byte for byte.
TEST_CASE(pipeline_matches_serial) {
    const std::string model = pipeline_model();
    if (model.empty()) {
        std::printf("SKIP pipeline_matches_serial (no EDGE_PREFILL_TEST_MODEL)\n");
        return;
    }
    const auto dir = std::filesystem::temp_directory_path() / "edge_prefill_pipe_test";
    CHECK(pipeline_dataset(dir, 400, 2));

    auto run = [&](bool pipeline, std::vector<Sample>* out) -> Status {
        BenchmarkConfig bcfg;
        bcfg.model = model;
        bcfg.data_dir = dir.string();
        // Balanced mode keeps the in-graph page gating off, so the prefill is a
        // pure function of the admitted tokens and the two arms must agree
        // bit for bit. With page gating the gating pages depend on the context
        // size, which legitimately differs between the two contexts.
        bcfg.modes = {Mode::Balanced};
        bcfg.budgets = {0.05};
        bcfg.lengths = {0};
        bcfg.repeats = 1;
        bcfg.warmup = 0;
        bcfg.max_new_tokens = 8;
        bcfg.n_threads = 4;
        bcfg.n_threads_batch = 4;
        bcfg.n_ubatch = 128;
        bcfg.position_compaction = 1;
        bcfg.decode_batch_size = pipeline ? 2 : 1;
        // The prefill context must expose the same number of KV streams as the
        // decoder, so it is sized for the batch even though only seq 0 is used.
        bcfg.decode_batch_ctx = 2 * 3072;
        bcfg.pipeline_mode = pipeline;
        bcfg.pipeline_prefill_gpu_layers = 0;
        bcfg.pipeline_prefill_threads = 4;
        bcfg.pipeline_ahead = 2;
        bcfg.plan_source = "cli";
        bcfg.out_dir = dir.string();
        std::vector<std::string> warnings;
        return BenchmarkRunner().run(bcfg, out, &warnings);
    };

    std::vector<Sample> piped;
    CHECK_OK(run(true, &piped));
    std::vector<Sample> serial;
    CHECK_OK(run(false, &serial));

    // The dense arm is disabled in batch/pipeline mode (AR018 FR-4), so only
    // the sparse rows are comparable; match them by case id, not by index.
    std::vector<const Sample*> sparse_serial;
    std::vector<const Sample*> sparse_piped;
    for (const auto& s : serial) if (!s.dense) sparse_serial.push_back(&s);
    for (const auto& s : piped) if (!s.dense) sparse_piped.push_back(&s);
    CHECK_EQ(sparse_serial.size(), 2u);
    CHECK_EQ(sparse_piped.size(), 2u);
    if (sparse_serial.size() != 2u || sparse_piped.size() != 2u) return;

    for (size_t i = 0; i < 2; ++i) {
        CHECK_EQ(sparse_serial[i]->case_id, sparse_piped[i]->case_id);
        CHECK_EQ(sparse_serial[i]->generated_hash, sparse_piped[i]->generated_hash);
        CHECK_NEAR(sparse_serial[i]->quality, sparse_piped[i]->quality, 1e-9);
    }
    // The run really used the pipeline: the audit columns must say so, and the
    // KV must have crossed as state (a silently serial run would report 0).
    CHECK_EQ(piped[0].pipeline_mode, 1);
    CHECK(std::string(piped[0].prefill_device_state) == "cpu");
    CHECK_GT(static_cast<double>(piped[0].handoff_bytes), 0.0);
}
