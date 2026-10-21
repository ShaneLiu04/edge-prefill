#include "test_framework.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "edge_prefill/config.h"
#include "edge_prefill/head_profile.h"

using namespace edge_prefill;

namespace {

// Non-default probe values so a parser that silently falls back to the
// HeadProbeParams defaults cannot pass the valid-load case by accident.
const char* kValidProfileJson =
    "{\"model_sha\":\"sha3-abc123\",\"n_layer\":28,\"n_head\":14,"
    "\"probe_params\":{\"block_len\":64,\"repeats\":4,\"sink\":2,\"recent\":6,"
    "\"probe_k\":10,\"seed\":7},"
    "\"heads\":["
    "{\"layer\":0,\"head\":0,\"echo_score\":0.02,\"induction_score\":0.01,"
    "\"class\":\"retrieval\"},"
    "{\"layer\":3,\"head\":5,\"echo_score\":0.20,\"induction_score\":0.05,"
    "\"class\":\"normal\"}"
    "]}";

std::string write_profile_file(const std::string& name, const std::string& content) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / "edge_prefill_ar003_t001_head_profile";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const fs::path path = dir / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    return path.string();
}

HeadProfile load_valid_profile() {
    const std::string path = write_profile_file("valid.json", kValidProfileJson);
    HeadProfile profile;
    CHECK_OK(load_head_profile(path, &profile));
    return profile;
}

bool message_mentions(const Status& st, const std::string& field) {
    return st.message.find(field) != std::string::npos;
}

} // namespace

TEST_CASE(head_profile_load_valid_full) {
    const std::string path = write_profile_file("valid.json", kValidProfileJson);
    HeadProfile profile;
    CHECK_OK(load_head_profile(path, &profile));
    CHECK_EQ(profile.model_sha, std::string("sha3-abc123"));
    CHECK_EQ(profile.n_layer, 28);
    CHECK_EQ(profile.n_head, 14);
    CHECK_EQ(profile.probe.block_len, 64u);
    CHECK_EQ(profile.probe.repeats, 4u);
    CHECK_EQ(profile.probe.sink, 2u);
    CHECK_EQ(profile.probe.recent, 6u);
    CHECK_EQ(profile.probe.probe_k, 10u);
    CHECK_EQ(profile.probe.seed, 7ull);
    CHECK_EQ(profile.heads.size(), 2u);

    CHECK_EQ(profile.heads[0].layer, 0);
    CHECK_EQ(profile.heads[0].head, 0);
    CHECK_NEAR(profile.heads[0].echo_score, 0.02, 1e-9);
    CHECK_NEAR(profile.heads[0].induction_score, 0.01, 1e-9);
    CHECK(profile.heads[0].retrieval);

    CHECK_EQ(profile.heads[1].layer, 3);
    CHECK_EQ(profile.heads[1].head, 5);
    CHECK_NEAR(profile.heads[1].echo_score, 0.20, 1e-9);
    CHECK_NEAR(profile.heads[1].induction_score, 0.05, 1e-9);
    CHECK(!profile.heads[1].retrieval);
}

TEST_CASE(head_profile_load_missing_file) {
    HeadProfile profile;
    const std::string path = write_profile_file("absent.json", "");
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove_all(fs::path(path).parent_path(), ec);
    CHECK_ERR(load_head_profile(path, &profile), ErrorCode::Io);
}

TEST_CASE(head_profile_load_malformed_json) {
    HeadProfile profile;
    // Missing opening quote on the first key.
    CHECK_ERR(load_head_profile(
                  write_profile_file("bad_quote.json",
                                     "{model_sha\":\"x\",\"n_layer\":1,\"n_head\":1,"
                                     "\"probe_params\":{\"block_len\":32,\"repeats\":8,"
                                     "\"sink\":4,\"recent\":8,\"probe_k\":8,\"seed\":42},"
                                     "\"heads\":[]}"),
                  &profile),
              ErrorCode::InvalidArgument);
    // Truncated document (no closing brace).
    CHECK_ERR(load_head_profile(
                  write_profile_file("truncated.json",
                                     "{\"model_sha\":\"x\",\"n_layer\":1,\"n_head\":1"),
                  &profile),
              ErrorCode::InvalidArgument);
}

TEST_CASE(head_profile_load_missing_required_field) {
    HeadProfile profile;
    const struct {
        const char* name;
        std::string json;
        std::string field;
    } cases[] = {
        {"no_model_sha",
         "{\"n_layer\":28,\"n_head\":14,\"probe_params\":{\"block_len\":32,\"repeats\":8,"
         "\"sink\":4,\"recent\":8,\"probe_k\":8,\"seed\":42},\"heads\":[]}",
         "model_sha"},
        {"no_n_layer",
         "{\"model_sha\":\"x\",\"n_head\":14,\"probe_params\":{\"block_len\":32,"
         "\"repeats\":8,\"sink\":4,\"recent\":8,\"probe_k\":8,\"seed\":42},\"heads\":[]}",
         "n_layer"},
        {"no_n_head",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"probe_params\":{\"block_len\":32,"
         "\"repeats\":8,\"sink\":4,\"recent\":8,\"probe_k\":8,\"seed\":42},\"heads\":[]}",
         "n_head"},
        {"no_probe_params",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,\"heads\":[]}",
         "probe_params"},
        {"no_heads",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
         "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
         "\"probe_k\":8,\"seed\":42}}",
         "heads"},
        {"head_entry_no_head",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
         "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
         "\"probe_k\":8,\"seed\":42},"
         "\"heads\":[{\"layer\":0,\"echo_score\":0.1,\"induction_score\":0.1,"
         "\"class\":\"normal\"}]}",
         "head"},
    };
    for (const auto& c : cases) {
        const std::string path = write_profile_file(std::string(c.name) + ".json", c.json);
        const Status st = load_head_profile(path, &profile);
        CHECK_EQ(st.code, ErrorCode::InvalidArgument);
        CHECK(message_mentions(st, c.field));
    }
}

TEST_CASE(head_profile_load_type_errors) {
    HeadProfile profile;
    const struct {
        const char* name;
        std::string json;
        std::string field;
    } cases[] = {
        {"n_layer_string",
         "{\"model_sha\":\"x\",\"n_layer\":\"28\",\"n_head\":14,"
         "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
         "\"probe_k\":8,\"seed\":42},\"heads\":[]}",
         "n_layer"},
        {"echo_score_bool",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
         "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
         "\"probe_k\":8,\"seed\":42},"
         "\"heads\":[{\"layer\":0,\"head\":0,\"echo_score\":true,"
         "\"induction_score\":0.1,\"class\":\"normal\"}]}",
         "echo_score"},
        {"probe_seed_string",
         "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
         "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
         "\"probe_k\":8,\"seed\":\"42\"},\"heads\":[]}",
         "seed"},
    };
    for (const auto& c : cases) {
        const std::string path = write_profile_file(std::string(c.name) + ".json", c.json);
        const Status st = load_head_profile(path, &profile);
        CHECK_EQ(st.code, ErrorCode::InvalidArgument);
        CHECK(message_mentions(st, c.field));
    }
}

TEST_CASE(head_profile_load_bad_class) {
    HeadProfile profile;
    const std::string path = write_profile_file(
        "bad_class.json",
        "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
        "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
        "\"probe_k\":8,\"seed\":42},"
        "\"heads\":[{\"layer\":0,\"head\":0,\"echo_score\":0.1,"
        "\"induction_score\":0.1,\"class\":\"bogus\"}]}");
    const Status st = load_head_profile(path, &profile);
    CHECK_EQ(st.code, ErrorCode::InvalidArgument);
    CHECK(message_mentions(st, "class"));
}

TEST_CASE(head_profile_load_empty_heads) {
    // The design does not forbid an empty heads array: a profile whose
    // structure is complete is legal even with zero scored heads.
    const std::string path = write_profile_file(
        "empty_heads.json",
        "{\"model_sha\":\"x\",\"n_layer\":28,\"n_head\":14,"
        "\"probe_params\":{\"block_len\":32,\"repeats\":8,\"sink\":4,\"recent\":8,"
        "\"probe_k\":8,\"seed\":42},\"heads\":[]}");
    HeadProfile profile;
    CHECK_OK(load_head_profile(path, &profile));
    CHECK(profile.heads.empty());
}

TEST_CASE(head_profile_load_null_out) {
    const std::string path = write_profile_file("valid.json", kValidProfileJson);
    CHECK_ERR(load_head_profile(path, nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(head_profile_validate_match) {
    const HeadProfile profile = load_valid_profile();
    CHECK_OK(validate_head_profile(profile, 28, 14));
}

TEST_CASE(head_profile_validate_layer_mismatch) {
    HeadProfile profile = load_valid_profile();
    // Model has more layers than the profile covers.
    CHECK_ERR(validate_head_profile(profile, 32, 14), ErrorCode::InvalidArgument);
    // Model has fewer layers than the profile claims.
    profile.n_layer = 32;
    CHECK_ERR(validate_head_profile(profile, 28, 14), ErrorCode::InvalidArgument);
}

TEST_CASE(head_profile_validate_head_mismatch) {
    HeadProfile profile = load_valid_profile();
    CHECK_ERR(validate_head_profile(profile, 28, 16), ErrorCode::InvalidArgument);
    profile.n_head = 16;
    CHECK_ERR(validate_head_profile(profile, 28, 14), ErrorCode::InvalidArgument);
}

TEST_CASE(head_profile_validate_entry_out_of_range) {
    HeadProfile profile = load_valid_profile();
    profile.heads.push_back(HeadProfileEntry{28, 0, 0.1, 0.1, false});
    CHECK_ERR(validate_head_profile(profile, 28, 14), ErrorCode::InvalidArgument);

    profile = load_valid_profile();
    profile.heads.push_back(HeadProfileEntry{0, 14, 0.1, 0.1, false});
    CHECK_ERR(validate_head_profile(profile, 28, 14), ErrorCode::InvalidArgument);

    profile = load_valid_profile();
    profile.heads.push_back(HeadProfileEntry{-1, 0, 0.1, 0.1, false});
    CHECK_ERR(validate_head_profile(profile, 28, 14), ErrorCode::InvalidArgument);
}

TEST_CASE(head_adaptive_config_defaults) {
    Config cfg;
    CHECK(cfg.head_profile_path.empty());
    CHECK(!cfg.head_adaptive);
    CHECK_NEAR(cfg.head_retrieval_threshold, 0.10, 1e-12);
    CHECK(cfg.risk_guard);
    CHECK_NEAR(cfg.risk_sep_threshold, 0.5, 1e-12);
    CHECK_NEAR(cfg.risk_cov_threshold, 0.35, 1e-12);
}
