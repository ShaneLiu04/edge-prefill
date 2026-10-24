#include "test_framework.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "edge_prefill/attention_sparsity.h"
#include "edge_prefill/backend.h"
#include "edge_prefill/embedding_scorer.h"
#include "edge_prefill/selector.h"

using namespace edge_prefill;

namespace {

std::string embedding_model_path() {
    const char* env = std::getenv("EDGE_PREFILL_TEST_EMBEDDING");
    return env ? env : "";
}

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

Selection make_selection(uint32_t n = 4096, double budget = 0.5) {
    std::vector<llama_token> tokens(n);
    std::vector<std::string> pieces(n, "filler");
    for (uint32_t i = 0; i < n; ++i) tokens[i] = static_cast<llama_token>(i % 1000);
    constexpr uint32_t needle_pos = 37 * 32 + 5;
    if (needle_pos < n) pieces[needle_pos] = "needle";

    SelectionOptions opts;
    opts.block_size = 32;
    opts.budget = budget;
    opts.min_dense_tokens = 512;
    Selection sel;
    BlockSelector selector;
    auto st = selector.select(tokens, "needle", pieces, opts, nullptr, &sel);
    if (!st.ok()) {
        CHECK_OK(st);
    }
    return sel;
}

QueryGatingOptions qg_options() {
    QueryGatingOptions opts;
    opts.query_chunk_size = 128;
    opts.query_chunk_max = 16;
    opts.top_k_per_group = 256;
    return opts;
}

// Counts how many scores in `row` are strictly greater than row[needle].
// The needle ranks in the top 10% of 125 blocks when this stays below 12.
uint32_t count_higher(const std::vector<double>& row, uint32_t needle) {
    uint32_t higher = 0;
    for (uint32_t i = 0; i < row.size(); ++i) {
        if (i != needle && row[i] > row[needle]) ++higher;
    }
    return higher;
}

constexpr uint32_t n_blocks = 125;
constexpr uint32_t needle_block = 77;

std::vector<std::string> make_block_texts() {
    std::vector<std::string> blocks;
    blocks.reserve(n_blocks);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (b == needle_block) {
            blocks.push_back(
                "Archive note for record " + std::to_string(b) + ". The magic passcode "
                "stored in the archive is 4271. Keep this note away from the general "
                "weather and harvest discussion. The librarian locked the drawer before "
                "leaving the reading room. No other record mentions a code.");
            continue;
        }
        std::string text;
        for (uint32_t s = 0; s < 6; ++s) {
            text += "Passage " + std::to_string(b) + " note " + std::to_string(s) +
                    ": The weather report describes mild winds across the valley while "
                    "farmers discuss harvest " + std::to_string(b * 10 + s) +
                    " and the slow river crossing near the mill. ";
        }
        blocks.push_back(text);
    }
    return blocks;
}

std::vector<std::string> make_group_texts() {
    return {
        "What is the capital of the country mentioned in the report? What is the "
        "capital of the country mentioned in the report? What is the capital of the "
        "country mentioned in the report?",
        "Summarize the yearly budget for the committee. Summarize the yearly budget "
        "for the committee. Summarize the yearly budget for the committee.",
        "What is the magic passcode stored in the archive? What is the magic passcode "
        "stored in the archive? What is the magic passcode stored in the archive? "
        "What is the magic passcode stored in the archive?",
        "Which river crossing is discussed by the farmers? Which river crossing is "
        "discussed by the farmers? Which river crossing is discussed by the farmers?",
        "How strong are the winds described in the valley? How strong are the winds "
        "described in the valley? How strong are the winds described in the valley?",
        "List the harvest numbers recorded this season. List the harvest numbers "
        "recorded this season. List the harvest numbers recorded this season.",
        "Who locked the drawer in the reading room? Who locked the drawer in the "
        "reading room? Who locked the drawer in the reading room?",
        "Describe the route from the mill to the village. Describe the route from "
        "the mill to the village. Describe the route from the mill to the village?",
    };
}

} // namespace

TEST_CASE(embedding_scorer_load_validation) {
    EmbeddingScorer scorer;
    CHECK_ERR(scorer.load_model(""), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.load_model("definitely/not/a/model.gguf"), ErrorCode::ModelLoad);
    CHECK(!scorer.loaded());

    // Nothing is scored before a model is loaded, whatever the inputs.
    std::vector<std::vector<double>> emb;
    CHECK_ERR(scorer.encode({"text"}, &emb), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.encode({"text"}, nullptr), ErrorCode::InvalidArgument);
    std::vector<std::vector<double>> matrix;
    CHECK_ERR(scorer.score_group_block({"g"}, {"b"}, &matrix), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_group_block({"g"}, {"b"}, nullptr), ErrorCode::InvalidArgument);
    std::vector<double> scores;
    CHECK_ERR(scorer.score_query_blocks("q", {"b"}, &scores), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_query_blocks("q", {"b"}, nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(embedding_scorer_encode_validation) {
    EmbeddingScorer scorer;
    std::vector<std::vector<double>> emb;
    CHECK_ERR(scorer.encode({}, &emb), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.encode({"alpha", ""}, &emb), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.encode({"alpha"}, nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(embedding_scorer_group_block_validation) {
    EmbeddingScorer scorer;
    std::vector<std::vector<double>> matrix;
    CHECK_ERR(scorer.score_group_block({}, {"b"}, &matrix), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_group_block({"g"}, {}, &matrix), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_group_block({"g"}, {"b"}, nullptr), ErrorCode::InvalidArgument);

    std::vector<double> scores;
    CHECK_ERR(scorer.score_query_blocks("", {"b"}, &scores), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_query_blocks("q", {}, &scores), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.score_query_blocks("q", {"b"}, nullptr), ErrorCode::InvalidArgument);
}

TEST_CASE(embedding_scorer_real_model_geometry_and_needle) {
    const std::string path = embedding_model_path();
    if (path.empty()) {
        std::cout << "SKIP embedding_scorer_real_model_geometry_and_needle "
                     "(EDGE_PREFILL_TEST_EMBEDDING unset)\n";
        return;
    }

    LlamaBackend::init();
    EmbeddingScorer scorer;
    CHECK_OK(scorer.load_model(path));
    CHECK(scorer.loaded());
    CHECK_GT(scorer.dim(), 0u);
    // bge-small-en-v1.5 emits 384-dim mean-pooled embeddings.
    CHECK_EQ(scorer.dim(), 384u);
    CHECK_EQ(scorer.n_parallel(), 8u);

    // Geometry: one row per text, in input order, each of model dimension.
    std::vector<std::vector<double>> emb;
    CHECK_OK(scorer.encode({"The quick brown fox jumps over the lazy dog.",
                            "The quick brown fox jumps over the lazy dog.",
                            "Quarterly invoices were reconciled against the ledger."},
                           &emb));
    CHECK_EQ(emb.size(), 3u);
    for (const auto& row : emb) {
        CHECK_EQ(row.size(), static_cast<size_t>(scorer.dim()));
        for (double v : row) CHECK(std::isfinite(v));
    }
    // Cosine properties: identical texts -> cos == 1.
    CHECK_NEAR(cosine(emb[0], emb[1]), 1.0, 1e-6);
    // Unrelated texts stay well below 1.
    CHECK(std::fabs(cosine(emb[0], emb[2])) < 0.9);

    // Validation still fires with a loaded model.
    CHECK_ERR(scorer.encode({}, &emb), ErrorCode::InvalidArgument);
    CHECK_ERR(scorer.encode({"ok", ""}, &emb), ErrorCode::InvalidArgument);

    // 125 prompt blocks under the Selection blocking (block b covers
    // [b*32, (b+1)*32)); the needle block hides the passcode sentence.
    const std::vector<std::string> blocks = make_block_texts();
    CHECK_EQ(blocks.size(), static_cast<size_t>(n_blocks));
    const std::vector<std::string> groups = make_group_texts();
    CHECK_EQ(groups.size(), 8u);
    constexpr uint32_t passcode_group = 2;

    // G x B matrix: all finite, needle ranks top 10% for the question group.
    std::vector<std::vector<double>> matrix;
    CHECK_OK(scorer.score_group_block(groups, blocks, &matrix));
    CHECK_EQ(matrix.size(), 8u);
    for (uint32_t g = 0; g < 8u; ++g) {
        CHECK_EQ(matrix[g].size(), static_cast<size_t>(n_blocks));
        for (uint32_t b = 0; b < n_blocks; ++b) {
            CHECK(std::isfinite(matrix[g][b]));
            CHECK(matrix[g][b] >= -1.0 - 1e-9);
            CHECK(matrix[g][b] <= 1.0 + 1e-9);
        }
    }
    // Top 10% of 125 blocks: at most 11 blocks may outrank the needle.
    CHECK(count_higher(matrix[passcode_group], needle_block) < 12);

    // Query -> block scores: same needle ranking property.
    std::vector<double> query_scores;
    CHECK_OK(scorer.score_query_blocks(
        "What is the magic passcode stored in the archive?", blocks, &query_scores));
    CHECK_EQ(query_scores.size(), static_cast<size_t>(n_blocks));
    for (double s : query_scores) CHECK(std::isfinite(s));
    CHECK(count_higher(query_scores, needle_block) < 12);

    // Over-long text (600+ tokens) is truncated with a warning, not an error.
    std::string long_text;
    for (uint32_t i = 0; i < 60; ++i) {
        long_text += "Sentence " + std::to_string(i) +
                     " about quiet hills and distant bridges repeats here. ";
    }
    std::vector<std::vector<double>> long_emb;
    CHECK_OK(scorer.encode({long_text}, &long_emb));
    CHECK_EQ(long_emb.size(), 1u);
    CHECK_EQ(long_emb[0].size(), static_cast<size_t>(scorer.dim()));
    for (double v : long_emb[0]) CHECK(std::isfinite(v));

    LlamaBackend::shutdown();
}

TEST_CASE(build_group_spans_consistency) {
    const Selection sel = make_selection(4096, 0.5);
    CHECK(!sel.dense);
    const QueryGatingOptions opts = qg_options();

    // v1 partition of the same row space for cross-checking.
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> q_bounds;
    std::string qg_reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            sel, std::vector<double>(sel.total_tokens, 0.0), opts,
            &groups, &q_bounds, &qg_reason));
    CHECK(!groups.empty());

    std::vector<std::pair<uint32_t, uint32_t>> spans;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_group_spans(sel, opts, &spans, &reason));
    CHECK(!spans.empty());
    // Same group count as the v1 partition.
    CHECK_EQ(spans.size(), groups.size());
    CHECK_EQ(spans.size() + 1, q_bounds.size());

    // Well-formed, ordered, non-overlapping [lo, hi) token ranges.
    for (size_t g = 0; g < spans.size(); ++g) {
        CHECK(spans[g].first < spans[g].second);
        CHECK(spans[g].second <= sel.total_tokens);
        if (g > 0) CHECK(spans[g].first >= spans[g - 1].second);
    }

    // Each group's span covers exactly the token positions of the group's
    // rows in the v1 row-space partition.
    for (size_t g = 0; g < spans.size(); ++g) {
        for (int32_t r = q_bounds[g]; r < q_bounds[g + 1]; ++r) {
            CHECK(r >= 0);
            CHECK(static_cast<size_t>(r) < sel.token_indices.size());
            const uint32_t pos = sel.token_indices[static_cast<size_t>(r)];
            CHECK(pos >= spans[g].first);
            CHECK(pos < spans[g].second);
        }
    }
    // Union property: every selected position belongs to exactly one span,
    // so spans U unselected positions tile the whole prompt.
    for (uint32_t pos : sel.token_indices) {
        size_t owner = spans.size();
        for (size_t g = 0; g < spans.size(); ++g) {
            if (pos >= spans[g].first && pos < spans[g].second) {
                owner = g;
                break;
            }
        }
        CHECK(owner < spans.size());
    }

    // Dense selection: Ok with empty spans and a non-empty reason.
    const Selection dense = make_selection(256, 0.5);
    CHECK(dense.dense);
    std::vector<std::pair<uint32_t, uint32_t>> dense_spans;
    std::string dense_reason;
    CHECK_OK(AttentionSparsityController::build_group_spans(
            dense, opts, &dense_spans, &dense_reason));
    CHECK(dense_spans.empty());
    CHECK(!dense_reason.empty());

    // Empty row space: Ok with empty spans and a reason.
    Selection empty_rows;
    empty_rows.dense = false;
    empty_rows.total_tokens = 4096;
    std::vector<std::pair<uint32_t, uint32_t>> empty_spans;
    std::string empty_reason;
    CHECK_OK(AttentionSparsityController::build_group_spans(
            empty_rows, opts, &empty_spans, &empty_reason));
    CHECK(empty_spans.empty());
    CHECK(!empty_reason.empty());

    // Null outputs are rejected.
    CHECK_ERR(AttentionSparsityController::build_group_spans(sel, opts, nullptr, &reason),
              ErrorCode::InvalidArgument);
    std::vector<std::pair<uint32_t, uint32_t>> spans2;
    CHECK_ERR(AttentionSparsityController::build_group_spans(sel, opts, &spans2, nullptr),
              ErrorCode::InvalidArgument);
}

TEST_CASE(build_group_spans_degrade_single_group) {
    // 64-row row space (< query_chunk_size 128): the partition degrades to
    // a single group spanning the whole selected prompt prefix.
    Selection sel;
    sel.dense = false;
    sel.total_tokens = 4096;
    sel.selected_tokens = 64;
    sel.token_indices.reserve(64);
    for (uint32_t i = 0; i < 64; ++i) sel.token_indices.push_back(i);
    sel.block_indices = {0, 1};

    const QueryGatingOptions opts = qg_options();
    std::vector<std::pair<uint32_t, uint32_t>> spans;
    std::string reason;
    CHECK_OK(AttentionSparsityController::build_group_spans(sel, opts, &spans, &reason));
    CHECK_EQ(spans.size(), 1u);
    CHECK_EQ(spans[0].first, 0u);
    CHECK_EQ(spans[0].second, 64u);

    // v1 degrades the same way: one group over the whole row space.
    std::vector<std::vector<int32_t>> groups;
    std::vector<int32_t> q_bounds;
    std::string qg_reason;
    CHECK_OK(AttentionSparsityController::build_query_gated(
            sel, std::vector<double>(sel.total_tokens, 0.0), opts,
            &groups, &q_bounds, &qg_reason));
    CHECK_EQ(groups.size(), 1u);
}
