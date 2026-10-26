#include <cstdio>
#include <fstream>
#include <string>

#include "edge_prefill/portfolio.h"
#include "test_framework.h"

using namespace edge_prefill;

namespace {

std::string write_temp(const std::string& name, const std::string& body) {
    const std::string path = "/tmp/edge_test_" + name;
    std::ofstream out(path);
    out << body;
    return path;
}

const char* kTable =
        "# test portfolio\n"
        "model_sha256 = abc123\n"
        "default = selection\n"
        "low:lt4k = selection\n"
        "high:8k = selection\n"
        "high:16k = gate-p16\n"
        "high:32k = gate-p24-local32\n"
        "flat:any = gate-p24-local32\n";

} // namespace

TEST_CASE(portfolio_load_and_recommend) {
    const std::string path = write_temp("portfolio_ok.txt", kTable);
    PortfolioTable table;
    CHECK(table.load(path));
    CHECK_EQ(table.rows().size(), 5u);
    CHECK(table.matches_model("abc123"));
    CHECK(!table.matches_model("def456"));
    CHECK(!table.matches_model(""));

    PortfolioProfile p;
    p.n_tokens = 8000;
    p.specific_terms = 2;
    p.phrase_blocks = 3;
    p.lexical_concentration = 0.5;
    PortfolioRow row;
    CHECK(table.recommend(p, &row));
    CHECK_EQ(row.combo, std::string("selection"));
    CHECK_EQ(row.key, std::string("high:8k"));

    // 16K with discriminative terms routes to the gate in this fixture.
    p.n_tokens = 16000;
    CHECK(table.recommend(p, &row));
    CHECK_EQ(row.combo, std::string("gate-p16"));

    // 32K high -> gate-p24-local32.
    p.n_tokens = 32000;
    CHECK(table.recommend(p, &row));
    CHECK_EQ(row.combo, std::string("gate-p24-local32"));

    // No phrase hits and no concentration -> flat bucket.
    PortfolioProfile flat;
    flat.n_tokens = 16000;
    flat.specific_terms = 0;
    flat.phrase_blocks = 0;
    flat.lexical_concentration = 0.1;
    CHECK(table.recommend(flat, &row));
    CHECK_EQ(row.key, std::string("flat:any"));
    CHECK_EQ(row.combo, std::string("gate-p24-local32"));

    // Unmatched bucket falls back to the default row/combination.
    PortfolioProfile odd;
    odd.n_tokens = 8000;
    odd.specific_terms = 0;
    odd.phrase_blocks = 2;
    odd.lexical_concentration = 0.8;
    CHECK(table.recommend(odd, &row));
    CHECK_EQ(row.key, std::string("default"));
    CHECK_EQ(row.combo, std::string("selection"));

    std::remove(path.c_str());
}

TEST_CASE(portfolio_rejects_unknown_combo) {
    const std::string path = write_temp("portfolio_bad.txt",
                                        "model_sha256 = abc\n"
                                        "high:8k = turbo-9000\n"
                                        "low:8k = selection\n");
    PortfolioTable table;
    CHECK(table.load(path));
    // The unknown combo is dropped, the valid row survives.
    CHECK_EQ(table.rows().size(), 1u);
    PortfolioProfile p;
    p.n_tokens = 8000;
    p.specific_terms = 5;
    p.phrase_blocks = 1;
    p.lexical_concentration = 0.9;
    PortfolioRow row;
    CHECK(table.recommend(p, &row));
    CHECK_EQ(row.combo, std::string("selection"));
    CHECK_EQ(row.key, std::string("default"));
    std::remove(path.c_str());
}

TEST_CASE(portfolio_buckets_are_stable) {
    CHECK_EQ(PortfolioTable::length_bucket(1000), std::string("lt4k"));
    CHECK_EQ(PortfolioTable::length_bucket(4999), std::string("lt4k"));
    CHECK_EQ(PortfolioTable::length_bucket(5000), std::string("8k"));
    CHECK_EQ(PortfolioTable::length_bucket(12000), std::string("16k"));
    CHECK_EQ(PortfolioTable::length_bucket(23999), std::string("16k"));
    CHECK_EQ(PortfolioTable::length_bucket(24000), std::string("32k"));
    CHECK_EQ(PortfolioTable::discriminability(0), std::string("low"));
    CHECK_EQ(PortfolioTable::discriminability(1), std::string("low"));
    CHECK_EQ(PortfolioTable::discriminability(2), std::string("high"));
}
