#include "test_framework.h"

#include <memory>
#include <numeric>
#include <vector>

#include "edge_prefill/prefix_cache.h"

using namespace edge_prefill;

namespace {

KvSignature signature(const std::string& key = "model-a", uint32_t n_ctx = 8192) {
    KvSignature sig;
    sig.model_key = key;
    sig.n_ctx = n_ctx;
    sig.type_k = KvPrecision::F16;
    sig.type_v = KvPrecision::F16;
    return sig;
}

std::vector<llama_token> tokens(uint32_t n, llama_token base = 1) {
    std::vector<llama_token> out(n);
    for (uint32_t i = 0; i < n; ++i) out[i] = base + static_cast<llama_token>(i);
    return out;
}

} // namespace

TEST_CASE(prefix_cache_exact_and_shorter_hit) {
    std::unique_ptr<PrefixCache> cache;
    CHECK_OK(PrefixCache::create(1 << 20, "model-a", &cache));
    auto state = std::vector<uint8_t>(1024, 7);
    CHECK_OK(cache->insert(tokens(100), signature(), state));

    auto hit = cache->find(tokens(100), signature());
    CHECK(hit.has_value());
    if (hit) {
        CHECK_EQ(hit->prefix_len, 100u);
        CHECK(hit->state == state);
    }

    auto shorter = cache->find(tokens(40), signature());
    CHECK(shorter.has_value());
    if (shorter) CHECK_EQ(shorter->prefix_len, 40u);

    auto stats = cache->stats();
    CHECK_EQ(stats.hits, 2u);
    CHECK_EQ(stats.misses, 0u);
    CHECK_EQ(stats.entries, 1u);
}

TEST_CASE(prefix_cache_stale_mismatch_and_signature) {
    std::unique_ptr<PrefixCache> cache;
    CHECK_OK(PrefixCache::create(1 << 20, "model-a", &cache));
    CHECK_OK(cache->insert(tokens(64), signature(), std::vector<uint8_t>(64, 1)));

    CHECK(!cache->find({}, signature()).has_value());
    CHECK(!cache->find(tokens(16, 99), signature()).has_value());
    CHECK(!cache->find(tokens(64), signature("model-b")).has_value());
    CHECK(!cache->find(tokens(64), signature("model-a", 4096)).has_value());
    CHECK_EQ(cache->stats().misses, 4u);
}

TEST_CASE(prefix_cache_evicts_lru) {
    std::unique_ptr<PrefixCache> cache;
    CHECK_OK(PrefixCache::create(100, "model-a", &cache));
    CHECK_OK(cache->insert(tokens(10, 1), signature(), std::vector<uint8_t>(60, 1)));
    CHECK_OK(cache->insert(tokens(10, 100), signature(), std::vector<uint8_t>(60, 2)));
    auto stats = cache->stats();
    CHECK_EQ(stats.entries, 1u);
    CHECK_EQ(stats.evictions, 1u);
    CHECK(stats.bytes <= 100u);
    CHECK(!cache->find(tokens(10, 1), signature()).has_value());
    CHECK(cache->find(tokens(10, 100), signature()).has_value());
}

TEST_CASE(prefix_cache_clear_and_invalid) {
    std::unique_ptr<PrefixCache> cache;
    CHECK_ERR(PrefixCache::create(1000, "model-a", nullptr), ErrorCode::InvalidArgument);
    CHECK_ERR(PrefixCache::create(1000, "", &cache), ErrorCode::InvalidArgument);
    CHECK_OK(PrefixCache::create(1000, "model-a", &cache));

    CHECK_ERR(cache->insert({}, signature(), std::vector<uint8_t>(1, 0)),
              ErrorCode::InvalidArgument);
    CHECK_ERR(cache->insert(tokens(4), signature(), {}), ErrorCode::InvalidArgument);
    CHECK_ERR(cache->insert(tokens(4), KvSignature{}, std::vector<uint8_t>(1, 0)),
              ErrorCode::Cache);
    CHECK_ERR(cache->insert(tokens(4), signature(), std::vector<uint8_t>(2000, 0)),
              ErrorCode::Cache);

    CHECK_OK(cache->insert(tokens(4), signature(), std::vector<uint8_t>(8, 0)));
    cache->clear();
    auto stats = cache->stats();
    CHECK_EQ(stats.entries, 0u);
    CHECK_EQ(stats.bytes, 0u);
}
