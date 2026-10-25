#include "edge_prefill/embedding_scorer.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <utility>

#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

// Encoder window of bge-small style models; longer texts are truncated
// with an audited warning (never silently).
constexpr uint32_t kMaxSeqTokens = 512;
constexpr uint32_t kDefaultParallel = 8;

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return std::max(-1.0, std::min(1.0, dot / (std::sqrt(na) * std::sqrt(nb))));
}

bool vector_is_zero(const std::vector<double>& v) {
    for (double x : v) {
        if (std::fabs(x) > 1e-12) return false;
    }
    return true;
}

// Tiny encoder graphs collapse under many threads: measured on the 16-core
// bench host, 16 threads runs ~7x SLOWER than 8 for 32-token block waves
// (spin-wait contention dominates the small ggml ops). Cap the pool.
constexpr int32_t kEmbedThreadsCap = 8;

} // namespace

void EmbeddingScorer::texts_fingerprint(const std::vector<std::string>& texts,
                                        std::array<uint64_t, 2>* out) {
    // Dual-seed FNV-1a: two independent 64-bit lanes make an accidental
    // collision (which would silently reuse foreign embeddings) negligible.
    uint64_t h0 = 1469598103934665603ull;
    uint64_t h1 = 14695981039346656037ull;
    for (const std::string& t : texts) {
        for (char c : t) {
            h0 = (h0 ^ static_cast<uint64_t>(static_cast<unsigned char>(c))) *
                 1099511628211ull;
            h1 = (h1 ^ static_cast<uint64_t>(static_cast<unsigned char>(c))) *
                 1099511628211ull;
        }
        // Separator so ["ab","c"] never collides with ["a","bc"].
        h0 = (h0 ^ 0x1full) * 1099511628211ull;
        h1 = (h1 ^ 0x9eull) * 1099511628211ull;
    }
    (*out)[0] = h0;
    (*out)[1] = h1;
}

bool EmbeddingScorer::block_cache_match(
        const std::vector<std::string>& block_texts) const {
    if (block_cache_.size() != block_texts.size() ||
           block_cache_key_[0] == 0) {
        return false;
    }
    std::array<uint64_t, 2> key = {0, 0};
    texts_fingerprint(block_texts, &key);
    return key[0] == block_cache_key_[0] && key[1] == block_cache_key_[1];
}

EmbeddingScorer::~EmbeddingScorer() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

Status EmbeddingScorer::load_model(const std::string& path, int32_t n_threads) {
    if (ctx_ || model_) {
        return Status::error(ErrorCode::InvalidArgument, "embedding model already loaded");
    }
    if (path.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty embedding model path");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_ = llama_model_load_from_file(path.c_str(), mparams);
    if (!model_) {
        return Status::error(ErrorCode::ModelLoad, "failed to load embedding model: " + path);
    }
    vocab_ = llama_model_get_vocab(model_);

    n_parallel_ = kDefaultParallel;
    const uint32_t n_ctx = kMaxSeqTokens * n_parallel_;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    // Encoder models process the whole batch as one ubatch (no splitting),
    // so the batch window must hold a full parallel wave: n_parallel * 512.
    cparams.n_batch = kMaxSeqTokens * kDefaultParallel;
    cparams.n_ubatch = kMaxSeqTokens * kDefaultParallel;
    cparams.n_seq_max = n_parallel_;
    const uint32_t hw = std::thread::hardware_concurrency();
    int32_t threads = n_threads > 0 ? n_threads
                                    : static_cast<int32_t>(hw > 0 ? hw : 4);
    if (threads > kEmbedThreadsCap) {
        log_info("embedding threads capped to " +
                 std::to_string(kEmbedThreadsCap) + " (requested " +
                 std::to_string(threads) +
                 "); tiny encoder graphs degrade beyond the cap");
        threads = kEmbedThreadsCap;
    }
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.embeddings = true;
    // Pooling is left unspecified so the GGUF's own pooling head wins
    // (bge = CLS, generic embedders = MEAN). Both produce pooled sequence
    // embeddings; the check below rejects anything else.
    cparams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    cparams.no_perf = true;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return Status::error(ErrorCode::OutOfMemory, "failed to create embedding context");
    }
    // Pooled sequence embeddings only exist when the model resolves to a
    // pooling head; a causal decoder GGUF (NONE) would fabricate scores
    // from unrelated hidden states, so it is rejected outright.
    const enum llama_pooling_type pooling = llama_pooling_type(ctx_);
    if (pooling != LLAMA_POOLING_TYPE_MEAN && pooling != LLAMA_POOLING_TYPE_CLS) {
        llama_free(ctx_);
        ctx_ = nullptr;
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return Status::error(ErrorCode::Unsupported,
                             "not an embedding model (no pooling head): " + path);
    }
    dim_ = static_cast<size_t>(llama_model_n_embd(model_));
    if (dim_ == 0) {
        llama_free(ctx_);
        ctx_ = nullptr;
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return Status::error(ErrorCode::Unsupported, "embedding model has zero dimension");
    }
    log_info("embedding scorer loaded: dim=" + std::to_string(dim_) +
             " n_parallel=" + std::to_string(n_parallel_) +
             " n_ctx=" + std::to_string(llama_n_ctx(ctx_)));
    return Status::success();
}

Status EmbeddingScorer::encode(const std::vector<std::string>& texts,
                               std::vector<std::vector<double>>* embeddings) {
    if (!embeddings) {
        return Status::error(ErrorCode::InvalidArgument, "null embedding output");
    }
    if (texts.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty text list for encoding");
    }
    for (const std::string& t : texts) {
        if (t.empty()) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "empty string in text list for encoding");
        }
    }
    if (!ctx_) {
        return Status::error(ErrorCode::InvalidArgument, "embedding model is not loaded");
    }

    // Tokenize every text up front, truncating to the encoder window with
    // an audited warning (never a silent drop).
    std::vector<std::vector<llama_token>> seq_tokens(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        const std::string& text = texts[i];
        // Buffer-too-small returns the negative needed size; normalize.
        int32_t needed = llama_tokenize(vocab_, text.c_str(),
                                        static_cast<int32_t>(text.size()),
                                        nullptr, 0, true, true);
        if (needed < 0) {
            needed = -needed;
        }
        if (needed <= 0) {
            return Status::error(ErrorCode::Tokenize, "embedding tokenization failed");
        }
        std::vector<llama_token> toks(static_cast<size_t>(needed));
        const int32_t written = llama_tokenize(vocab_, text.c_str(),
                                               static_cast<int32_t>(text.size()),
                                               toks.data(), needed, true, true);
        if (written <= 0) {
            return Status::error(ErrorCode::Tokenize, "embedding tokenization failed");
        }
        toks.resize(static_cast<size_t>(written));
        if (toks.size() > kMaxSeqTokens) {
            log_warn("embedding text " + std::to_string(i) + " truncated from " +
                        std::to_string(toks.size()) + " to " +
                        std::to_string(kMaxSeqTokens) + " tokens");
            toks.resize(kMaxSeqTokens);
        }
        seq_tokens[i] = std::move(toks);
    }

    embeddings->clear();
    embeddings->resize(texts.size());
    // Encode in parallel sequence waves: wave members are independent
    // encoder inputs, each with its own position range restarting at 0.
    for (size_t begin = 0; begin < texts.size(); begin += n_parallel_) {
        const size_t end = std::min(texts.size(), begin + n_parallel_);
        uint32_t wave_tokens = 0;
        for (size_t i = begin; i < end; ++i) {
            wave_tokens += static_cast<uint32_t>(seq_tokens[i].size());
        }
        llama_batch batch = llama_batch_init(static_cast<int32_t>(wave_tokens), 0, 1);
        if (!batch.token) {
            return Status::error(ErrorCode::OutOfMemory, "failed to allocate embedding batch");
        }
        int32_t row = 0;
        for (size_t i = begin; i < end; ++i) {
            const llama_seq_id seq = static_cast<llama_seq_id>(i - begin);
            for (size_t k = 0; k < seq_tokens[i].size(); ++k) {
                batch.token[row] = seq_tokens[i][k];
                batch.pos[row] = static_cast<llama_pos>(k);
                batch.n_seq_id[row] = 1;
                batch.seq_id[row][0] = seq;
                batch.logits[row] = 0;
                ++row;
            }
        }
        batch.n_tokens = row;
        const int32_t ret = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            return Status::error(ErrorCode::Decode, "embedding forward failed");
        }
        for (size_t i = begin; i < end; ++i) {
            const llama_seq_id seq = static_cast<llama_seq_id>(i - begin);
            const float* embd = llama_get_embeddings_seq(ctx_, seq);
            if (!embd) {
                return Status::error(ErrorCode::Unsupported,
                                     "model does not expose sequence embeddings");
            }
            std::vector<double>& out = (*embeddings)[i];
            out.resize(dim_);
            for (size_t k = 0; k < dim_; ++k) {
                out[k] = static_cast<double>(embd[k]);
            }
            if (vector_is_zero(out)) {
                return Status::error(ErrorCode::Internal, "degenerate all-zero embedding");
            }
        }
    }
    return Status::success();
}

Status EmbeddingScorer::score_group_block(
        const std::vector<std::string>& group_texts,
        const std::vector<std::string>& block_texts,
        std::vector<std::vector<double>>* matrix) {
    if (!matrix) {
        return Status::error(ErrorCode::InvalidArgument, "null group-block matrix output");
    }
    if (group_texts.empty() || block_texts.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "empty group or block text list");
    }
    if (!ctx_) {
        return Status::error(ErrorCode::InvalidArgument, "embedding model is not loaded");
    }

    // One combined encode pass: every text (group and block) is encoded
    // exactly once, then paired into the G x B cosine matrix. Block
    // vectors hit the memo cache when selection already encoded the same
    // list (same case, or a sibling case sharing the context).
    const bool cached = block_cache_match(block_texts);
    std::vector<std::string> to_encode;
    if (cached) {
        to_encode.insert(to_encode.end(), group_texts.begin(), group_texts.end());
    } else {
        to_encode.reserve(group_texts.size() + block_texts.size());
        to_encode.insert(to_encode.end(), group_texts.begin(), group_texts.end());
        to_encode.insert(to_encode.end(), block_texts.begin(), block_texts.end());
    }
    std::vector<std::vector<double>> rows;
    Status st = encode(to_encode, &rows);
    if (!st.ok()) {
        return st;
    }

    const size_t n_groups = group_texts.size();
    const size_t n_blocks = block_texts.size();
    if (!cached) {
        block_cache_store(block_texts, rows, n_groups);
    }
    const std::vector<std::vector<double>>& block_rows =
            cached ? block_cache_ : rows;
    const size_t block_row_base = cached ? 0 : n_groups;
    matrix->assign(n_groups, std::vector<double>(n_blocks, 0.0));
    for (size_t g = 0; g < n_groups; ++g) {
        for (size_t b = 0; b < n_blocks; ++b) {
            const double c = cosine(rows[g], block_rows[block_row_base + b]);
            if (!std::isfinite(c)) {
                return Status::error(ErrorCode::Internal,
                                     "non-finite group-block cosine score");
            }
            (*matrix)[g][b] = c;
        }
    }
    return Status::success();
}

void EmbeddingScorer::block_cache_store(
        const std::vector<std::string>& block_texts,
        const std::vector<std::vector<double>>& rows,
        size_t block_row_offset) {
    block_cache_.assign(block_texts.size(), {});
    for (size_t b = 0; b < block_texts.size(); ++b) {
        block_cache_[b] = rows[block_row_offset + b];
    }
    texts_fingerprint(block_texts, &block_cache_key_);
}

Status EmbeddingScorer::score_query_blocks(
        const std::string& query,
        const std::vector<std::string>& block_texts,
        std::vector<double>* block_scores) {
    if (!block_scores) {
        return Status::error(ErrorCode::InvalidArgument, "null block score output");
    }
    if (query.empty() || block_texts.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "empty query or block text list");
    }
    if (!ctx_) {
        return Status::error(ErrorCode::InvalidArgument, "embedding model is not loaded");
    }

    // The query is always encoded; block vectors come from the memo cache
    // when the same list was scored before (deterministic encoder, exact
    // reuse). A miss encodes query + blocks and primes the cache.
    const bool cached = block_cache_match(block_texts);
    std::vector<std::string> to_encode;
    to_encode.push_back(query);
    if (!cached) {
        to_encode.insert(to_encode.end(), block_texts.begin(), block_texts.end());
    }
    std::vector<std::vector<double>> rows;
    Status st = encode(to_encode, &rows);
    if (!st.ok()) {
        return st;
    }
    if (!cached) {
        block_cache_store(block_texts, rows, 1);
    }
    const std::vector<std::vector<double>>& block_rows =
            cached ? block_cache_ : rows;
    const size_t block_row_base = cached ? 0 : 1;

    block_scores->assign(block_texts.size(), 0.0);
    for (size_t b = 0; b < block_texts.size(); ++b) {
        const double c = cosine(rows[0], block_rows[block_row_base + b]);
        if (!std::isfinite(c)) {
            return Status::error(ErrorCode::Internal,
                                 "non-finite query-block cosine score");
        }
        (*block_scores)[b] = c;
    }
    return Status::success();
}

} // namespace edge_prefill
