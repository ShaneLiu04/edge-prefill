#include "edge_prefill/draft_scorer.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

// Cap the draft context at 32k positions: large enough for every planned
// benchmark prompt while keeping the draft KV cache small on edge devices.
constexpr uint32_t kMaxDraftCtx = 32768;

double cosine_normalized(const float* a, const double* b, size_t n) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return std::max(-1.0, std::min(1.0, dot / (std::sqrt(na) * std::sqrt(nb))));
}

} // namespace

LlamaDraftScorer::~LlamaDraftScorer() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

Status LlamaDraftScorer::load_model(const std::string& path, const DraftScorerOptions& opts) {
    if (ctx_ || model_) {
        return Status::error(ErrorCode::InvalidArgument, "draft model already loaded");
    }
    if (path.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty draft model path");
    }
    if (opts.block_size == 0 || opts.chunk_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "draft scorer block/chunk size must be > 0");
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_ = llama_model_load_from_file(path.c_str(), mparams);
    if (!model_) {
        return Status::error(ErrorCode::ModelLoad, "failed to load draft model: " + path);
    }
    vocab_ = llama_model_get_vocab(model_);

    const int32_t n_ctx_train = llama_model_n_ctx_train(model_);
    const uint32_t n_ctx = std::min<uint32_t>(n_ctx_train > 0 ? static_cast<uint32_t>(n_ctx_train)
                                                              : kMaxDraftCtx,
                                              kMaxDraftCtx);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = std::max<uint32_t>(opts.chunk_size, 512);
    cparams.n_ubatch = std::max<uint32_t>(opts.chunk_size, 256);
    cparams.n_seq_max = 1;
    const uint32_t hw = std::thread::hardware_concurrency();
    cparams.n_threads = opts.n_threads > 0 ? opts.n_threads
                                           : static_cast<int32_t>(hw > 0 ? hw : 4);
    cparams.n_threads_batch = cparams.n_threads;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf = true;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
        return Status::error(ErrorCode::OutOfMemory, "failed to create draft context");
    }
    opts_ = opts;
    log_info("draft scorer loaded: n_vocab=" + std::to_string(llama_vocab_n_tokens(vocab_)) +
             " n_ctx=" + std::to_string(llama_n_ctx(ctx_)) +
             " n_embd=" + std::to_string(llama_model_n_embd(model_)));
    return Status::success();
}

int32_t LlamaDraftScorer::n_vocab() const {
    return vocab_ ? llama_vocab_n_tokens(vocab_) : 0;
}

uint32_t LlamaDraftScorer::n_ctx() const {
    return ctx_ ? llama_n_ctx(ctx_) : 0;
}

Status LlamaDraftScorer::score(const std::vector<llama_token>& tokens,
                               const std::string& query,
                               std::vector<double>* per_token_scores) {
    if (!per_token_scores) {
        return Status::error(ErrorCode::InvalidArgument, "null draft score output");
    }
    if (tokens.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty token sequence for draft scoring");
    }
    if (query.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty query for draft scoring");
    }
    if (!ctx_) {
        return Status::error(ErrorCode::Unsupported, "draft model is not loaded");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab_);
    for (llama_token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            return Status::error(ErrorCode::Unsupported,
                                 "draft vocab does not cover main-model token ids");
        }
    }

    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());
    const uint32_t block_size = opts_.block_size;
    const uint32_t n_ctx = llama_n_ctx(ctx_);
    if (n_tokens > n_ctx) {
        return Status::error(ErrorCode::Unsupported, "prompt exceeds draft context window");
    }

    // Encode the query with the draft tokenizer.
    const int32_t q_needed = -llama_tokenize(vocab_, query.c_str(),
                                             static_cast<int32_t>(query.size()), nullptr, 0,
                                             true, true);
    if (q_needed <= 0) {
        return Status::error(ErrorCode::Tokenize, "draft query tokenization failed");
    }
    std::vector<llama_token> q_tokens(static_cast<size_t>(q_needed));
    const int32_t q_written = llama_tokenize(vocab_, query.c_str(),
                                             static_cast<int32_t>(query.size()),
                                             q_tokens.data(), q_needed, true, true);
    if (q_written <= 0) {
        return Status::error(ErrorCode::Tokenize, "draft query tokenization failed");
    }
    q_tokens.resize(static_cast<size_t>(q_written));
    if (q_tokens.size() > n_ctx) {
        return Status::error(ErrorCode::Unsupported, "query exceeds draft context window");
    }

    const int32_t n_embd = llama_model_n_embd(model_);

    // Query pass: mean-pool the embeddings of all query tokens.
    llama_memory_clear(llama_get_memory(ctx_), true);
    {
        llama_batch batch = llama_batch_init(static_cast<int32_t>(q_tokens.size()), 0, 1);
        if (!batch.token) {
            return Status::error(ErrorCode::OutOfMemory, "failed to allocate draft query batch");
        }
        for (size_t i = 0; i < q_tokens.size(); ++i) {
            batch.token[i] = q_tokens[i];
            batch.pos[i] = static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = static_cast<int32_t>(q_tokens.size());
        const int32_t ret = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            return Status::error(ErrorCode::Decode, "draft query forward failed");
        }
    }
    std::vector<double> q_vec(static_cast<size_t>(n_embd), 0.0);
    for (size_t j = 0; j < q_tokens.size(); ++j) {
        const float* embd = llama_get_embeddings_ith(ctx_, static_cast<int32_t>(j));
        if (!embd) {
            return Status::error(ErrorCode::Unsupported,
                                 "draft model does not expose token embeddings");
        }
        for (int32_t k = 0; k < n_embd; ++k) {
            q_vec[k] += static_cast<double>(embd[k]);
        }
    }
    double q_norm = 0.0;
    for (double v : q_vec) q_norm += v * v;
    q_norm = std::sqrt(q_norm);
    if (q_norm < 1e-12) {
        return Status::error(ErrorCode::Internal, "draft query embedding is degenerate");
    }
    for (double& v : q_vec) v /= q_norm;

    // Prompt pass: one forward in chunks, collecting the embedding at each
    // block boundary (last token of every block, plus the final token).
    llama_memory_clear(llama_get_memory(ctx_), true);
    const uint32_t chunk = std::max<uint32_t>(opts_.chunk_size, block_size);
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    std::vector<double> block_scores(n_blocks, 0.0);
    std::vector<bool> block_scored(n_blocks, false);

    for (uint32_t begin = 0; begin < n_tokens; begin += chunk) {
        const uint32_t end = std::min(n_tokens, begin + chunk);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(end - begin), 0, 1);
        if (!batch.token) {
            return Status::error(ErrorCode::OutOfMemory, "failed to allocate draft batch");
        }
        std::vector<uint32_t> boundary_blocks;
        std::vector<int32_t> boundary_rows;
        int32_t j = 0;
        for (uint32_t i = begin; i < end; ++i, ++j) {
            batch.token[j] = tokens[i];
            batch.pos[j] = static_cast<llama_pos>(i);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            const bool boundary = ((i + 1) % block_size == 0) || (i + 1 == n_tokens);
            batch.logits[j] = boundary ? 1 : 0;
            if (boundary) {
                boundary_blocks.push_back(i / block_size);
                boundary_rows.push_back(j);
            }
        }
        batch.n_tokens = j;
        const int32_t ret = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            return Status::error(ErrorCode::Decode, "draft prompt forward failed");
        }
        for (size_t o = 0; o < boundary_blocks.size(); ++o) {
            // llama_get_embeddings_ith indexes by batch token position, not by
            // output counter: with embeddings=true the batch layer forces every
            // token to be an output, so the chunk-local row of the boundary
            // token is the index that resolves to its embedding.
            const float* embd = llama_get_embeddings_ith(ctx_, boundary_rows[o]);
            if (!embd) {
                return Status::error(ErrorCode::Unsupported,
                                     "draft model does not expose token embeddings");
            }
            block_scores[boundary_blocks[o]] =
                cosine_normalized(embd, q_vec.data(), static_cast<size_t>(n_embd));
            block_scored[boundary_blocks[o]] = true;
        }
    }

    for (uint32_t b = 0; b < n_blocks; ++b) {
        if (!block_scored[b]) {
            return Status::error(ErrorCode::Internal, "draft block score missing");
        }
    }

    per_token_scores->assign(n_tokens, 0.0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        (*per_token_scores)[i] = block_scores[i / block_size];
    }
    return Status::success();
}

} // namespace edge_prefill
