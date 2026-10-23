#include "edge_prefill/backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <sys/stat.h>

#include <ggml.h>

#include "edge_prefill/log.h"

namespace edge_prefill {
namespace {

ggml_type to_ggml_type(KvPrecision precision) {
    switch (precision) {
        case KvPrecision::F16: return GGML_TYPE_F16;
        case KvPrecision::Q8_0: return GGML_TYPE_Q8_0;
        case KvPrecision::Q4_0: return GGML_TYPE_Q4_0;
    }
    return GGML_TYPE_F16;
}

uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
    return std::string(buf);
}

} // namespace

std::string ModelFingerprint::to_key() const {
    return arch + ":" + std::to_string(file_size) + ":" + std::to_string(mtime_ns) + ":" +
           head_hash + ":" + std::to_string(n_layer) + ":" + std::to_string(n_ctx_train);
}

LlamaBackend::~LlamaBackend() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

Status LlamaBackend::init() {
    llama_backend_init();
    return Status::success();
}

void LlamaBackend::shutdown() { llama_backend_free(); }

Status LlamaBackend::load_model(const std::string& path, bool use_mmap) {
    if (model_) {
        return Status::error(ErrorCode::InvalidArgument, "model already loaded");
    }
    if (path.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty model path");
    }
    llama_model_params params = llama_model_default_params();
    params.load_mode = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    params.n_gpu_layers = n_gpu_layers_;
    model_ = llama_model_load_from_file(path.c_str(), params);
    if (!model_) {
        return Status::error(ErrorCode::ModelLoad, "failed to load model: " + path);
    }
    vocab_ = llama_model_get_vocab(model_);
    model_path_ = path;
    return Status::success();
}

void LlamaBackend::set_n_gpu_layers(int32_t n) { n_gpu_layers_ = n; }

Status LlamaBackend::create_context(const ContextOptions& opts) {
    if (!model_) {
        return Status::error(ErrorCode::ContextInit, "model is not loaded");
    }
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (opts.n_ubatch == 0) {
        return Status::error(ErrorCode::InvalidArgument, "n_ubatch must be > 0");
    }
    if (opts.n_ctx > (1u << 30)) {
        return Status::error(ErrorCode::InvalidArgument, "n_ctx exceeds implementation limit");
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = opts.n_ctx;
    params.n_batch = std::max<uint32_t>(opts.n_batch, opts.n_ubatch);
    params.n_ubatch = opts.n_ubatch;
    params.n_seq_max = std::max<uint32_t>(1, opts.n_seq_max);
    const uint32_t hw = std::thread::hardware_concurrency();
    const int32_t threads =
        opts.n_threads > 0 ? opts.n_threads : static_cast<int32_t>(hw > 0 ? hw : 4);
    params.n_threads = threads;
    params.n_threads_batch = opts.n_threads_batch > 0 ? opts.n_threads_batch : threads;
    params.type_k = to_ggml_type(opts.type_k);
    params.type_v = to_ggml_type(opts.type_v);
    params.flash_attn_type = opts.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = opts.offload_kqv;
    params.embeddings = opts.embeddings;
    params.no_perf = true;

    ctx_ = llama_init_from_model(model_, params);
    if (!ctx_) {
        return Status::error(ErrorCode::OutOfMemory, "failed to create llama context");
    }
    context_options_ = opts;
    context_options_.n_ctx = llama_n_ctx(ctx_);
    return Status::success();
}

Status LlamaBackend::tokenize(const std::string& text, std::vector<llama_token>* out,
                              bool add_special, bool parse_special) const {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null token output");
    if (!vocab_) return Status::error(ErrorCode::Tokenize, "vocab is not loaded");
    out->clear();
    const int32_t needed = -llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                                           nullptr, 0, add_special, parse_special);
    if (needed < 0) {
        return Status::error(ErrorCode::Tokenize, "llama_tokenize failed");
    }
    out->resize(static_cast<size_t>(needed));
    if (needed > 0) {
        const int32_t written = llama_tokenize(vocab_, text.c_str(),
                                               static_cast<int32_t>(text.size()),
                                               out->data(), needed, add_special, parse_special);
        if (written < 0) {
            return Status::error(ErrorCode::Tokenize, "llama_tokenize failed");
        }
        out->resize(static_cast<size_t>(written));
    }
    return Status::success();
}

Status LlamaBackend::token_to_piece(llama_token token, std::string* out) const {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null piece output");
    if (!vocab_) return Status::error(ErrorCode::InvalidArgument, "vocab is not loaded");
    if (token < 0 || token >= llama_vocab_n_tokens(vocab_)) {
        return Status::error(ErrorCode::InvalidArgument, "token out of range");
    }
    char buf[256];
    const int32_t len = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, true);
    if (len < 0) {
        return Status::error(ErrorCode::InvalidArgument, "cannot convert token to piece");
    }
    out->assign(buf, static_cast<size_t>(len));
    return Status::success();
}

Status LlamaBackend::prefill(const std::vector<llama_token>& tokens,
                             const std::vector<llama_pos>& positions,
                             uint32_t chunk_size,
                             PrefillStats* stats) {
    if (!ctx_) return Status::error(ErrorCode::Prefill, "context is not created");
    if (!stats) return Status::error(ErrorCode::InvalidArgument, "null prefill stats");
    if (tokens.empty() || tokens.size() != positions.size()) {
        return Status::error(ErrorCode::InvalidArgument, "tokens/positions size mismatch");
    }

    const uint32_t n_ctx = llama_n_ctx(ctx_);
    for (size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] < 0 || static_cast<uint32_t>(positions[i]) >= n_ctx) {
            return Status::error(ErrorCode::InvalidArgument, "position out of context range");
        }
        if (i > 0 && positions[i] <= positions[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument, "positions must be strictly increasing");
        }
    }

    std::vector<Range> ranges;
    auto st = ChunkScheduler::plan(static_cast<uint32_t>(tokens.size()), chunk_size,
                                   llama_n_ubatch(ctx_), &ranges);
    if (!st.ok()) return st;

    stats->prompt_tokens = static_cast<uint32_t>(tokens.size());
    stats->selected_tokens = static_cast<uint32_t>(tokens.size());
    stats->chunks = static_cast<uint32_t>(ranges.size());
    stats->chunk_stats.clear();
    const auto begin = std::chrono::steady_clock::now();

    set_prefill_mode(true);
    for (size_t r = 0; r < ranges.size(); ++r) {
        const auto& range = ranges[r];
        llama_batch batch = llama_batch_init(static_cast<int32_t>(range.size()), 0, 1);
        if (!batch.token) {
            set_prefill_mode(false);
            return Status::error(ErrorCode::OutOfMemory, "failed to allocate llama batch");
        }
        int32_t j = 0;
        for (uint32_t i = range.begin; i < range.end; ++i, ++j) {
            batch.token[j] = tokens[i];
            batch.pos[j] = positions[i];
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            const bool last_chunk = (r + 1 == ranges.size());
            const bool last_token = (i + 1 == tokens.size());
            batch.logits[j] = (last_chunk && last_token) ? 1 : 0;
        }
        batch.n_tokens = j;
        const auto chunk_begin = std::chrono::steady_clock::now();
        const int32_t ret = llama_decode(ctx_, batch);
        const auto chunk_end = std::chrono::steady_clock::now();
        const uint64_t elapsed_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(chunk_end - chunk_begin).count());
        llama_batch_free(batch);
        if (ret != 0) {
            set_prefill_mode(false);
            return Status::error(ErrorCode::Prefill,
                                 "llama_decode failed in chunk " + std::to_string(r) +
                                     " (ret=" + std::to_string(ret) + ")");
        }
        if (stats->collect_chunk_stats) {
            ChunkStats cs;
            cs.index = static_cast<uint32_t>(r);
            cs.tokens = range.size();
            cs.elapsed_us = elapsed_us;
            cs.kv_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0);
            stats->chunk_stats.push_back(cs);
        }
    }
    set_prefill_mode(false);

    const auto end = std::chrono::steady_clock::now();
    stats->total_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
    stats->ttft_us = stats->total_us;
    auto* mem = llama_get_memory(ctx_);
    stats->kv_pos_min = llama_memory_seq_pos_min(mem, 0);
    stats->kv_pos_max = llama_memory_seq_pos_max(mem, 0);
    stats->kv_bytes = llama_state_seq_get_size(ctx_, 0);
    return Status::success();
}

Status LlamaBackend::decode(llama_token token, llama_pos pos, std::vector<float>* logits) {
    if (!ctx_) return Status::error(ErrorCode::Decode, "context is not created");
    if (!logits) return Status::error(ErrorCode::InvalidArgument, "null logits");
    if (pos < 0) return Status::error(ErrorCode::InvalidArgument, "negative position");

    llama_batch batch = llama_batch_init(1, 0, 1);
    if (!batch.token) {
        return Status::error(ErrorCode::OutOfMemory, "failed to allocate decode batch");
    }
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    const int32_t ret = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (ret != 0) {
        return Status::error(ErrorCode::Decode, "llama_decode failed");
    }
    const float* data = llama_get_logits_ith(ctx_, -1);
    if (!data) return Status::error(ErrorCode::Decode, "missing logits");
    logits->assign(data, data + llama_vocab_n_tokens(vocab_));
    return Status::success();
}

Status LlamaBackend::probe_argmax(const std::vector<llama_token>& tokens,
                                  std::vector<int32_t>* top1) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    if (!top1) return Status::error(ErrorCode::InvalidArgument, "null top1 output");
    if (tokens.empty()) return Status::error(ErrorCode::InvalidArgument, "empty tokens");
    if (!vocab_) return Status::error(ErrorCode::InvalidArgument, "vocab is not loaded");
    const uint32_t n_ctx = llama_n_ctx(ctx_);
    if (tokens.size() > static_cast<size_t>(n_ctx)) {
        return Status::error(ErrorCode::InvalidArgument, "tokens exceed n_ctx");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab_);
    top1->clear();
    top1->resize(tokens.size(), -1);

    // One logits row is n_vocab floats; chunking bounds the output buffer
    // the runtime must reserve for a probe pass. The chunk must also stay
    // within n_batch: llama_context::decode asserts n_tokens_all <=
    // n_batch, and contexts created with a small n_ctx (padded up by the
    // runtime) can still carry a small n_batch.
    constexpr uint32_t kProbeChunk = 128;
    const uint32_t chunk_cap =
            std::min<uint32_t>(kProbeChunk, llama_n_batch(ctx_));
    if (chunk_cap == 0) {
        set_prefill_mode(false);
        return Status::error(ErrorCode::Internal, "n_batch is zero");
    }
    set_prefill_mode(true);
    for (size_t offset = 0; offset < tokens.size(); offset += chunk_cap) {
        const int32_t chunk =
                static_cast<int32_t>(std::min<size_t>(chunk_cap, tokens.size() - offset));
        llama_batch batch = llama_batch_init(chunk, 0, 1);
        if (!batch.token) {
            set_prefill_mode(false);
            return Status::error(ErrorCode::OutOfMemory, "failed to allocate llama batch");
        }
        for (int32_t j = 0; j < chunk; ++j) {
            batch.token[j] = tokens[offset + static_cast<size_t>(j)];
            batch.pos[j] = static_cast<llama_pos>(offset + static_cast<size_t>(j));
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            // Every position needs its logits: the classifier compares
            // per-position argmax against the dense baseline.
            batch.logits[j] = 1;
        }
        batch.n_tokens = chunk;
        const int32_t ret = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            set_prefill_mode(false);
            return Status::error(ErrorCode::Prefill, "llama_decode failed in probe forward");
        }
        for (int32_t j = 0; j < chunk; ++j) {
            const float* row = llama_get_logits_ith(ctx_, j);
            if (!row) {
                set_prefill_mode(false);
                return Status::error(ErrorCode::Prefill, "missing probe logits");
            }
            int32_t best = 0;
            for (int32_t t = 1; t < n_vocab; ++t) {
                if (row[t] > row[best]) best = t;
            }
            (*top1)[offset + static_cast<size_t>(j)] = best;
        }
    }
    set_prefill_mode(false);
    return Status::success();
}

Status LlamaBackend::save_state(std::vector<uint8_t>* out) const {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null state output");
    if (!ctx_) return Status::error(ErrorCode::Cache, "context is not created");
    const size_t size = llama_state_seq_get_size(ctx_, 0);
    if (size == 0) return Status::error(ErrorCode::Cache, "empty state");
    out->resize(size);
    const size_t written = llama_state_seq_get_data(ctx_, out->data(), size, 0);
    if (written != size) {
        return Status::error(ErrorCode::Cache, "failed to serialize sequence state");
    }
    return Status::success();
}

Status LlamaBackend::load_state(const std::vector<uint8_t>& in) {
    if (!ctx_) return Status::error(ErrorCode::Cache, "context is not created");
    if (in.size() < 16) return Status::error(ErrorCode::InvalidArgument, "state too small");
    const size_t read = llama_state_seq_set_data(ctx_, in.data(), in.size(), 0);
    if (read != in.size()) {
        return Status::error(ErrorCode::Cache, "failed to restore sequence state");
    }
    return Status::success();
}

Status LlamaBackend::reset_sequence() {
    if (!ctx_) return Status::error(ErrorCode::Cache, "context is not created");
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, -1, -1);
    return Status::success();
}

Status LlamaBackend::set_prefill_mode(bool prefill) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    llama_set_prefill_mode(ctx_, prefill);
    return Status::success();
}

bool LlamaBackend::prefill_mode() const {
    return ctx_ ? llama_get_prefill_mode(ctx_) : false;
}

Status LlamaBackend::set_ffn_skip_mask(uint64_t mask) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    if (n_layer() > 64) {
        return Status::error(ErrorCode::Unsupported, "n_layer > 64 is not supported");
    }
    const uint64_t effective = n_layer() >= 64 ? mask : (mask & ((uint64_t(1) << n_layer()) - 1));
    if (effective != mask) {
        log_warn("ffn skip mask truncated to n_layer=" + std::to_string(n_layer()));
    }
    llama_set_prefill_ffn_skip(ctx_, effective);
    return Status::success();
}

uint64_t LlamaBackend::ffn_skip_mask() const {
    return ctx_ ? llama_get_prefill_ffn_skip(ctx_) : 0;
}

Status LlamaBackend::set_attn_top_k(int32_t top_k) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_attn_top_k(ctx_, top_k);
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_attn_top_k rejected top_k=" +
                                 std::to_string(top_k));
    }
    return Status::success();
}

int32_t LlamaBackend::attn_top_k() const {
    return ctx_ ? llama_get_prefill_attn_top_k(ctx_) : 0;
}

Status LlamaBackend::set_attn_indices(const std::vector<int32_t>& indices) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_attn_indices(
            ctx_, indices.empty() ? nullptr : indices.data(), indices.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_attn_indices rejected indices (n=" +
                                 std::to_string(indices.size()) + ")");
    }
    return Status::success();
}

int32_t LlamaBackend::attn_indices_size() const {
    return ctx_ ? llama_get_prefill_attn_indices_size(ctx_) : 0;
}

Status LlamaBackend::set_attn_ranked_rows(const std::vector<int32_t>& rows) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_attn_ranked_rows(
            ctx_, rows.empty() ? nullptr : rows.data(), rows.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_attn_ranked_rows rejected rows (n=" +
                                 std::to_string(rows.size()) + ")");
    }
    return Status::success();
}

Status LlamaBackend::set_attn_head_top_k(const std::vector<int32_t>& matrix) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_attn_head_top_k(
            ctx_, matrix.empty() ? nullptr : matrix.data(), matrix.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_attn_head_top_k rejected matrix (n=" +
                                 std::to_string(matrix.size()) + ")");
    }
    return Status::success();
}

int32_t LlamaBackend::attn_head_top_k_size() const {
    return ctx_ ? llama_get_prefill_attn_head_top_k_size(ctx_) : 0;
}

int32_t LlamaBackend::attn_ranked_rows_size() const {
    return ctx_ ? llama_get_prefill_attn_ranked_rows_size(ctx_) : 0;
}

bool LlamaBackend::attn_per_head_supported() const {
    // This build links against the EdgePrefill-patched llama.cpp, which
    // always provides the per-head C API; unpatched builds fail at link
    // time, so runtime symbol probing is unnecessary.
    return true;
}

Status LlamaBackend::set_attn_pool_blocks(const std::vector<int32_t>& starts,
                                          const std::vector<int32_t>& sizes,
                                          const std::vector<int32_t>& max_pos) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    if (starts.size() != sizes.size() || starts.size() != max_pos.size()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "pool block arrays must have equal length");
    }
    const int32_t rc = llama_set_prefill_attn_pool_blocks(
            ctx_,
            starts.empty() ? nullptr : starts.data(),
            sizes.empty()  ? nullptr : sizes.data(),
            max_pos.empty() ? nullptr : max_pos.data(),
            starts.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_attn_pool_blocks rejected blocks (n=" +
                                 std::to_string(starts.size()) + ")");
    }
    return Status::success();
}

int32_t LlamaBackend::attn_pool_blocks_size() const {
    return ctx_ ? llama_get_prefill_attn_pool_blocks_size(ctx_) : 0;
}

Status LlamaBackend::set_qg_count(int32_t n_groups) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_qg_count(ctx_, n_groups);
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_qg_count rejected n_groups=" +
                                 std::to_string(n_groups));
    }
    return Status::success();
}

Status LlamaBackend::set_qg_rows(int32_t group, const std::vector<int32_t>& rows) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    // Local pre-validation keeps the context free of half-writes (the C
    // API would reject the same inputs, but only after mutation-free
    // checks).
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i > 0 && rows[i] <= rows[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "qg rows must be strictly ascending");
        }
    }
    const int32_t rc = llama_set_prefill_qg_rows(
            ctx_, group, rows.empty() ? nullptr : rows.data(), rows.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_qg_rows rejected group " +
                                 std::to_string(group) + " (n=" +
                                 std::to_string(rows.size()) + ")");
    }
    return Status::success();
}

Status LlamaBackend::set_qg_bounds(const std::vector<int32_t>& bounds) {
    if (!ctx_) return Status::error(ErrorCode::InvalidArgument, "context is not created");
    const int32_t rc = llama_set_prefill_qg_bounds(
            ctx_, bounds.empty() ? nullptr : bounds.data(), bounds.size());
    if (rc != 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "llama_set_prefill_qg_bounds rejected bounds (n=" +
                                 std::to_string(bounds.size()) + ")");
    }
    return Status::success();
}

int32_t LlamaBackend::qg_count() const {
    return ctx_ ? llama_get_prefill_qg_count(ctx_) : 0;
}

int32_t LlamaBackend::n_vocab() const {
    return vocab_ ? llama_vocab_n_tokens(vocab_) : 0;
}

uint32_t LlamaBackend::n_ctx() const {
    return ctx_ ? llama_n_ctx(ctx_) : context_options_.n_ctx;
}

int32_t LlamaBackend::n_layer() const {
    return model_ ? llama_model_n_layer(model_) : 0;
}

int32_t LlamaBackend::n_head() const {
    return model_ ? llama_model_n_head(model_) : 0;
}

std::string LlamaBackend::arch() const {
    if (!model_) return {};
    char buf[128] = {};
    const int32_t len = llama_model_meta_val_str(model_, "general.architecture", buf, sizeof(buf));
    return len > 0 ? std::string(buf, static_cast<size_t>(len)) : std::string();
}

ModelFingerprint LlamaBackend::fingerprint() const {
    ModelFingerprint fp;
    if (!model_) return fp;
    struct stat st {};
    if (!model_path_.empty() && stat(model_path_.c_str(), &st) == 0) {
        fp.file_size = static_cast<uint64_t>(st.st_size);
#if defined(__linux__)
        fp.mtime_ns = static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ull +
                      static_cast<uint64_t>(st.st_mtim.tv_nsec);
#else
        fp.mtime_ns = static_cast<uint64_t>(st.st_mtime) * 1000000000ull;
#endif
        std::ifstream file(model_path_, std::ios::binary);
        if (file) {
            std::vector<uint8_t> head(1024 * 1024);
            file.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
            const auto read = static_cast<size_t>(file.gcount());
            head.resize(read);
            fp.head_hash = hex64(fnv1a(head.data(), head.size()));
        }
    }
    fp.arch = arch();
    fp.n_layer = n_layer();
    fp.n_ctx_train = llama_model_n_ctx_train(model_);
    return fp;
}

uint64_t LlamaBackend::kv_bytes() const {
    return ctx_ ? llama_state_seq_get_size(ctx_, 0) : 0;
}

} // namespace edge_prefill
