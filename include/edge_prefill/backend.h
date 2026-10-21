#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/chunk_scheduler.h"
#include "edge_prefill/config.h"
#include "edge_prefill/status.h"

namespace edge_prefill {

struct ContextOptions {
    uint32_t n_ctx = 0;
    uint32_t n_batch = 2048;
    uint32_t n_ubatch = 512;
    int32_t n_threads = 0;
    int32_t n_threads_batch = 0;
    uint32_t n_seq_max = 1;
    KvPrecision type_k = KvPrecision::F16;
    KvPrecision type_v = KvPrecision::F16;
    bool flash_attn = true;
    bool offload_kqv = true;
    bool embeddings = false;
    int32_t n_gpu_layers = 0;
};

struct PrefillStats {
    uint32_t prompt_tokens = 0;
    uint32_t selected_tokens = 0;
    uint32_t chunks = 0;
    uint64_t total_us = 0;
    uint64_t ttft_us = 0;
    int32_t kv_pos_min = -1;
    int32_t kv_pos_max = -1;
    std::vector<ChunkStats> chunk_stats;
    uint64_t kv_bytes = 0;
    bool collect_chunk_stats = true;
};

struct ModelFingerprint {
    uint64_t file_size = 0;
    uint64_t mtime_ns = 0;
    std::string head_hash;
    std::string arch;
    int32_t n_layer = 0;
    int32_t n_ctx_train = 0;
    std::string to_key() const;
};

class LlamaBackend {
public:
    LlamaBackend() = default;
    ~LlamaBackend();
    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    static Status init();
    static void shutdown();

    Status load_model(const std::string& path, bool use_mmap = true);
    Status create_context(const ContextOptions& opts);
    Status tokenize(const std::string& text, std::vector<llama_token>* out,
                    bool add_special = true, bool parse_special = true) const;
    Status token_to_piece(llama_token token, std::string* out) const;

    Status prefill(const std::vector<llama_token>& tokens,
                   const std::vector<llama_pos>& positions,
                   uint32_t chunk_size,
                   PrefillStats* stats);
    Status decode(llama_token token, llama_pos pos, std::vector<float>* logits);
    Status save_state(std::vector<uint8_t>* out) const;
    Status load_state(const std::vector<uint8_t>& in);
    Status reset_sequence();

    Status set_prefill_mode(bool prefill);
    bool prefill_mode() const;
    Status set_ffn_skip_mask(uint64_t mask);
    uint64_t ffn_skip_mask() const;

    // Attention-level top-k sparse prefill (requires the llama.cpp
    // llama-prefill-attn-top-k patch). top_k == 0 disables gathering and
    // restores the original dense attention path.
    Status set_attn_top_k(int32_t top_k);
    int32_t attn_top_k() const;
    // indices must be strictly ascending and < n_ctx; empty disables.
    Status set_attn_indices(const std::vector<int32_t>& indices);
    int32_t attn_indices_size() const;

    const llama_model* model() const { return model_; }
    llama_context* context() const { return ctx_; }
    const llama_vocab* vocab() const { return vocab_; }
    int32_t n_vocab() const;
    uint32_t n_ctx() const;
    int32_t n_layer() const;
    std::string arch() const;
    ModelFingerprint fingerprint() const;
    const ContextOptions& context_options() const { return context_options_; }
    uint64_t kv_bytes() const;

private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    std::string model_path_;
    ContextOptions context_options_;
};

} // namespace edge_prefill
