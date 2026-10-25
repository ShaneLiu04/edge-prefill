#include "edge_prefill/config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>

#include "edge_prefill/json_writer.h"

namespace edge_prefill {
namespace {

std::string trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

bool parse_u64(const std::string& text, uint64_t* out) {
    try {
        size_t pos = 0;
        const auto value = std::stoull(text, &pos, 10);
        if (pos != text.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_i64(const std::string& text, int64_t* out) {
    try {
        size_t pos = 0;
        const auto value = std::stoll(text, &pos, 10);
        if (pos != text.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_double(const std::string& text, double* out) {
    try {
        size_t pos = 0;
        const auto value = std::stod(text, &pos);
        if (pos != text.size()) return false;
        *out = value;
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bool(const std::string& text, bool* out) {
    if (text == "1" || text == "true" || text == "yes" || text == "on") {
        *out = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "no" || text == "off") {
        *out = false;
        return true;
    }
    return false;
}

} // namespace

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::Latency: return "latency";
        case Mode::Balanced: return "balanced";
        case Mode::Quality: return "quality";
        case Mode::Reuse: return "reuse";
        case Mode::Dense: return "dense";
    }
    return "unknown";
}

const char* kv_precision_name(KvPrecision precision) {
    switch (precision) {
        case KvPrecision::F16: return "f16";
        case KvPrecision::Q8_0: return "q8_0";
        case KvPrecision::Q4_0: return "q4_0";
    }
    return "unknown";
}

Status parse_kv_precision(const std::string& text, KvPrecision* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null output");
    if (text == "f16") {
        *out = KvPrecision::F16;
        return Status::success();
    }
    if (text == "q8_0") {
        *out = KvPrecision::Q8_0;
        return Status::success();
    }
    if (text == "q4_0") {
        *out = KvPrecision::Q4_0;
        return Status::success();
    }
    return Status::error(ErrorCode::InvalidArgument, "invalid-kv-type: " + text);
}

Status parse_mode(const std::string& text, Mode* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null output");
    if (text == "latency") *out = Mode::Latency;
    else if (text == "balanced") *out = Mode::Balanced;
    else if (text == "quality") *out = Mode::Quality;
    else if (text == "reuse") *out = Mode::Reuse;
    else if (text == "dense") *out = Mode::Dense;
    else return Status::error(ErrorCode::InvalidArgument, "invalid mode: " + text);
    return Status::success();
}

Status parse_u32_list(const std::string& text, std::vector<uint32_t>* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null output");
    out->clear();
    std::string item;
    std::stringstream stream(text);
    while (std::getline(stream, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            return Status::error(ErrorCode::InvalidArgument, "empty list element");
        }
        uint64_t u = 0;
        if (!parse_u64(item, &u) || u > 0xffffffffull) {
            return Status::error(ErrorCode::InvalidArgument, "invalid list element: " + item);
        }
        out->push_back(static_cast<uint32_t>(u));
    }
    if (out->empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty list");
    }
    return Status::success();
}

std::string Plan::to_json() const {
    std::string out;
    JsonWriter json(&out);
    json.begin_object();
    json.key("passes");
    json.begin_array();
    for (const auto& pass : passes) json.value(pass);
    json.end_array();
    json.key("budget");
    json.value(budget);
    json.key("block_size");
    json.value(static_cast<uint64_t>(block_size));
    json.key("chunk_size");
    json.value(static_cast<uint64_t>(chunk_size));
    json.key("skip_ffn_mask");
    json.value(skip_ffn_mask);
    json.key("kv_type");
    json.value(kv_precision_name(kv_type));
    json.key("attention_top_k");
    json.value(static_cast<uint64_t>(attention_top_k));
    json.key("dense_fallback");
    json.value(dense_fallback);
    json.key("budget_source");
    json.value(budget_source);
    json.key("reasons");
    json.begin_array();
    for (const auto& reason : reasons) json.value(reason);
    json.end_array();
    json.end_object();
    return out;
}

Status parse_config_file(const std::string& path, Config* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null config");
    std::ifstream file(path);
    if (!file) return Status::error(ErrorCode::Io, "cannot open config: " + path);

    std::string line;
    uint32_t line_no = 0;
    std::map<std::string, std::string> seen;
    while (std::getline(file, line)) {
        ++line_no;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "config line " + std::to_string(line_no) + ": missing '='");
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        const auto seen_it = seen.find(key);
        if (seen_it != seen.end() && seen_it->second != value) {
            out->warnings.push_back("duplicate-key:" + key);
        }
        seen[key] = value;
        const auto warn_duplicate = [&](const std::string&) {};

        uint64_t u = 0;
        int64_t i = 0;
        double d = 0.0;
        bool b = false;
        if (key == "model") {
            warn_duplicate(out->model_path);
            out->model_path = value;
        } else if (key == "draft_model") {
            out->draft_model_path = value;
        } else if (key == "embedding_model") {
            out->embedding_model = value;
        } else if (key == "selector_w_lexical") {
            if (!parse_double(value, &d)) goto bad_value;
            out->weights.lexical = d;
            out->weights_overridden = true;
        } else if (key == "selector_w_diversity") {
            if (!parse_double(value, &d)) goto bad_value;
            out->weights.diversity = d;
            out->weights_overridden = true;
        } else if (key == "selector_w_position") {
            if (!parse_double(value, &d)) goto bad_value;
            out->weights.position = d;
            out->weights_overridden = true;
        } else if (key == "selector_w_draft") {
            if (!parse_double(value, &d)) goto bad_value;
            out->weights.draft = d;
            out->weights_overridden = true;
        } else if (key == "data_dir") {
            out->data_dir = value;
        } else if (key == "results_dir") {
            out->results_dir = value;
        } else if (key == "prompt") {
            out->prompt = value;
        } else if (key == "prompt_file") {
            out->prompt_file = value;
        } else if (key == "n_ctx") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->n_ctx = static_cast<uint32_t>(u);
        } else if (key == "threads") {
            if (!parse_i64(value, &i)) goto bad_value;
            out->n_threads = static_cast<int32_t>(i);
        } else if (key == "threads_batch") {
            if (!parse_i64(value, &i)) goto bad_value;
            out->n_threads_batch = static_cast<int32_t>(i);
        } else if (key == "n_ubatch") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->n_ubatch = static_cast<uint32_t>(u);
        } else if (key == "n_gpu_layers") {
            if (!parse_i64(value, &i)) goto bad_value;
            out->n_gpu_layers = static_cast<int32_t>(i);
        } else if (key == "max_new_tokens") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->max_new_tokens = static_cast<uint32_t>(u);
        } else if (key == "block_size") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->block_size = static_cast<uint32_t>(u);
        } else if (key == "chunk_size") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->chunk_size = static_cast<uint32_t>(u);
        } else if (key == "min_dense_tokens") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->min_dense_tokens = static_cast<uint32_t>(u);
        } else if (key == "sink_blocks") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->sink_blocks = static_cast<uint32_t>(u);
        } else if (key == "recent_blocks") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->recent_blocks = static_cast<uint32_t>(u);
        } else if (key == "coverage_stride") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->coverage_stride = static_cast<uint32_t>(u);
        } else if (key == "anchor_ratio") {
            if (!parse_double(value, &d) || !(d >= 0.05 && d <= 1.0)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "anchor_ratio must be in [0.05,1]");
            }
            out->anchor_ratio = d;
        } else if (key == "anchor_stride_scale") {
            if (!parse_double(value, &d) || !(d >= 0.0 && d <= 8.0)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "anchor_stride_scale must be in [0,8]");
            }
            out->anchor_stride_scale = d;
        } else if (key == "mmr_threshold") {
            if (!parse_double(value, &d) || !(d >= 0.0 && d <= 1.0)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "mmr_threshold must be in [0,1]");
            }
            out->mmr_threshold = d;
        } else if (key == "mmr_lambda") {
            if (!parse_double(value, &d) || !(d >= 0.0 && d <= 1.0)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "mmr_lambda must be in [0,1]");
            }
            out->mmr_lambda = d;
        } else if (key == "budget_curve") {
            out->budget_curve = value;
        } else if (key == "repeats") {
            if (!parse_i64(value, &i)) goto bad_value;
            if (i < 0) return Status::error(ErrorCode::InvalidArgument, "negative repeats");
            if (i == 0) {
                out->warnings.push_back("repeats-defaulted");
                i = 5;
            }
            out->repeats = static_cast<uint32_t>(i);
        } else if (key == "warmup") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->warmup = static_cast<uint32_t>(u);
        } else if (key == "budget") {
            if (!parse_double(value, &d)) goto bad_value;
            out->budget = d;
            out->budget_overridden = true;
        } else if (key == "mode") {
            auto st = parse_mode(value, &out->mode);
            if (!st.ok()) return st;
        } else if (key == "kv_type") {
            auto st = parse_kv_precision(value, &out->kv_type);
            if (!st.ok()) return st;
            out->prefill_kv_type = out->kv_type;
            out->decode_kv_type = out->kv_type;
            out->kv_overridden = true;
        } else if (key == "prefill_kv_type") {
            auto st = parse_kv_precision(value, &out->prefill_kv_type);
            if (!st.ok()) return st;
        } else if (key == "decode_kv_type") {
            auto st = parse_kv_precision(value, &out->decode_kv_type);
            if (!st.ok()) return st;
        } else if (key == "skip_ffn") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->skip_ffn_mask = u;
            out->skip_ffn_overridden = true;
        } else if (key == "attn_top_k") {
            if (!parse_u64(value, &u) || u > 0xffffffffull) goto bad_value;
            out->attn_top_k = static_cast<uint32_t>(u);
        } else if (key == "attn_top_k_layers") {
            out->attn_top_k_layers = value;
        } else if (key == "head_profile") {
            out->head_profile_path = value;
        } else if (key == "head_adaptive") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->head_adaptive = b;
        } else if (key == "head_top_k") {
            if (!parse_u64(value, &u) || u > 0xffffffffull) goto bad_value;
            out->head_top_k = static_cast<uint32_t>(u);
        } else if (key == "prefix_cache") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->enable_prefix_cache = b;
        } else if (key == "prefix_cache_bytes") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->prefix_cache_bytes = u;
        } else if (key == "ffn_skip") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->enable_ffn_skip = b;
        } else if (key == "chunking") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->enable_chunking = b;
        } else if (key == "chat_template") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->use_chat_template = b;
        } else if (key == "seed") {
            if (!parse_u64(value, &u)) goto bad_value;
            out->seed = u;
        } else if (key == "head_pattern_routing") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->head_pattern_routing = b;
        } else if (key == "pyramid_prior") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->pyramid_prior = b;
        } else if (key == "pyramid_k_min") {
            if (!parse_u64(value, &u) || u > 0xffffffffull) goto bad_value;
            out->pyramid_k_min = static_cast<uint32_t>(u);
        } else if (key == "pyramid_k_max") {
            if (!parse_u64(value, &u) || u > 0xffffffffull) goto bad_value;
            out->pyramid_k_max = static_cast<uint32_t>(u);
        } else if (key == "pool_branches") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->pool_branches = b;
        } else if (key == "query_gating") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->query_gating = b;
        } else if (key == "query_chunk_size") {
            if (!parse_u64(value, &u) || u == 0 || u > 0xffffffffull) goto bad_value;
            out->query_chunk_size = static_cast<uint32_t>(u);
        } else if (key == "query_chunk_max") {
            if (!parse_u64(value, &u) || u == 0 || u > 16) goto bad_value;
            out->query_chunk_max = static_cast<uint32_t>(u);
        } else if (key == "query_top_k") {
            if (!parse_u64(value, &u) || u > 0xffffffffull) goto bad_value;
            out->query_top_k_per_group = static_cast<uint32_t>(u);
        } else if (key == "qg_bound") {
            if (!parse_bool(value, &b)) goto bad_value;
            out->qg_bound = b;
        } else if (key == "qg_sel_pages") {
            if (!parse_u64(value, &u) || u > 128) goto bad_value;
            out->qg_sel_pages = static_cast<uint32_t>(u);
        } else if (key == "qg_local_rows") {
            if (!parse_u64(value, &u) || u > 32768) goto bad_value;
            out->qg_local_rows = static_cast<uint32_t>(u);
        } else if (key == "qg_pool_pages") {
            if (!parse_u64(value, &u) || u > 4096) goto bad_value;
            out->qg_pool_pages = static_cast<uint32_t>(u);
        } else {
            return Status::error(ErrorCode::InvalidArgument, "unknown config key: " + key);
        }
        continue;
    bad_value:
        return Status::error(ErrorCode::InvalidArgument,
                             "invalid value for " + key + ": " + value);
    }

    return Status::success();
}

} // namespace edge_prefill
