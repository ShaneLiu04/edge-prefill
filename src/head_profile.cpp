#include "edge_prefill/head_profile.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "edge_prefill/json_writer.h"

namespace edge_prefill {
namespace {

// Minimal DOM for the profile JSON. The repository parses JSON by hand
// (config.cpp style); a tiny value tree keeps field-name error messages
// uniform instead of threading them through a streaming parser.
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool bool_value = false;
    double number = 0.0;
    bool number_is_int = false;
    int64_t int_value = 0;
    std::string string_value;
    std::vector<JsonValue> array;
    std::vector<std::pair<std::string, JsonValue>> object;

    const JsonValue* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : object) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    // Returns false with a message on the first syntax error.
    bool parse(JsonValue* out, std::string* error) {
        skip_ws();
        if (!parse_value(out)) {
            *error = "invalid JSON at offset " + std::to_string(pos_) + ": " + detail_;
            return false;
        }
        skip_ws();
        if (pos_ != text_.size()) {
            *error = "trailing content at offset " + std::to_string(pos_);
            return false;
        }
        return true;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
    std::string detail_;

    void skip_ws() {
        while (pos_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool fail(const char* detail) {
        detail_ = detail;
        return false;
    }

    bool peek(char c) const { return pos_ < text_.size() && text_[pos_] == c; }

    bool consume(char c) {
        if (!peek(c)) return fail("expected char");
        ++pos_;
        return true;
    }

    bool parse_value(JsonValue* out) {
        if (pos_ >= text_.size()) return fail("unexpected end");
        const char c = text_[pos_];
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == '"') {
            out->type = JsonValue::Type::String;
            return parse_string(&out->string_value);
        }
        if (c == 't' || c == 'f') return parse_bool(out);
        if (c == 'n') return parse_null(out);
        return parse_number(out);
    }

    bool parse_object(JsonValue* out) {
        out->type = JsonValue::Type::Object;
        if (!consume('{')) return false;
        skip_ws();
        if (consume('}')) return true;
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws();
            if (!consume(':')) return false;
            skip_ws();
            JsonValue value;
            if (!parse_value(&value)) return false;
            out->object.emplace_back(std::move(key), std::move(value));
            skip_ws();
            if (consume(',')) continue;
            return consume('}');
        }
    }

    bool parse_array(JsonValue* out) {
        out->type = JsonValue::Type::Array;
        if (!consume('[')) return false;
        skip_ws();
        if (consume(']')) return true;
        while (true) {
            skip_ws();
            JsonValue value;
            if (!parse_value(&value)) return false;
            out->array.push_back(std::move(value));
            skip_ws();
            if (consume(',')) continue;
            return consume(']');
        }
    }

    bool parse_string(std::string* out) {
        if (!consume('"')) return false;
        std::string result;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                *out = std::move(result);
                return true;
            }
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) return fail("unterminated escape");
            const char esc = text_[pos_++];
            switch (esc) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: return fail("unsupported escape");
            }
        }
        return fail("unterminated string");
    }

    bool parse_bool(JsonValue* out) {
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out->type = JsonValue::Type::Bool;
            out->bool_value = true;
            return true;
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out->type = JsonValue::Type::Bool;
            out->bool_value = false;
            return true;
        }
        return fail("invalid literal");
    }

    bool parse_null(JsonValue* out) {
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out->type = JsonValue::Type::Null;
            return true;
        }
        return fail("invalid literal");
    }

    bool parse_number(JsonValue* out) {
        const size_t begin = pos_;
        if (peek('-')) ++pos_;
        bool has_digits = false;
        bool is_int = true;
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c))) {
                has_digits = true;
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                is_int = false;
                ++pos_;
            } else {
                break;
            }
        }
        if (!has_digits) return fail("invalid number");
        const std::string token = text_.substr(begin, pos_ - begin);
        out->type = JsonValue::Type::Number;
        out->number_is_int = is_int;
        if (is_int) {
            out->int_value = std::strtoll(token.c_str(), nullptr, 10);
            out->number = static_cast<double>(out->int_value);
        } else {
            out->number = std::strtod(token.c_str(), nullptr);
        }
        return true;
    }
};

const char* json_type_name(JsonValue::Type type) {
    switch (type) {
        case JsonValue::Type::Null: return "null";
        case JsonValue::Type::Bool: return "bool";
        case JsonValue::Type::Number: return "number";
        case JsonValue::Type::String: return "string";
        case JsonValue::Type::Array: return "array";
        case JsonValue::Type::Object: return "object";
    }
    return "unknown";
}

Status type_error(const std::string& field, JsonValue::Type actual,
                  const char* expected) {
    return Status::error(ErrorCode::InvalidArgument,
                         "field " + field + ": expected " + expected +
                             ", got " + json_type_name(actual));
}

Status require_string(const JsonValue& parent, const std::string& key,
                      std::string* out) {
    const JsonValue* value = parent.find(key);
    if (!value) {
        return Status::error(ErrorCode::InvalidArgument, "missing field: " + key);
    }
    if (value->type != JsonValue::Type::String) return type_error(key, value->type, "string");
    *out = value->string_value;
    return Status::success();
}

Status require_int(const JsonValue& parent, const std::string& key, int64_t* out) {
    const JsonValue* value = parent.find(key);
    if (!value) {
        return Status::error(ErrorCode::InvalidArgument, "missing field: " + key);
    }
    if (value->type != JsonValue::Type::Number || !value->number_is_int) {
        return type_error(key, value->type, "integer");
    }
    *out = value->int_value;
    return Status::success();
}

Status require_uint32(const JsonValue& parent, const std::string& key, uint32_t* out) {
    int64_t value = 0;
    auto st = require_int(parent, key, &value);
    if (!st.ok()) return st;
    if (value < 0 || value > 0x7fffffffll) {
        return Status::error(ErrorCode::InvalidArgument,
                             "field " + key + ": out of uint32 range");
    }
    *out = static_cast<uint32_t>(value);
    return Status::success();
}

Status require_double(const JsonValue& parent, const std::string& key, double* out) {
    const JsonValue* value = parent.find(key);
    if (!value) {
        return Status::error(ErrorCode::InvalidArgument, "missing field: " + key);
    }
    if (value->type != JsonValue::Type::Number) return type_error(key, value->type, "number");
    *out = value->number;
    return Status::success();
}

Status parse_probe_params(const JsonValue& root, HeadProbeParams* out) {
    const JsonValue* probe = root.find("probe_params");
    if (!probe) {
        return Status::error(ErrorCode::InvalidArgument, "missing field: probe_params");
    }
    if (probe->type != JsonValue::Type::Object) {
        return type_error("probe_params", probe->type, "object");
    }
    auto st = require_uint32(*probe, "block_len", &out->block_len);
    if (!st.ok()) return st;
    st = require_uint32(*probe, "repeats", &out->repeats);
    if (!st.ok()) return st;
    st = require_uint32(*probe, "sink", &out->sink);
    if (!st.ok()) return st;
    st = require_uint32(*probe, "recent", &out->recent);
    if (!st.ok()) return st;
    st = require_uint32(*probe, "probe_k", &out->probe_k);
    if (!st.ok()) return st;
    int64_t seed = 0;
    st = require_int(*probe, "seed", &seed);
    if (!st.ok()) return st;
    if (seed < 0) {
        return Status::error(ErrorCode::InvalidArgument, "field seed: negative");
    }
    out->seed = static_cast<uint64_t>(seed);
    return Status::success();
}

Status parse_head_entry(const JsonValue& entry, HeadProfileEntry* out) {
    if (entry.type != JsonValue::Type::Object) {
        return type_error("heads[]", entry.type, "object");
    }
    int64_t layer = 0;
    auto st = require_int(entry, "layer", &layer);
    if (!st.ok()) return st;
    int64_t head = 0;
    st = require_int(entry, "head", &head);
    if (!st.ok()) return st;
    st = require_double(entry, "echo_score", &out->echo_score);
    if (!st.ok()) return st;
    st = require_double(entry, "induction_score", &out->induction_score);
    if (!st.ok()) return st;
    std::string head_class;
    st = require_string(entry, "class", &head_class);
    if (!st.ok()) return st;
    if (head_class == "retrieval") {
        out->retrieval = true;
    } else if (head_class == "normal") {
        out->retrieval = false;
    } else {
        return Status::error(ErrorCode::InvalidArgument,
                             "field class: unknown head class '" + head_class + "'");
    }
    out->layer = static_cast<int32_t>(layer);
    out->head = static_cast<int32_t>(head);
    return Status::success();
}

} // namespace

Status load_head_profile(const std::string& path, HeadProfile* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null output");

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Status::error(ErrorCode::Io, "cannot open head profile: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    JsonValue root;
    std::string error;
    JsonParser parser(text);
    if (!parser.parse(&root, &error)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "malformed head profile: " + error);
    }
    if (root.type != JsonValue::Type::Object) {
        return type_error("<root>", root.type, "object");
    }

    HeadProfile profile;
    auto st = require_string(root, "model_sha", &profile.model_sha);
    if (!st.ok()) return st;
    int64_t n_layer = 0;
    st = require_int(root, "n_layer", &n_layer);
    if (!st.ok()) return st;
    int64_t n_head = 0;
    st = require_int(root, "n_head", &n_head);
    if (!st.ok()) return st;
    profile.n_layer = static_cast<int32_t>(n_layer);
    profile.n_head = static_cast<int32_t>(n_head);

    st = parse_probe_params(root, &profile.probe);
    if (!st.ok()) return st;

    const JsonValue* heads = root.find("heads");
    if (!heads) {
        return Status::error(ErrorCode::InvalidArgument, "missing field: heads");
    }
    if (heads->type != JsonValue::Type::Array) {
        return type_error("heads", heads->type, "array");
    }
    profile.heads.reserve(heads->array.size());
    for (const auto& entry : heads->array) {
        HeadProfileEntry parsed;
        st = parse_head_entry(entry, &parsed);
        if (!st.ok()) return st;
        profile.heads.push_back(parsed);
    }

    *out = std::move(profile);
    return Status::success();
}

Status save_head_profile(const std::string& path, const HeadProfile& profile) {
    std::string text;
    JsonWriter writer(&text);
    writer.begin_object();
    writer.key("model_sha");
    writer.value(profile.model_sha);
    writer.key("n_layer");
    writer.value(static_cast<uint64_t>(profile.n_layer));
    writer.key("n_head");
    writer.value(static_cast<uint64_t>(profile.n_head));
    writer.key("probe_params");
    writer.begin_object();
    writer.key("block_len");
    writer.value(static_cast<uint64_t>(profile.probe.block_len));
    writer.key("repeats");
    writer.value(static_cast<uint64_t>(profile.probe.repeats));
    writer.key("sink");
    writer.value(static_cast<uint64_t>(profile.probe.sink));
    writer.key("recent");
    writer.value(static_cast<uint64_t>(profile.probe.recent));
    writer.key("probe_k");
    writer.value(static_cast<uint64_t>(profile.probe.probe_k));
    writer.key("seed");
    writer.value(profile.probe.seed);
    writer.end_object();
    writer.key("heads");
    writer.begin_array();
    for (const HeadProfileEntry& entry : profile.heads) {
        writer.begin_object();
        writer.key("layer");
        writer.value(static_cast<uint64_t>(entry.layer));
        writer.key("head");
        writer.value(static_cast<uint64_t>(entry.head));
        writer.key("echo_score");
        writer.value(entry.echo_score);
        writer.key("induction_score");
        writer.value(entry.induction_score);
        writer.key("class");
        writer.value(std::string(entry.retrieval ? "retrieval" : "normal"));
        writer.end_object();
    }
    writer.end_array();
    writer.end_object();
    text.push_back('\n');

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return Status::error(ErrorCode::Io, "cannot write head profile: " + path);
    }
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    file.flush();
    if (!file) {
        return Status::error(ErrorCode::Io, "failed writing head profile: " + path);
    }
    return Status::success();
}

Status validate_head_profile(const HeadProfile& profile, int32_t n_layer,
                             int32_t n_head) {
    if (profile.n_layer != n_layer) {
        return Status::error(ErrorCode::InvalidArgument,
                             "head profile n_layer mismatch: profile=" +
                                 std::to_string(profile.n_layer) + " model=" +
                                 std::to_string(n_layer));
    }
    if (profile.n_head != n_head) {
        return Status::error(ErrorCode::InvalidArgument,
                             "head profile n_head mismatch: profile=" +
                                 std::to_string(profile.n_head) + " model=" +
                                 std::to_string(n_head));
    }
    for (const auto& entry : profile.heads) {
        if (entry.layer < 0 || entry.layer >= n_layer) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "head profile entry layer out of range: " +
                                     std::to_string(entry.layer));
        }
        if (entry.head < 0 || entry.head >= n_head) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "head profile entry head out of range: " +
                                     std::to_string(entry.head));
        }
    }
    return Status::success();
}

Status validate_calibration_text(const std::string& text, uint32_t block_len) {
    if (block_len == 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "probe block_len must be >= 1");
    }
    for (unsigned char c : text) {
        if (!std::isspace(c)) return Status::success();
    }
    return Status::error(ErrorCode::InvalidArgument,
                         "calibration text is empty or whitespace-only");
}

} // namespace edge_prefill
