#include "edge_prefill/json_writer.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace edge_prefill {

void write_json_string(std::string* out, const std::string& value) {
    out->push_back('"');
    for (char c : value) {
        switch (c) {
            case '"': out->append("\\\""); break;
            case '\\': out->append("\\\\"); break;
            case '\n': out->append("\\n"); break;
            case '\r': out->append("\\r"); break;
            case '\t': out->append("\\t"); break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out->append(buf);
                } else {
                    out->push_back(c);
                }
        }
    }
    out->push_back('"');
}

JsonWriter::JsonWriter(std::string* out) : out_(out) {
    if (!out_) {
        static std::string sink;
        out_ = &sink;
    }
}

void JsonWriter::before_value() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (!first_.empty()) {
        if (!first_.back()) out_->push_back(',');
        first_.back() = false;
    }
}

void JsonWriter::begin_object() {
    before_value();
    out_->push_back('{');
    first_.push_back(true);
}

void JsonWriter::end_object() {
    out_->push_back('}');
    if (!first_.empty()) first_.pop_back();
}

void JsonWriter::key(const std::string& key) {
    before_value();
    write_json_string(out_, key);
    out_->push_back(':');
    after_key_ = true;
}

void JsonWriter::value(const std::string& value) {
    before_value();
    write_json_string(out_, value);
}

void JsonWriter::value(double value) {
    before_value();
    if (std::isnan(value) || std::isinf(value)) {
        out_->append("null");
        return;
    }
    std::ostringstream ss;
    ss << std::setprecision(17) << value;
    out_->append(ss.str());
}

void JsonWriter::value(uint64_t value) {
    before_value();
    out_->append(std::to_string(value));
}

void JsonWriter::value(bool value) {
    before_value();
    out_->append(value ? "true" : "false");
}

void JsonWriter::begin_array() {
    before_value();
    out_->push_back('[');
    first_.push_back(true);
}

void JsonWriter::end_array() {
    out_->push_back(']');
    if (!first_.empty()) first_.pop_back();
}

} // namespace edge_prefill
