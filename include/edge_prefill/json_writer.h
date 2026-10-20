#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace edge_prefill {

class JsonWriter {
public:
    explicit JsonWriter(std::string* out);

    void begin_object();
    void end_object();
    void key(const std::string& key);
    void value(const std::string& value);
    void value(const char* text) { value(std::string(text ? text : "")); }
    void value(double value);
    void value(uint64_t value);
    void value(bool value);
    void begin_array();
    void end_array();
    std::string str() const { return *out_; }

private:
    void before_value();
    void escape(const std::string& value);

    std::string* out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

void write_json_string(std::string* out, const std::string& value);

} // namespace edge_prefill
