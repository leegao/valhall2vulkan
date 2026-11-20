#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace valhall::spirv2valhall {

struct JsonValue {
    enum class Kind { Null, Bool, Int, Double, String, Array, Object };

    Kind kind = Kind::Null;
    bool bool_val = false;
    int64_t int_val = 0;
    double double_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    bool is_null() const { return kind == Kind::Null; }
    bool is_bool() const { return kind == Kind::Bool; }
    bool is_int() const { return kind == Kind::Int; }
    bool is_number() const { return kind == Kind::Int || kind == Kind::Double; }
    bool is_string() const { return kind == Kind::String; }
    bool is_array() const { return kind == Kind::Array; }
    bool is_object() const { return kind == Kind::Object; }

    const JsonValue* get(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        auto it = obj_val.find(key);
        return (it == obj_val.end()) ? nullptr : &it->second;
    }

    std::string get_string(const std::string& key, const std::string& fallback = "") const {
        const JsonValue* v = get(key);
        return (v && v->is_string()) ? v->str_val : fallback;
    }

    int64_t get_int(const std::string& key, int64_t fallback = 0) const {
        const JsonValue* v = get(key);
        if (!v) return fallback;
        if (v->is_int()) return v->int_val;
        if (v->kind == Kind::Double) return static_cast<int64_t>(v->double_val);
        return fallback;
    }

    bool get_bool(const std::string& key, bool fallback = false) const {
        const JsonValue* v = get(key);
        return (v && v->is_bool()) ? v->bool_val : fallback;
    }
};

/// Parse a UTF-8 JSON document into a JsonValue tree.
bool parse_json(const std::string& text, JsonValue* out, std::string* err = nullptr);

}  // namespace valhall::spirv2valhall
