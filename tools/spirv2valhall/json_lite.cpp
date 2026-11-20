#include "json_lite.h"

#include <cctype>
#include <cstdlib>

namespace valhall::spirv2valhall {

namespace {

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    bool parse(JsonValue* out, std::string* err) {
        skip_ws();
        if (!parse_value(out, err)) return false;
        skip_ws();
        if (pos_ != s_.size()) {
            fail(err, "Trailing characters after JSON root value");
            return false;
        }
        return true;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    void fail(std::string* err, const std::string& msg) const {
        if (!err) return;
        size_t line = 1, col = 1;
        for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
            if (s_[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        *err = "JSON parse error at " + std::to_string(line) + ":" + std::to_string(col) + ": " + msg;
    }

    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool parse_value(JsonValue* out, std::string* err) {
        skip_ws();
        if (pos_ >= s_.size()) {
            fail(err, "Unexpected end of input");
            return false;
        }
        char c = s_[pos_];
        if (c == '{') return parse_object(out, err);
        if (c == '[') return parse_array(out, err);
        if (c == '"') {
            out->kind = JsonValue::Kind::String;
            return parse_string(&out->str_val, err);
        }
        if (c == 't' && s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out->kind = JsonValue::Kind::Bool;
            out->bool_val = true;
            return true;
        }
        if (c == 'f' && s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out->kind = JsonValue::Kind::Bool;
            out->bool_val = false;
            return true;
        }
        if (c == 'n' && s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out->kind = JsonValue::Kind::Null;
            return true;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number(out, err);
        }
        fail(err, std::string("Unexpected character '") + c + "'");
        return false;
    }

    bool parse_string(std::string* out, std::string* err) {
        if (pos_ >= s_.size() || s_[pos_] != '"') {
            fail(err, "Expected '\"'");
            return false;
        }
        ++pos_;
        out->clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (pos_ >= s_.size()) {
                    fail(err, "Unterminated escape sequence");
                    return false;
                }
                char e = s_[pos_++];
                switch (e) {
                    case '"': out->push_back('"'); break;
                    case '\\': out->push_back('\\'); break;
                    case '/': out->push_back('/'); break;
                    case 'b': out->push_back('\b'); break;
                    case 'f': out->push_back('\f'); break;
                    case 'n': out->push_back('\n'); break;
                    case 'r': out->push_back('\r'); break;
                    case 't': out->push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > s_.size()) {
                            fail(err, "Truncated \\uXXXX escape");
                            return false;
                        }
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = s_[pos_++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
                            else {
                                fail(err, "Invalid hex digit in \\uXXXX");
                                return false;
                            }
                        }
                        if (code < 0x80) {
                            out->push_back(static_cast<char>(code));
                        } else if (code < 0x800) {
                            out->push_back(static_cast<char>(0xC0 | (code >> 6)));
                            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        } else {
                            out->push_back(static_cast<char>(0xE0 | (code >> 12)));
                            out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                            out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
                        }
                        break;
                    }
                    default:
                        fail(err, std::string("Unknown escape '\\") + e + "'");
                        return false;
                }
            } else {
                out->push_back(c);
            }
        }
        fail(err, "Unterminated string literal");
        return false;
    }

    bool parse_number(JsonValue* out, std::string* err) {
        size_t start = pos_;
        if (s_[pos_] == '-') ++pos_;
        bool is_float = false;
        if (pos_ >= s_.size() || !std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            fail(err, "Invalid number");
            return false;
        }
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        if (pos_ < s_.size() && s_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        }
        std::string num_str = s_.substr(start, pos_ - start);
        if (is_float) {
            out->kind = JsonValue::Kind::Double;
            out->double_val = std::strtod(num_str.c_str(), nullptr);
        } else {
            out->kind = JsonValue::Kind::Int;
            out->int_val = std::strtoll(num_str.c_str(), nullptr, 10);
        }
        return true;
    }

    bool parse_array(JsonValue* out, std::string* err) {
        ++pos_;  // '['
        out->kind = JsonValue::Kind::Array;
        out->arr_val.clear();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            JsonValue elem;
            if (!parse_value(&elem, err)) return false;
            out->arr_val.push_back(std::move(elem));
            skip_ws();
            if (pos_ >= s_.size()) {
                fail(err, "Unterminated array");
                return false;
            }
            if (s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            if (s_[pos_] != ',') {
                fail(err, "Expected ',' or ']' in array");
                return false;
            }
            ++pos_;
        }
    }

    bool parse_object(JsonValue* out, std::string* err) {
        ++pos_;  // '{'
        out->kind = JsonValue::Kind::Object;
        out->obj_val.clear();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            return true;
        }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(&key, err)) return false;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') {
                fail(err, "Expected ':' after object key");
                return false;
            }
            ++pos_;
            JsonValue val;
            if (!parse_value(&val, err)) return false;
            out->obj_val[std::move(key)] = std::move(val);
            skip_ws();
            if (pos_ >= s_.size()) {
                fail(err, "Unterminated object");
                return false;
            }
            if (s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            if (s_[pos_] != ',') {
                fail(err, "Expected ',' or '}' in object");
                return false;
            }
            ++pos_;
        }
    }
};

}  // namespace

bool parse_json(const std::string& text, JsonValue* out, std::string* err) {
    if (!out) {
        if (err) *err = "Null JsonValue output pointer";
        return false;
    }
    JsonParser parser(text);
    return parser.parse(out, err);
}

}  // namespace valhall::spirv2valhall
