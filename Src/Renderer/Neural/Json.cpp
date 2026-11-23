#include "Renderer/Neural/Json.h"


#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace Evans::Neural {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}
    Json parse() {
        Json result = value(0);
        whitespace();
        if (position_ != text_.size()) fail("unexpected trailing input");
        return result;
    }
private:
    const std::string& text_;
    size_t position_ = 0;
    [[noreturn]] void fail(const std::string& message) const {
        size_t line = 1, column = 1;
        for (size_t i = 0; i < position_; ++i) {
            if (text_[i] == '\n') { ++line; column = 1; } else ++column;
        }
        throw std::runtime_error("JSON " + std::to_string(line) + ":" + std::to_string(column) + ": " + message);
    }
    void whitespace() {
        while (position_ < text_.size() && (text_[position_] == ' ' || text_[position_] == '\n' || text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }
    bool consume(char c) {
        whitespace();
        if (position_ < text_.size() && text_[position_] == c) { ++position_; return true; }
        return false;
    }
    void expect(char c) { if (!consume(c)) fail(std::string("expected '") + c + "'"); }
    static void utf8(std::string& out, uint32_t c) {
        if (c <= 0x7f) out.push_back(static_cast<char>(c));
        else if (c <= 0x7ff) { out.push_back(static_cast<char>(0xc0 | (c >> 6))); out.push_back(static_cast<char>(0x80 | (c & 63))); }
        else if (c <= 0xffff) { out.push_back(static_cast<char>(0xe0 | (c >> 12))); out.push_back(static_cast<char>(0x80 | ((c >> 6) & 63))); out.push_back(static_cast<char>(0x80 | (c & 63))); }
        else { out.push_back(static_cast<char>(0xf0 | (c >> 18))); out.push_back(static_cast<char>(0x80 | ((c >> 12) & 63))); out.push_back(static_cast<char>(0x80 | ((c >> 6) & 63))); out.push_back(static_cast<char>(0x80 | (c & 63))); }
    }
    uint32_t hex4() {
        uint32_t result = 0;
        for (int i = 0; i < 4; ++i) {
            if (position_ == text_.size()) fail("incomplete Unicode escape");
            char c = text_[position_++];
            result <<= 4;
            if (c >= '0' && c <= '9') result |= c - '0';
            else if (c >= 'a' && c <= 'f') result |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') result |= c - 'A' + 10;
            else fail("invalid Unicode escape");
        }
        return result;
    }
    std::string string() {
        expect('"');
        std::string result;
        while (position_ < text_.size()) {
            unsigned char c = static_cast<unsigned char>(text_[position_++]);
            if (c == '"') return result;
            if (c < 32) fail("unescaped control character");
            if (c != '\\') { result.push_back(static_cast<char>(c)); continue; }
            if (position_ == text_.size()) fail("incomplete escape");
            char e = text_[position_++];
            switch (e) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                uint32_t code = hex4();
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (position_ + 2 > text_.size() || text_[position_] != '\\' || text_[position_ + 1] != 'u') fail("missing low surrogate");
                    position_ += 2;
                    uint32_t low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
                    code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                } else if (code >= 0xdc00 && code <= 0xdfff) fail("unpaired low surrogate");
                utf8(result, code);
                break;
            }
            default: fail("invalid escape");
            }
        }
        fail("unterminated string");
    }
    Json value(int depth) {
        if (depth > 128) fail("maximum nesting depth exceeded");
        whitespace();
        if (position_ == text_.size()) fail("expected value");
        char c = text_[position_];
        if (c == '"') return Json(string());
        if (consume('{')) {
            Json::Object object;
            if (consume('}')) return Json(std::move(object));
            do {
                std::string key = string();
                expect(':');
                if (!object.emplace(key, value(depth + 1)).second) fail("duplicate key: " + key);
            } while (consume(','));
            expect('}');
            return Json(std::move(object));
        }
        if (consume('[')) {
            Json::Array array;
            if (consume(']')) return Json(std::move(array));
            do { array.push_back(value(depth + 1)); } while (consume(','));
            expect(']');
            return Json(std::move(array));
        }
        for (const std::string& literal : {std::string("true"), std::string("false"), std::string("null")}) {
            if (text_.compare(position_, literal.size(), literal) == 0) {
                position_ += literal.size();
                if (literal == "null") return Json();
                return Json(literal == "true");
            }
        }
        size_t start = position_;
        if (text_[position_] == '-') ++position_;
        if (position_ == text_.size()) fail("incomplete number");
        if (text_[position_] == '0') ++position_;
        else {
            if (text_[position_] < '1' || text_[position_] > '9') fail("expected number");
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            size_t digits = position_;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
            if (position_ == digits) fail("missing fraction digits");
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            size_t digits = position_;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
            if (position_ == digits) fail("missing exponent digits");
        }
        try {
            double number = std::stod(text_.substr(start, position_ - start));
            if (!std::isfinite(number)) fail("number out of range");
            return Json(number);
        } catch (const std::invalid_argument&) { fail("invalid number"); }
          catch (const std::out_of_range&) { fail("number out of range"); }
    }
};

std::string quote(const std::string& text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
            else out << c;
        }
    }
    out << '"';
    return out.str();
}

}

Json Json::parse(const std::string& text) { return Parser(text).parse(); }
Json Json::read(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open JSON file: " + path.u8string());
    std::ostringstream text;
    text << in.rdbuf();
    return parse(text.str());
}
bool Json::is_null() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool Json::is_object() const { return std::holds_alternative<Object>(value_); }
bool Json::is_array() const { return std::holds_alternative<Array>(value_); }
bool Json::contains(const std::string& key) const { return is_object() && object().find(key) != object().end(); }
const Json& Json::at(const std::string& key) const {
    auto it = object().find(key);
    if (it == object().end()) throw std::runtime_error("Missing JSON field: " + key);
    return it->second;
}
const Json& Json::at(size_t index) const { return array().at(index); }
const Json::Array& Json::array() const {
    if (!is_array()) throw std::runtime_error("Expected JSON array");
    return std::get<Array>(value_);
}
const Json::Object& Json::object() const {
    if (!is_object()) throw std::runtime_error("Expected JSON object");
    return std::get<Object>(value_);
}
std::string Json::string() const {
    if (!std::holds_alternative<std::string>(value_)) throw std::runtime_error("Expected JSON string");
    return std::get<std::string>(value_);
}
double Json::number() const {
    if (!std::holds_alternative<double>(value_)) throw std::runtime_error("Expected JSON number");
    return std::get<double>(value_);
}
int Json::integer() const {
    double n = number();
    if (std::floor(n) != n || n < std::numeric_limits<int>::min() || n > std::numeric_limits<int>::max()) throw std::runtime_error("Expected 32-bit JSON integer");
    return static_cast<int>(n);
}
bool Json::boolean() const {
    if (!std::holds_alternative<bool>(value_)) throw std::runtime_error("Expected JSON boolean");
    return std::get<bool>(value_);
}
std::string Json::string_or(const std::string& key, const std::string& fallback) const { return contains(key) ? at(key).string() : fallback; }
double Json::number_or(const std::string& key, double fallback) const { return contains(key) ? at(key).number() : fallback; }
int Json::int_or(const std::string& key, int fallback) const { return contains(key) ? at(key).integer() : fallback; }
bool Json::bool_or(const std::string& key, bool fallback) const { return contains(key) ? at(key).boolean() : fallback; }
std::string Json::dump(int indent) const {
    if (indent < 0 || indent > 16) throw std::runtime_error("JSON indentation must be between 0 and 16");
    if (is_null()) return "null";
    if (std::holds_alternative<bool>(value_)) return boolean() ? "true" : "false";
    if (std::holds_alternative<double>(value_)) {
        if (!std::isfinite(number())) throw std::runtime_error("Cannot serialize non-finite JSON number");
        std::ostringstream out; out << std::setprecision(17) << number(); return out.str();
    }
    if (std::holds_alternative<std::string>(value_)) return quote(string());
    std::ostringstream out;
    bool obj = is_object();
    out << (obj ? '{' : '[');
    size_t count = obj ? object().size() : array().size();
    size_t index = 0;
    auto element = [&](const std::string& key, const Json& child) {
        if (index++) out << ',';
        if (indent) out << '\n' << std::string(indent, ' ');
        if (obj) out << quote(key) << (indent ? ": " : ":");
        std::string child_text = child.dump(indent);
        for (char c : child_text) { out << c; if (indent && c == '\n') out << std::string(indent, ' '); }
    };
    if (obj) for (const std::pair<const std::string, Json>& entry : object()) element(entry.first, entry.second);
    else for (const Json& child : array()) element("", child);
    if (indent && count) out << '\n';
    out << (obj ? '}' : ']');
    return out.str();
}
void Json::write(const std::string& path) const {
    const auto destination = std::filesystem::u8path(path);
    std::filesystem::path parent = destination.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out(destination, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write JSON file: " + path);
    out << dump() << '\n';
    if (!out) throw std::runtime_error("JSON write failed: " + path);
}

}
