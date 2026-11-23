#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace Evans::Neural {

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    Json() = default;
    Json(bool value) : value_(value) {}
    Json(double value) : value_(value) {}
    Json(int value) : value_(static_cast<double>(value)) {}
    Json(const char* value) : value_(std::string(value)) {}
    Json(std::string value) : value_(std::move(value)) {}
    Json(Array value) : value_(std::move(value)) {}
    Json(Object value) : value_(std::move(value)) {}

    static Json parse(const std::string& text);
    static Json read(const std::filesystem::path& path);
    std::string dump(int indent = 2) const;
    void write(const std::string& path) const;
    bool is_null() const;
    bool is_object() const;
    bool is_array() const;
    bool contains(const std::string& key) const;
    const Json& at(const std::string& key) const;
    const Json& at(size_t index) const;
    const Array& array() const;
    const Object& object() const;
    std::string string() const;
    double number() const;
    int integer() const;
    bool boolean() const;
    std::string string_or(const std::string& key, const std::string& fallback) const;
    double number_or(const std::string& key, double fallback) const;
    int int_or(const std::string& key, int fallback) const;
    bool bool_or(const std::string& key, bool fallback) const;

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value_{nullptr};
};

}
