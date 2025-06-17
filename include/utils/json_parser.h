// Simple JSON parser for production use without external dependencies
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace tacs {

class JsonValue {
public:
    enum Type {
        NULL_TYPE,
        BOOL,
        NUMBER,
        STRING,
        ARRAY,
        OBJECT
    };

    JsonValue() : type_(NULL_TYPE) {}
    JsonValue(bool b) : type_(BOOL), bool_value_(b) {}
    JsonValue(double n) : type_(NUMBER), number_value_(n) {}
    JsonValue(const std::string& s) : type_(STRING), string_value_(s) {}
    
    Type getType() const { return type_; }
    
    bool asBool() const {
        if (type_ != BOOL) throw std::runtime_error("Not a boolean");
        return bool_value_;
    }
    
    double asNumber() const {
        if (type_ != NUMBER) throw std::runtime_error("Not a number");
        return number_value_;
    }
    
    int asInt() const {
        return static_cast<int>(asNumber());
    }
    
    float asFloat() const {
        return static_cast<float>(asNumber());
    }
    
    const std::string& asString() const {
        if (type_ != STRING) throw std::runtime_error("Not a string");
        return string_value_;
    }
    
    const std::vector<JsonValue>& asArray() const {
        if (type_ != ARRAY) throw std::runtime_error("Not an array");
        return array_value_;
    }
    
    const std::unordered_map<std::string, JsonValue>& asObject() const {
        if (type_ != OBJECT) throw std::runtime_error("Not an object");
        return object_value_;
    }
    
    // Array operations
    void push_back(const JsonValue& value) {
        if (type_ != ARRAY) {
            type_ = ARRAY;
            array_value_.clear();
        }
        array_value_.push_back(value);
    }
    
    void clearArray() {
        if (type_ == ARRAY) {
            array_value_.clear();
        }
    }
    
    const JsonValue& operator[](size_t index) const {
        return asArray()[index];
    }
    
    // Object operations
    void set(const std::string& key, const JsonValue& value) {
        if (type_ != OBJECT) {
            type_ = OBJECT;
            object_value_.clear();
        }
        object_value_[key] = value;
    }
    
    const JsonValue& operator[](const std::string& key) const {
        const auto& obj = asObject();
        auto it = obj.find(key);
        if (it == obj.end()) {
            static JsonValue null_value;
            return null_value;
        }
        return it->second;
    }
    
    bool has(const std::string& key) const {
        if (type_ != OBJECT) return false;
        return object_value_.find(key) != object_value_.end();
    }

private:
    Type type_;
    bool bool_value_;
    double number_value_;
    std::string string_value_;
    std::vector<JsonValue> array_value_;
    std::unordered_map<std::string, JsonValue> object_value_;
};

// Simple JSON parser
class JsonParser {
public:
    static JsonValue parse(const std::string& json);
    static JsonValue parseFile(const std::string& filename);
    
private:
    static JsonValue parseValue(const std::string& json, size_t& pos);
    static JsonValue parseObject(const std::string& json, size_t& pos);
    static JsonValue parseArray(const std::string& json, size_t& pos);
    static JsonValue parseString(const std::string& json, size_t& pos);
    static JsonValue parseNumber(const std::string& json, size_t& pos);
    static JsonValue parseBool(const std::string& json, size_t& pos);
    static JsonValue parseNull(const std::string& json, size_t& pos);
    static void skipWhitespace(const std::string& json, size_t& pos);
};

} // namespace tacs