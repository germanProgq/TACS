// Simple JSON parser implementation
#include "utils/json_parser.h"
#include <fstream>
#include <cctype>

namespace tacs {

JsonValue JsonParser::parse(const std::string& json) {
    size_t pos = 0;
    return parseValue(json, pos);
}

JsonValue JsonParser::parseFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse(buffer.str());
}

void JsonParser::skipWhitespace(const std::string& json, size_t& pos) {
    while (pos < json.length() && std::isspace(json[pos])) {
        ++pos;
    }
}

JsonValue JsonParser::parseValue(const std::string& json, size_t& pos) {
    skipWhitespace(json, pos);
    
    if (pos >= json.length()) {
        throw std::runtime_error("Unexpected end of JSON");
    }
    
    char ch = json[pos];
    
    if (ch == '{') {
        return parseObject(json, pos);
    } else if (ch == '[') {
        return parseArray(json, pos);
    } else if (ch == '"') {
        return parseString(json, pos);
    } else if (ch == 't' || ch == 'f') {
        return parseBool(json, pos);
    } else if (ch == 'n') {
        return parseNull(json, pos);
    } else if (ch == '-' || std::isdigit(ch)) {
        return parseNumber(json, pos);
    } else {
        throw std::runtime_error("Unexpected character in JSON");
    }
}

JsonValue JsonParser::parseObject(const std::string& json, size_t& pos) {
    JsonValue obj;
    obj.set("", JsonValue()); // Initialize as object
    obj.asObject(); // Force type
    
    ++pos; // Skip '{'
    skipWhitespace(json, pos);
    
    if (pos < json.length() && json[pos] == '}') {
        ++pos;
        return obj;
    }
    
    while (pos < json.length()) {
        skipWhitespace(json, pos);
        
        // Parse key
        if (json[pos] != '"') {
            throw std::runtime_error("Expected string key in object");
        }
        
        JsonValue keyValue = parseString(json, pos);
        std::string key = keyValue.asString();
        
        skipWhitespace(json, pos);
        
        if (pos >= json.length() || json[pos] != ':') {
            throw std::runtime_error("Expected ':' after object key");
        }
        ++pos; // Skip ':'
        
        // Parse value
        JsonValue value = parseValue(json, pos);
        obj.set(key, value);
        
        skipWhitespace(json, pos);
        
        if (pos < json.length() && json[pos] == ',') {
            ++pos; // Skip ','
            continue;
        } else if (pos < json.length() && json[pos] == '}') {
            ++pos; // Skip '}'
            break;
        } else {
            throw std::runtime_error("Expected ',' or '}' in object");
        }
    }
    
    return obj;
}

JsonValue JsonParser::parseArray(const std::string& json, size_t& pos) {
    JsonValue arr;
    // Initialize as array type
    arr.push_back(JsonValue());  // This sets type to ARRAY
    arr.clearArray();  // Clear the dummy element
    
    ++pos; // Skip '['
    skipWhitespace(json, pos);
    
    if (pos < json.length() && json[pos] == ']') {
        ++pos;
        return arr;
    }
    
    while (pos < json.length()) {
        JsonValue value = parseValue(json, pos);
        arr.push_back(value);
        
        skipWhitespace(json, pos);
        
        if (pos < json.length() && json[pos] == ',') {
            ++pos; // Skip ','
            continue;
        } else if (pos < json.length() && json[pos] == ']') {
            ++pos; // Skip ']'
            break;
        } else {
            throw std::runtime_error("Expected ',' or ']' in array");
        }
    }
    
    return arr;
}

JsonValue JsonParser::parseString(const std::string& json, size_t& pos) {
    ++pos; // Skip opening '"'
    
    std::string result;
    while (pos < json.length() && json[pos] != '"') {
        if (json[pos] == '\\') {
            ++pos;
            if (pos >= json.length()) {
                throw std::runtime_error("Unexpected end of string");
            }
            
            switch (json[pos]) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case '/': result += '/'; break;
                case 'b': result += '\b'; break;
                case 'f': result += '\f'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                default:
                    throw std::runtime_error("Invalid escape sequence");
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    
    if (pos >= json.length()) {
        throw std::runtime_error("Unterminated string");
    }
    
    ++pos; // Skip closing '"'
    return JsonValue(result);
}

JsonValue JsonParser::parseNumber(const std::string& json, size_t& pos) {
    size_t start = pos;
    
    if (json[pos] == '-') {
        ++pos;
    }
    
    while (pos < json.length() && std::isdigit(json[pos])) {
        ++pos;
    }
    
    if (pos < json.length() && json[pos] == '.') {
        ++pos;
        while (pos < json.length() && std::isdigit(json[pos])) {
            ++pos;
        }
    }
    
    if (pos < json.length() && (json[pos] == 'e' || json[pos] == 'E')) {
        ++pos;
        if (pos < json.length() && (json[pos] == '+' || json[pos] == '-')) {
            ++pos;
        }
        while (pos < json.length() && std::isdigit(json[pos])) {
            ++pos;
        }
    }
    
    std::string numStr = json.substr(start, pos - start);
    double value = std::stod(numStr);
    return JsonValue(value);
}

JsonValue JsonParser::parseBool(const std::string& json, size_t& pos) {
    if (json.substr(pos, 4) == "true") {
        pos += 4;
        return JsonValue(true);
    } else if (json.substr(pos, 5) == "false") {
        pos += 5;
        return JsonValue(false);
    } else {
        throw std::runtime_error("Invalid boolean value");
    }
}

JsonValue JsonParser::parseNull(const std::string& json, size_t& pos) {
    if (json.substr(pos, 4) == "null") {
        pos += 4;
        return JsonValue();
    } else {
        throw std::runtime_error("Invalid null value");
    }
}

} // namespace tacs