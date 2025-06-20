/*
 * Configuration Manager
 * Production-ready configuration management system
 */

#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mutex>

namespace tacs {
namespace utils {

class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }
    
    // Load configuration from file
    bool loadFromFile(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open config file: " << filepath << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Remove comments
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);
            
            if (line.empty()) continue;
            
            // Parse key=value
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = line.substr(0, equals_pos);
                std::string value = line.substr(equals_pos + 1);
                
                // Trim key and value
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                config_[key] = value;
            }
        }
        
        return true;
    }
    
    // Get string value
    std::string getString(const std::string& key, const std::string& default_value = "") const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = config_.find(key);
        return (it != config_.end()) ? it->second : default_value;
    }
    
    // Get integer value
    int getInt(const std::string& key, int default_value = 0) const {
        std::string str_value = getString(key);
        if (str_value.empty()) return default_value;
        
        try {
            return std::stoi(str_value);
        } catch (...) {
            return default_value;
        }
    }
    
    // Get float value
    float getFloat(const std::string& key, float default_value = 0.0f) const {
        std::string str_value = getString(key);
        if (str_value.empty()) return default_value;
        
        try {
            return std::stof(str_value);
        } catch (...) {
            return default_value;
        }
    }
    
    // Get boolean value
    bool getBool(const std::string& key, bool default_value = false) const {
        std::string str_value = getString(key);
        if (str_value.empty()) return default_value;
        
        // Convert to lowercase
        std::transform(str_value.begin(), str_value.end(), str_value.begin(), ::tolower);
        
        return (str_value == "true" || str_value == "1" || str_value == "yes" || str_value == "on");
    }
    
    // Set value
    void setValue(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_[key] = value;
    }
    
    // Save configuration to file
    bool saveToFile(const std::string& filepath) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::ofstream file(filepath);
        if (!file.is_open()) {
            return false;
        }
        
        file << "# TACS Configuration File\n";
        file << "# Generated automatically\n\n";
        
        for (const auto& [key, value] : config_) {
            file << key << " = " << value << "\n";
        }
        
        return true;
    }
    
private:
    ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> config_;
};

} // namespace utils
} // namespace tacs