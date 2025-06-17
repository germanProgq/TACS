#include <iostream>
#include <fstream>
#include "utils/json_parser.h"

int main() {
    try {
        // Create test JSON
        std::ofstream out("test.json");
        out << "{\n";
        out << "  \"class\": \"test\",\n";
        out << "  \"bbox\": [1, 2, 3, 4],\n";
        out << "  \"timestamp\": 1000000\n";
        out << "}\n";
        out.close();
        
        // Parse it
        tacs::JsonValue root = tacs::JsonParser::parseFile("test.json");
        
        std::cout << "Class: " << root["class"].asString() << std::endl;
        std::cout << "Timestamp: " << root["timestamp"].asNumber() << std::endl;
        
        const tacs::JsonValue& bbox = root["bbox"];
        std::cout << "Bbox type: " << bbox.getType() << " (should be " << tacs::JsonValue::ARRAY << ")" << std::endl;
        
        for (int i = 0; i < 4; ++i) {
            std::cout << "bbox[" << i << "] = " << bbox[i].asFloat() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}