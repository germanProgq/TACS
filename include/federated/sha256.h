/*
 * Simple SHA256 implementation for model hashing
 * Avoids external dependencies
 */

#ifndef SHA256_H
#define SHA256_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace TACS {

class SHA256 {
public:
    SHA256();
    void update(const uint8_t* data, size_t len);
    void update(const void* data, size_t len) {
        update(static_cast<const uint8_t*>(data), len);
    }
    std::string final();
    
    static std::string hash(const void* data, size_t len);
    
private:
    uint32_t h[8];
    uint8_t buffer[64];
    size_t buffer_size;
    uint64_t total_size;
    
    void process_block(const uint8_t* block);
    
    static const uint32_t k[64];
};

} // namespace TACS

#endif // SHA256_H