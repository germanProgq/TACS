/*
 * Simple SHA256 implementation
 */

#include "../../include/federated/sha256.h"
#include <iomanip>
#include <sstream>

namespace TACS {

const uint32_t SHA256::k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static inline uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static inline uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

SHA256::SHA256() {
    h[0] = 0x6a09e667;
    h[1] = 0xbb67ae85;
    h[2] = 0x3c6ef372;
    h[3] = 0xa54ff53a;
    h[4] = 0x510e527f;
    h[5] = 0x9b05688c;
    h[6] = 0x1f83d9ab;
    h[7] = 0x5be0cd19;
    
    buffer_size = 0;
    total_size = 0;
}

void SHA256::process_block(const uint8_t* block) {
    uint32_t w[64];
    
    // Copy block into first 16 words
    for (int i = 0; i < 16; i++) {
        w[i] = (block[i*4] << 24) | (block[i*4+1] << 16) | 
               (block[i*4+2] << 8) | block[i*4+3];
    }
    
    // Extend the first 16 words into the remaining 48 words
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }
    
    // Initialize working variables
    uint32_t a = h[0];
    uint32_t b = h[1];
    uint32_t c = h[2];
    uint32_t d = h[3];
    uint32_t e = h[4];
    uint32_t f = h[5];
    uint32_t g = h[6];
    uint32_t h_val = h[7];
    
    // Compression function main loop
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h_val + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h_val = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add the compressed chunk to the current hash value
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += h_val;
}

void SHA256::update(const uint8_t* data, size_t len) {
    total_size += len;
    
    // Process any buffered data
    if (buffer_size > 0) {
        size_t to_copy = std::min(len, 64 - buffer_size);
        memcpy(buffer + buffer_size, data, to_copy);
        buffer_size += to_copy;
        data += to_copy;
        len -= to_copy;
        
        if (buffer_size == 64) {
            process_block(buffer);
            buffer_size = 0;
        }
    }
    
    // Process complete blocks
    while (len >= 64) {
        process_block(data);
        data += 64;
        len -= 64;
    }
    
    // Buffer remaining data
    if (len > 0) {
        memcpy(buffer, data, len);
        buffer_size = len;
    }
}

std::string SHA256::final() {
    // Pad the message
    uint64_t bit_length = total_size * 8;
    
    // Add padding bit
    uint8_t padding[64] = {0};
    padding[0] = 0x80;
    
    size_t padding_len = (buffer_size < 56) ? (56 - buffer_size) : (120 - buffer_size);
    update(padding, padding_len);
    
    // Add length
    uint8_t length_bytes[8];
    for (int i = 0; i < 8; i++) {
        length_bytes[7-i] = (bit_length >> (i * 8)) & 0xff;
    }
    update(length_bytes, 8);
    
    // Convert hash to hex string
    std::stringstream ss;
    for (int i = 0; i < 8; i++) {
        ss << std::hex << std::setw(8) << std::setfill('0') << h[i];
    }
    
    return ss.str();
}

std::string SHA256::hash(const void* data, size_t len) {
    SHA256 sha;
    sha.update(static_cast<const uint8_t*>(data), len);
    return sha.final();
}

} // namespace TACS