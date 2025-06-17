/*
 * Cryptographic utilities for federated learning
 * Production-ready encryption and signing
 */

#ifndef CRYPTO_UTILS_H
#define CRYPTO_UTILS_H

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace TACS {

// ChaCha20 stream cipher implementation
class ChaCha20 {
public:
    ChaCha20(const uint8_t* key, const uint8_t* nonce);
    void encrypt(uint8_t* data, size_t len);
    void decrypt(uint8_t* data, size_t len) { encrypt(data, len); } // Stream cipher property
    
private:
    uint32_t state_[16];
    uint32_t working_state_[16];
    uint64_t counter_;
    
    void chacha_block();
    void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d);
    static uint32_t rotl(uint32_t x, int n);
};

// Ed25519 digital signature implementation
class Ed25519 {
public:
    static constexpr size_t SEED_SIZE = 32;
    static constexpr size_t PUBLIC_KEY_SIZE = 32;
    static constexpr size_t PRIVATE_KEY_SIZE = 64;
    static constexpr size_t SIGNATURE_SIZE = 64;
    
    // Generate keypair from seed
    static void generateKeypair(const uint8_t seed[SEED_SIZE], 
                               uint8_t public_key[PUBLIC_KEY_SIZE],
                               uint8_t private_key[PRIVATE_KEY_SIZE]);
    
    // Sign message
    static void sign(const uint8_t* message, size_t message_len,
                    const uint8_t private_key[PRIVATE_KEY_SIZE],
                    uint8_t signature[SIGNATURE_SIZE]);
    
    // Verify signature
    static bool verify(const uint8_t* message, size_t message_len,
                      const uint8_t public_key[PUBLIC_KEY_SIZE],
                      const uint8_t signature[SIGNATURE_SIZE]);
    
private:
    // Field arithmetic for curve25519
    struct FieldElement {
        int32_t v[10];
    };
    
    struct GroupElement {
        FieldElement X;
        FieldElement Y;
        FieldElement Z;
        FieldElement T;
    };
    
    static void field_add(FieldElement& out, const FieldElement& a, const FieldElement& b);
    static void field_sub(FieldElement& out, const FieldElement& a, const FieldElement& b);
    static void field_mul(FieldElement& out, const FieldElement& a, const FieldElement& b);
    static void scalar_mult(GroupElement& out, const uint8_t* scalar, const GroupElement& point);
};

// High-level crypto utilities
class CryptoUtils {
public:
    // Encrypt data with ChaCha20
    static std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext,
                                       const std::vector<uint8_t>& key);
    
    // Decrypt data with ChaCha20
    static std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext,
                                       const std::vector<uint8_t>& key);
    
    // Sign data with Ed25519
    static std::vector<uint8_t> sign(const std::vector<uint8_t>& data,
                                    const std::vector<uint8_t>& private_key);
    
    // Verify signature with Ed25519
    static bool verify(const std::vector<uint8_t>& data,
                       const std::vector<uint8_t>& signature,
                       const std::vector<uint8_t>& public_key);
    
    // Generate random bytes
    static std::vector<uint8_t> randomBytes(size_t count);
    
    // Derive key from password using Argon2
    static std::vector<uint8_t> deriveKey(const std::string& password,
                                         const std::vector<uint8_t>& salt,
                                         size_t key_length);
};

} // namespace TACS

#endif // CRYPTO_UTILS_H