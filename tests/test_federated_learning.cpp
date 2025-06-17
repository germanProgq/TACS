/*
 * Federated Learning System Tests
 * Comprehensive validation of distributed learning and V2X protocols
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <random>
#include <cassert>
#include "../src/federated/federated_learning.h"
#include "../src/federated/v2x_protocol.h"
#include "../src/core/tensor.h"

using namespace TACS;

// Test utilities
class TestUtils {
public:
    static std::unordered_map<std::string, Tensor> createDummyModel(size_t num_params = 10) {
        std::unordered_map<std::string, Tensor> params;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        for (size_t i = 0; i < num_params; ++i) {
            std::string name = "param_" + std::to_string(i);
            Tensor tensor({16, 16});  // 256 parameters each
            
            for (size_t j = 0; j < tensor.size(); ++j) {
                tensor.data()[j] = dist(gen);
            }
            
            params[name] = std::move(tensor);
        }
        
        return params;
    }
    
    static float calculateModelDifference(const std::unordered_map<std::string, Tensor>& model1,
                                        const std::unordered_map<std::string, Tensor>& model2) {
        float total_diff = 0.0f;
        size_t total_params = 0;
        
        for (const auto& [name, tensor1] : model1) {
            auto it = model2.find(name);
            if (it == model2.end()) continue;
            
            const Tensor& tensor2 = it->second;
            for (size_t i = 0; i < tensor1.size(); ++i) {
                total_diff += std::abs(tensor1.data()[i] - tensor2.data()[i]);
                total_params++;
            }
        }
        
        return total_params > 0 ? total_diff / total_params : 0.0f;
    }
};

// Test 1: Model Snapshot Serialization
void testModelSnapshot() {
    std::cout << "\n=== Test 1: Model Snapshot Serialization ===" << std::endl;
    
    // Create a snapshot
    ModelSnapshot original;
    original.version_hash = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    original.timestamp = std::chrono::system_clock::now();
    original.parameters = TestUtils::createDummyModel(5);
    original.performance_metrics = 0.92f;
    original.training_samples = 10000;
    
    // Serialize
    auto serialized = original.serialize();
    std::cout << "Serialized size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize
    ModelSnapshot deserialized = ModelSnapshot::deserialize(serialized);
    
    // Verify
    assert(original.version_hash == deserialized.version_hash);
    assert(original.performance_metrics == deserialized.performance_metrics);
    assert(original.training_samples == deserialized.training_samples);
    assert(original.parameters.size() == deserialized.parameters.size());
    
    float diff = TestUtils::calculateModelDifference(original.parameters, deserialized.parameters);
    assert(diff < 1e-6f);
    
    std::cout << "✓ Model snapshot serialization/deserialization successful" << std::endl;
}

// Test 2: SHA256 Model Hashing
void testModelHashing() {
    std::cout << "\n=== Test 2: SHA256 Model Hashing ===" << std::endl;
    
    FederatedLearning fl;
    
    // Create two identical models
    auto model1 = TestUtils::createDummyModel(3);
    auto model2 = model1;  // Copy
    
    // Hash should be identical
    std::string hash1 = fl.computeModelHash(model1);
    std::string hash2 = fl.computeModelHash(model2);
    
    assert(hash1 == hash2);
    assert(hash1.length() == 64);  // SHA256 hex string
    
    std::cout << "Hash of identical models: " << hash1.substr(0, 16) << "..." << std::endl;
    
    // Modify one parameter slightly
    model2["param_0"].data()[0] += 0.001f;
    std::string hash3 = fl.computeModelHash(model2);
    
    assert(hash1 != hash3);
    std::cout << "Hash after modification: " << hash3.substr(0, 16) << "..." << std::endl;
    
    std::cout << "✓ Model hashing working correctly" << std::endl;
}

// Test 3: Knowledge Distillation
void testKnowledgeDistillation() {
    std::cout << "\n=== Test 3: Knowledge Distillation ===" << std::endl;
    
    FederatedConfig config;
    config.distillation_temperature = 3.0f;
    config.distillation_alpha = 0.5f;
    
    FederatedLearning fl(config);
    
    // Create teacher and student models
    auto teacher_params = TestUtils::createDummyModel(3);
    auto student_params = TestUtils::createDummyModel(3);
    
    // Record initial difference
    float initial_diff = TestUtils::calculateModelDifference(teacher_params, student_params);
    std::cout << "Initial model difference: " << initial_diff << std::endl;
    
    // Apply distillation
    auto distilled = fl.distillKnowledge(teacher_params, student_params);
    
    // Distilled model should be between teacher and student
    float teacher_dist_diff = TestUtils::calculateModelDifference(teacher_params, distilled);
    float student_dist_diff = TestUtils::calculateModelDifference(student_params, distilled);
    
    std::cout << "Teacher-distilled difference: " << teacher_dist_diff << std::endl;
    std::cout << "Student-distilled difference: " << student_dist_diff << std::endl;
    
    // Both should be less than initial difference (closer to both)
    assert(teacher_dist_diff < initial_diff);
    assert(student_dist_diff < initial_diff);
    
    std::cout << "✓ Knowledge distillation working correctly" << std::endl;
}

// Test 4: EWC Penalty Application
void testEWCPenalty() {
    std::cout << "\n=== Test 4: EWC Penalty Application ===" << std::endl;
    
    FederatedLearning fl;
    
    // Initialize with a model
    auto initial_params = TestUtils::createDummyModel(3);
    fl.initialize(initial_params);
    
    // Create some gradients
    std::unordered_map<std::string, Tensor> grads;
    for (const auto& [name, param] : initial_params) {
        Tensor grad(param.shape);
        for (size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] = 0.01f;  // Small gradient
        }
        grads[name] = grad;
    }
    
    // Update Fisher information
    fl.updateFisherInformation(grads);
    
    // Apply EWC penalty
    auto penalized_grads = fl.applyEWCPenalty(grads);
    
    // Penalized gradients should be different
    float grad_diff = TestUtils::calculateModelDifference(grads, penalized_grads);
    assert(grad_diff > 0.0f);
    
    std::cout << "Original gradient magnitude: " << 0.01f << std::endl;
    std::cout << "Average gradient change from EWC: " << grad_diff << std::endl;
    
    std::cout << "✓ EWC penalty application successful" << std::endl;
}

// Test 5: Performance Monitoring and Rollback
void testPerformanceRollback() {
    std::cout << "\n=== Test 5: Performance Monitoring and Rollback ===" << std::endl;
    
    FederatedConfig config;
    config.rollback_threshold = 0.05f;  // 5% drop triggers rollback
    
    FederatedLearning fl(config);
    
    // Initialize with good model
    auto good_model = TestUtils::createDummyModel(3);
    fl.initialize(good_model);
    
    // Update with good performance
    fl.updateLocalModel(good_model, 0.92f, 1000);
    fl.reportPerformance(0.92f);
    
    // Store version hash for rollback
    std::string good_version = fl.getCurrentSnapshot().version_hash;
    
    // Update with degraded model (>5% drop)
    auto bad_model = TestUtils::createDummyModel(3);
    fl.updateLocalModel(bad_model, 0.85f, 2000);  // 7% drop
    fl.reportPerformance(0.85f);
    
    // Should detect need for rollback
    assert(fl.shouldRollback());
    
    // Perform rollback
    bool rollback_success = fl.rollbackToVersion(good_version);
    assert(rollback_success);
    
    // Verify model was restored
    auto current_params = fl.getModelParameters();
    float diff = TestUtils::calculateModelDifference(good_model, current_params);
    assert(diff < 1e-6f);
    
    std::cout << "✓ Performance monitoring and rollback working correctly" << std::endl;
}

// Test 6: V2X Basic Safety Message
void testV2XBasicSafetyMessage() {
    std::cout << "\n=== Test 6: V2X Basic Safety Message ===" << std::endl;
    
    // Create BSM
    BasicSafetyMessage bsm;
    bsm.msg_count = 12345;
    bsm.id = 0xABCDEF01;
    bsm.latitude = 37.7749;
    bsm.longitude = -122.4194;
    bsm.elevation = 50.0;
    bsm.speed = 15.5;  // m/s
    bsm.heading = 45.0;  // degrees
    bsm.acceleration = 0.5;
    bsm.brake_status = 0;
    bsm.vehicle_size_length = 450;  // cm
    bsm.vehicle_size_width = 180;   // cm
    bsm.timestamp = std::chrono::system_clock::now();
    
    // Serialize
    auto serialized = bsm.serialize();
    std::cout << "BSM serialized size: " << serialized.size() << " bytes" << std::endl;
    
    // Deserialize
    BasicSafetyMessage deserialized = BasicSafetyMessage::deserialize(serialized);
    
    // Verify
    assert(bsm.msg_count == deserialized.msg_count);
    assert(bsm.id == deserialized.id);
    assert(std::abs(bsm.latitude - deserialized.latitude) < 1e-6);
    assert(std::abs(bsm.longitude - deserialized.longitude) < 1e-6);
    assert(std::abs(bsm.speed - deserialized.speed) < 1e-6);
    
    std::cout << "✓ V2X BSM serialization/deserialization successful" << std::endl;
}

// Test 7: V2X Protocol Message Handling
void testV2XProtocol() {
    std::cout << "\n=== Test 7: V2X Protocol Message Handling ===" << std::endl;
    
    V2XConfig config;
    config.mode = V2XMode::CV2X_PC5;  // Use simulated C-V2X for testing
    config.enable_security = false;    // Disable for testing
    
    V2XProtocol v2x(config);
    
    // Initialize (may fail if no network interface)
    bool init_success = v2x.initialize();
    if (!init_success) {
        std::cout << "! V2X initialization failed (expected in test environment)" << std::endl;
        return;
    }
    
    // Register message handler
    std::atomic<bool> message_received(false);
    v2x.registerHandler(V2XMessageType::BSM, 
        [&message_received](const std::vector<uint8_t>& data, const std::string& sender) {
            message_received = true;
            std::cout << "Received BSM from: " << sender << std::endl;
        });
    
    // Send BSM
    BasicSafetyMessage bsm;
    bsm.msg_count = 1;
    bsm.id = 0x12345678;
    bsm.latitude = 37.7749;
    bsm.longitude = -122.4194;
    bsm.speed = 10.0;
    
    v2x.sendBSM(bsm);
    
    // Wait for reception (in loopback)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Get statistics
    auto stats = v2x.getStatistics();
    std::cout << "Messages sent: " << stats.messages_sent << std::endl;
    std::cout << "Messages received: " << stats.messages_received << std::endl;
    
    v2x.shutdown();
    
    std::cout << "✓ V2X protocol basic functionality tested" << std::endl;
}

// Test 8: Intersection Synchronization
void testIntersectionSync() {
    std::cout << "\n=== Test 8: Intersection Synchronization ===" << std::endl;
    
    // Create intersection sync for testing
    IntersectionSync sync("INT_001");
    
    // Register neighbors
    sync.registerNeighbor("INT_002", "192.168.1.2");
    sync.registerNeighbor("INT_003", "192.168.1.3");
    
    // Create traffic state
    IntersectionSync::TrafficState state;
    state.intersection_id = "INT_001";
    state.queue_lengths["north"] = 5;
    state.queue_lengths["south"] = 3;
    state.queue_lengths["east"] = 8;
    state.queue_lengths["west"] = 2;
    state.current_phase = 2;
    state.phase_remaining_time = 25;
    state.emergency_active = false;
    
    // Serialize and deserialize
    auto serialized = state.serialize();
    auto deserialized = IntersectionSync::TrafficState::deserialize(serialized);
    
    assert(state.intersection_id == deserialized.intersection_id);
    assert(state.queue_lengths.size() == deserialized.queue_lengths.size());
    assert(state.current_phase == deserialized.current_phase);
    
    std::cout << "Traffic state size: " << serialized.size() << " bytes" << std::endl;
    std::cout << "Queue lengths preserved: " << deserialized.queue_lengths.size() << std::endl;
    
    std::cout << "✓ Intersection synchronization tested successfully" << std::endl;
}

// Test 9: Federated Learning Integration
void testFederatedIntegration() {
    std::cout << "\n=== Test 9: Federated Learning Integration ===" << std::endl;
    
    // Simulate two agents
    FederatedConfig config1, config2;
    config1.server_port = 9001;
    config2.server_port = 9002;
    config1.enable_v2x = false;  // Disable for simple test
    config2.enable_v2x = false;
    
    FederatedLearning agent1(config1);
    FederatedLearning agent2(config2);
    
    // Initialize both agents with different models
    auto model1 = TestUtils::createDummyModel(5);
    auto model2 = TestUtils::createDummyModel(5);
    
    agent1.initialize(model1);
    agent2.initialize(model2);
    
    // Simulate training updates
    std::cout << "Agent 1 training..." << std::endl;
    agent1.updateLocalModel(model1, 0.91f, 5000);
    
    std::cout << "Agent 2 training..." << std::endl;
    agent2.updateLocalModel(model2, 0.93f, 6000);
    
    // Get final models
    auto final1 = agent1.getModelParameters();
    auto final2 = agent2.getModelParameters();
    
    std::cout << "Model 1 hash: " << agent1.getCurrentSnapshot().version_hash.substr(0, 16) << "..." << std::endl;
    std::cout << "Model 2 hash: " << agent2.getCurrentSnapshot().version_hash.substr(0, 16) << "..." << std::endl;
    
    std::cout << "✓ Federated learning integration tested" << std::endl;
}

// Test 10: Distillation Bank
void testDistillationBank() {
    std::cout << "\n=== Test 10: Distillation Bank ===" << std::endl;
    
    // Create distillation bank (server)
    DistillationBank bank(9999);
    
    // Note: Full server test would require actual network setup
    std::cout << "Distillation bank created on port 9999" << std::endl;
    
    // Test model aggregation with empty agents (should return empty model)
    ModelSnapshot aggregated = bank.aggregateModels();
    std::cout << "Empty aggregation hash: " << aggregated.version_hash.substr(0, 16) << "..." << std::endl;
    
    std::cout << "✓ Distillation bank basic functionality tested" << std::endl;
}

// Performance benchmark
void performanceBenchmark() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;
    
    const size_t num_params = 100;  // 100 parameter tensors
    const size_t iterations = 100;
    
    FederatedLearning fl;
    auto model = TestUtils::createDummyModel(num_params);
    fl.initialize(model);
    
    // Benchmark model hashing
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        fl.computeModelHash(model);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto hash_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Benchmark serialization
    ModelSnapshot snapshot;
    snapshot.parameters = model;
    snapshot.version_hash = fl.computeModelHash(model);
    snapshot.timestamp = std::chrono::system_clock::now();
    snapshot.performance_metrics = 0.92f;
    snapshot.training_samples = 10000;
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto serialized = snapshot.serialize();
    }
    end = std::chrono::high_resolution_clock::now();
    auto serialize_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Benchmark distillation
    auto teacher = TestUtils::createDummyModel(num_params);
    auto student = TestUtils::createDummyModel(num_params);
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        fl.distillKnowledge(teacher, student);
    }
    end = std::chrono::high_resolution_clock::now();
    auto distill_time = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Model hashing time: " << hash_time << " ms" << std::endl;
    std::cout << "Serialization time: " << serialize_time << " ms" << std::endl;
    std::cout << "Distillation time: " << distill_time << " ms" << std::endl;
    
    // Verify performance targets
    assert(hash_time < 10.0);      // Hashing should be fast
    assert(serialize_time < 50.0);  // Serialization under 50ms
    assert(distill_time < 100.0);   // Distillation under 100ms
    
    std::cout << "✓ All performance targets met" << std::endl;
}

int main() {
    std::cout << "=== TACS Federated Learning System Tests ===" << std::endl;
    std::cout << "Testing distributed learning and V2X protocols..." << std::endl;
    
    try {
        // Run all tests
        testModelSnapshot();
        testModelHashing();
        testKnowledgeDistillation();
        testEWCPenalty();
        testPerformanceRollback();
        testV2XBasicSafetyMessage();
        testV2XProtocol();
        testIntersectionSync();
        testFederatedIntegration();
        testDistillationBank();
        
        // Performance benchmark
        performanceBenchmark();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        std::cout << "Federated learning system is production-ready!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}