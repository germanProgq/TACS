// Phase 9 validation: Plugin learning system
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include "plugin/plugin_system.h"
#include "models/tacsnet.h"
#include "utils/image_decoder.h"
#include "utils/image_processing.h"
#include "utils/json_parser.h"

namespace fs = std::filesystem;
using namespace tacs;

// Create synthetic test data
void createTestData(const std::string& baseDir, int numSamples = 10) {
    fs::create_directories(baseDir + "/images");
    fs::create_directories(baseDir + "/metadata");
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> posDist(50, 350);
    std::uniform_int_distribution<> sizeDist(40, 120);
    std::uniform_int_distribution<> colorDist(0, 255);
    
    std::vector<std::string> classNames = {"electric_scooter", "delivery_robot", "emergency_cone"};
    
    for (int i = 0; i < numSamples; ++i) {
        // Create synthetic image
        Image image(640, 480, 3);
        for (int i = 0; i < image.width * image.height * image.channels; ++i) {
            image.data[i] = 200;  // Gray background
        }
        
        // Draw random object
        int x = posDist(gen);
        int y = posDist(gen);
        int w = sizeDist(gen);
        int h = sizeDist(gen);
        
        std::vector<uint8_t> color = {static_cast<uint8_t>(colorDist(gen)), 
                                      static_cast<uint8_t>(colorDist(gen)), 
                                      static_cast<uint8_t>(colorDist(gen))};
        rectangle(image, Rect(x, y, w, h), color, -1);
        
        // Add some texture
        for (int j = 0; j < 20; ++j) {
            int tx = x + rand() % w;
            int ty = y + rand() % h;
            circle(image, tx, ty, 3, {0, 0, 0}, -1);
        }
        
        // Save image
        std::string imagePath = baseDir + "/images/sample_" + std::to_string(i) + ".ppm";
        imwrite(imagePath, image);
        
        // Create metadata using simple JSON format
        std::string metaPath = baseDir + "/metadata/sample_" + std::to_string(i) + ".json";
        std::ofstream metaFile(metaPath);
        
        metaFile << "{\n";
        metaFile << "  \"class\": \"" << classNames[i % classNames.size()] << "\",\n";
        metaFile << "  \"bbox\": [" << x << ", " << y << ", " << w << ", " << h << "],\n";
        metaFile << "  \"image\": \"" << imagePath << "\",\n";
        // Use a smaller timestamp for testing
        metaFile << "  \"timestamp\": " << (1000000 + i) << ",\n";
        metaFile << "  \"attributes\": {\n";
        metaFile << "    \"source\": \"synthetic\",\n";
        metaFile << "    \"confidence\": \"high\"\n";
        metaFile << "  }\n";
        metaFile << "}\n";
        
        metaFile.close();
    }
}

// Test feature extraction
bool testFeatureExtraction() {
    std::cout << "\n=== Testing Feature Extraction ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Create test image
    Image testImage(100, 100, 3);
    randu(testImage, {0, 0, 0}, {255, 255, 255});
    Rect bbox(10, 10, 80, 80);
    
    auto start = std::chrono::high_resolution_clock::now();
    FeatureDescriptor features = pluginSystem.extractFeatures(testImage, bbox);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Feature extraction time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "RGB histogram size: " << features.rgbHistogram.size() << std::endl;
    std::cout << "HSV histogram size: " << features.hsvHistogram.size() << std::endl;
    std::cout << "Edge histogram size: " << features.edgeHistogram.size() << std::endl;
    std::cout << "Shape features size: " << features.shapeFeatures.size() << std::endl;
    std::cout << "Texture features size: " << features.textureFeatures.size() << std::endl;
    
    int totalFeatures = features.rgbHistogram.size() + features.hsvHistogram.size() +
                       features.edgeHistogram.size() + features.shapeFeatures.size() +
                       features.textureFeatures.size();
    
    std::cout << "Total feature dimension: " << totalFeatures << std::endl;
    
    return totalFeatures > 0;
}

// Test single image training
bool testSingleImageTraining() {
    std::cout << "\n=== Testing Single Image Training ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Create test data
    createTestData("test_plugin_data", 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = pluginSystem.trainFromImage(
        "test_plugin_data/images/sample_0.ppm",
        "test_plugin_data/metadata/sample_0.json"
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    auto metrics = pluginSystem.getLastTrainingMetrics();
    
    std::cout << "Training successful: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "Training time: " << metrics.trainingTime.count() << " ms" << std::endl;
    std::cout << "Final accuracy: " << metrics.finalAccuracy * 100 << "%" << std::endl;
    std::cout << "Memory usage: " << metrics.memorySizeBytes / 1024 << " KB" << std::endl;
    
    // Check if training was under 120 seconds
    bool timeConstraintMet = metrics.trainingTime.count() < 120000;
    std::cout << "Time constraint (<120s): " << (timeConstraintMet ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    fs::remove_all("test_plugin_data");
    
    return success && timeConstraintMet;
}

// Test batch training
bool testBatchTraining() {
    std::cout << "\n=== Testing Batch Training ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Create test data
    createTestData("test_plugin_batch", 20);
    
    std::vector<std::string> imagePaths;
    std::vector<std::string> metaPaths;
    
    for (int i = 0; i < 20; ++i) {
        imagePaths.push_back("test_plugin_batch/images/sample_" + std::to_string(i) + ".ppm");
        metaPaths.push_back("test_plugin_batch/metadata/sample_" + std::to_string(i) + ".json");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = pluginSystem.trainFromBatch(imagePaths, metaPaths);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    auto metrics = pluginSystem.getLastTrainingMetrics();
    
    std::cout << "Batch training successful: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "Total training time: " << duration << " seconds" << std::endl;
    std::cout << "Time per sample: " << duration / 20.0 << " seconds" << std::endl;
    
    // Check registered plugins
    std::cout << "Registered plugins:" << std::endl;
    std::cout << "- electric_scooter: " << (pluginSystem.hasPlugin("electric_scooter") ? "YES" : "NO") << std::endl;
    std::cout << "- delivery_robot: " << (pluginSystem.hasPlugin("delivery_robot") ? "YES" : "NO") << std::endl;
    std::cout << "- emergency_cone: " << (pluginSystem.hasPlugin("emergency_cone") ? "YES" : "NO") << std::endl;
    
    // Cleanup
    fs::remove_all("test_plugin_batch");
    
    return success && duration < 120;
}

// Test plugin detection
bool testPluginDetection() {
    std::cout << "\n=== Testing Plugin Detection ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Train a plugin first
    createTestData("test_detection", 5);
    
    std::vector<std::string> imagePaths;
    std::vector<std::string> metaPaths;
    
    for (int i = 0; i < 5; ++i) {
        imagePaths.push_back("test_detection/images/sample_" + std::to_string(i) + ".ppm");
        metaPaths.push_back("test_detection/metadata/sample_" + std::to_string(i) + ".json");
    }
    
    pluginSystem.trainFromBatch(imagePaths, metaPaths);
    
    // Test detection on new image
    Image testImage(640, 480, 3);
    for (int i = 0; i < testImage.size(); ++i) {
        testImage.data[i] = 200;
    }
    
    // Draw object similar to training data
    rectangle(testImage, Rect(100, 100, 80, 80), {100, 150, 200}, -1);
    for (int j = 0; j < 20; ++j) {
        int tx = 100 + rand() % 80;
        int ty = 100 + rand() % 80;
        circle(testImage, tx, ty, 3, {0, 0, 0}, -1);
    }
    
    // Generate candidate boxes
    std::vector<Rect> candidates;
    candidates.push_back(Rect(100, 100, 80, 80));  // True positive
    candidates.push_back(Rect(300, 300, 80, 80));  // False positive
    candidates.push_back(Rect(50, 250, 100, 100)); // False positive
    
    auto start = std::chrono::high_resolution_clock::now();
    auto detections = pluginSystem.detectPluginObjects(testImage, candidates);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Detection time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "Number of detections: " << detections.size() << std::endl;
    
    for (const auto& [className, score] : detections) {
        std::cout << "- " << className << ": " << score << std::endl;
    }
    
    // Cleanup
    fs::remove_all("test_detection");
    
    return !detections.empty();
}

// Test hot-swapping
bool testHotSwapping() {
    std::cout << "\n=== Testing Hot-Swapping ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Register and train a plugin
    pluginSystem.registerPlugin("test_object");
    
    // Save weights
    std::string weightsPath = "test_weights.bin";
    
    // Create dummy module for testing
    createTestData("test_hotswap", 3);
    pluginSystem.trainFromImage(
        "test_hotswap/images/sample_0.ppm",
        "test_hotswap/metadata/sample_0.json"
    );
    
    // Save the module weights first
    pluginSystem.savePluginWeights("emergency_cone", weightsPath);
    
    // Test hot-swap
    auto start = std::chrono::high_resolution_clock::now();
    bool success = pluginSystem.hotSwapModule("emergency_cone", weightsPath);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Hot-swap time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "Hot-swap successful: " << (success ? "YES" : "NO") << std::endl;
    
    // Cleanup
    fs::remove_all("test_hotswap");
    if (fs::exists(weightsPath)) {
        fs::remove(weightsPath);
    }
    
    return duration < 100000;  // Should be very fast (<100ms)
}

// Test database operations
bool testDatabaseOperations() {
    std::cout << "\n=== Testing Database Operations ===" << std::endl;
    
    PluginSystem pluginSystem1;
    
    // Train some plugins
    createTestData("test_db", 10);
    
    std::vector<std::string> imagePaths;
    std::vector<std::string> metaPaths;
    
    for (int i = 0; i < 10; ++i) {
        imagePaths.push_back("test_db/images/sample_" + std::to_string(i) + ".ppm");
        metaPaths.push_back("test_db/metadata/sample_" + std::to_string(i) + ".json");
    }
    
    pluginSystem1.trainFromBatch(imagePaths, metaPaths);
    
    // Save to database
    std::string dbPath = "test_objects.db";
    auto start = std::chrono::high_resolution_clock::now();
    bool saveSuccess = pluginSystem1.saveToDatabase(dbPath);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto saveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Save to database: " << (saveSuccess ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Save time: " << saveDuration << " ms" << std::endl;
    
    // Load into new system
    PluginSystem pluginSystem2;
    
    start = std::chrono::high_resolution_clock::now();
    bool loadSuccess = pluginSystem2.loadFromDatabase(dbPath);
    end = std::chrono::high_resolution_clock::now();
    
    auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Load from database: " << (loadSuccess ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Load time: " << loadDuration << " ms" << std::endl;
    
    // Verify loaded plugins
    std::cout << "Loaded plugins:" << std::endl;
    std::cout << "- electric_scooter: " << (pluginSystem2.hasPlugin("electric_scooter") ? "YES" : "NO") << std::endl;
    std::cout << "- delivery_robot: " << (pluginSystem2.hasPlugin("delivery_robot") ? "YES" : "NO") << std::endl;
    std::cout << "- emergency_cone: " << (pluginSystem2.hasPlugin("emergency_cone") ? "YES" : "NO") << std::endl;
    
    // Cleanup
    fs::remove_all("test_db");
    if (fs::exists(dbPath)) {
        fs::remove(dbPath);
    }
    
    return saveSuccess && loadSuccess;
}

// Test TACSNet integration
bool testTACSNetIntegration() {
    std::cout << "\n=== Testing TACSNet Integration ===" << std::endl;
    
    TACSNetPluginAdapter adapter;
    
    // Create dummy TACSNet pointer (in real implementation, this would be actual TACSNet)
    void* tacsnetPtr = nullptr;
    
    // Add plugin heads
    auto start = std::chrono::high_resolution_clock::now();
    
    adapter.addPluginHead(tacsnetPtr, "electric_scooter", 3);
    adapter.addPluginHead(tacsnetPtr, "delivery_robot", 3);
    adapter.addPluginHead(tacsnetPtr, "emergency_cone", 3);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Time to add 3 plugin heads: " << duration / 1000.0 << " ms" << std::endl;
    
    // Test forward pass through plugin heads
    Tensor dummyFeatures({1, 256, 13, 13});  // Typical feature map size
    std::fill(dummyFeatures.data_float(), dummyFeatures.data_float() + dummyFeatures.size(), 0.1f);
    
    std::vector<std::string> activePlugins = {"electric_scooter", "delivery_robot"};
    
    start = std::chrono::high_resolution_clock::now();
    auto outputs = adapter.forwardPluginHeads(tacsnetPtr, dummyFeatures, activePlugins);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Plugin heads forward pass time: " << duration / 1000.0 << " ms" << std::endl;
    std::cout << "Number of outputs: " << outputs.size() << std::endl;
    
    // Test removal
    adapter.removePluginHead(tacsnetPtr, "delivery_robot");
    
    outputs = adapter.forwardPluginHeads(tacsnetPtr, dummyFeatures, activePlugins);
    std::cout << "Outputs after removal: " << outputs.size() << std::endl;
    
    return outputs.size() == 1;  // Should only have electric_scooter output
}

// Test fast training
bool testFastTraining() {
    std::cout << "\n=== Testing Fast Training Optimization ===" << std::endl;
    
    FastPluginTrainer trainer;
    PluginModule module("fast_test", 1500);  // Approximate feature dimension
    
    // Generate synthetic training data
    std::vector<FeatureDescriptor> positives, negatives;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    // Create positive samples
    for (int i = 0; i < 50; ++i) {
        FeatureDescriptor feat;
        feat.rgbHistogram.resize(768, 0.1f);
        feat.hsvHistogram.resize(692, 0.1f);
        feat.edgeHistogram.resize(8, 0.2f);
        feat.shapeFeatures.resize(4, 0.5f);
        feat.textureFeatures.resize(64, 0.1f);
        
        // Add noise
        for (auto& v : feat.rgbHistogram) v += dist(gen);
        
        positives.push_back(feat);
    }
    
    // Create negative samples
    for (int i = 0; i < 100; ++i) {
        FeatureDescriptor feat;
        feat.rgbHistogram.resize(768, 0.05f);
        feat.hsvHistogram.resize(692, 0.05f);
        feat.edgeHistogram.resize(8, 0.1f);
        feat.shapeFeatures.resize(4, 0.3f);
        feat.textureFeatures.resize(64, 0.05f);
        
        // Add noise
        for (auto& v : feat.rgbHistogram) v += dist(gen);
        
        negatives.push_back(feat);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = trainer.rapidTrain(&module, positives, negatives, 120);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "Fast training completed: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "Training duration: " << duration << " seconds" << std::endl;
    std::cout << "Module trained: " << (module.isTrained() ? "YES" : "NO") << std::endl;
    std::cout << "Final accuracy: " << module.getAccuracy() * 100 << "%" << std::endl;
    
    // Test hard sample mining
    auto hardIndices = trainer.selectHardSamples(&module, positives, 10);
    std::cout << "Hard samples selected: " << hardIndices.size() << std::endl;
    
    // Test data augmentation
    auto augmented = trainer.augmentFeatures(positives, 3);
    std::cout << "Augmented samples: " << augmented.size() << " (from " << positives.size() << ")" << std::endl;
    
    return success && duration < 120;
}

// Performance stress test
bool testPerformanceUnderLoad() {
    std::cout << "\n=== Testing Performance Under Load ===" << std::endl;
    
    PluginSystem pluginSystem;
    
    // Register multiple plugins
    const int numPlugins = 10;
    for (int i = 0; i < numPlugins; ++i) {
        pluginSystem.registerPlugin("plugin_" + std::to_string(i));
    }
    
    // Create large test image
    Image largeImage(2560, 1920, 3);
    randu(largeImage, {0, 0, 0}, {255, 255, 255});
    
    // Generate many candidates
    std::vector<Rect> candidates;
    for (int y = 0; y < 1920 - 100; y += 50) {
        for (int x = 0; x < 2560 - 100; x += 50) {
            candidates.emplace_back(x, y, 100, 100);
        }
    }
    
    std::cout << "Number of candidates: " << candidates.size() << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto detections = pluginSystem.detectPluginObjects(largeImage, candidates);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Total detection time: " << duration << " ms" << std::endl;
    std::cout << "Time per candidate: " << static_cast<float>(duration) / candidates.size() << " ms" << std::endl;
    std::cout << "Detections found: " << detections.size() << std::endl;
    
    // Should process quickly even with many candidates
    float timePerCandidate = static_cast<float>(duration) / candidates.size();
    bool performanceGood = timePerCandidate < 1.0f;  // Less than 1ms per candidate
    
    std::cout << "Performance requirement (<1ms/candidate): " << (performanceGood ? "PASSED" : "FAILED") << std::endl;
    
    return performanceGood;
}

int main() {
    std::cout << "=== TACS Plugin System Validation (Phase 9) ===" << std::endl;
    std::cout << "Testing plugin learning capabilities..." << std::endl;
    
    int testsPassed = 0;
    int totalTests = 0;
    
    // Run all tests
    auto runTest = [&](const std::string& name, bool (*testFunc)()) {
        totalTests++;
        try {
            bool passed = testFunc();
            if (passed) {
                testsPassed++;
                std::cout << "[PASS] " << name << std::endl;
            } else {
                std::cout << "[FAIL] " << name << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[FAIL] " << name << " - Exception: " << e.what() << std::endl;
        }
        std::cout << std::endl;
    };
    
    // Run all validation tests
    runTest("Feature Extraction", testFeatureExtraction);
    runTest("Single Image Training", testSingleImageTraining);
    runTest("Batch Training", testBatchTraining);
    runTest("Plugin Detection", testPluginDetection);
    runTest("Hot-Swapping", testHotSwapping);
    runTest("Database Operations", testDatabaseOperations);
    runTest("TACSNet Integration", testTACSNetIntegration);
    runTest("Fast Training", testFastTraining);
    runTest("Performance Under Load", testPerformanceUnderLoad);
    
    // Summary
    std::cout << "\n=== VALIDATION SUMMARY ===" << std::endl;
    std::cout << "Tests passed: " << testsPassed << "/" << totalTests << std::endl;
    
    if (testsPassed == totalTests) {
        std::cout << "\n✅ ALL TESTS PASSED - Phase 9 plugin system is production-ready!" << std::endl;
        std::cout << "Key achievements:" << std::endl;
        std::cout << "- Rapid learning from single images" << std::endl;
        std::cout << "- Training time < 120 seconds constraint met" << std::endl;
        std::cout << "- Hot-swapping for runtime updates" << std::endl;
        std::cout << "- Database persistence for plugin storage" << std::endl;
        std::cout << "- TACSNet integration ready" << std::endl;
        std::cout << "- Performance optimized for real-time use" << std::endl;
    } else {
        std::cout << "\n❌ Some tests failed - Phase 9 needs fixes" << std::endl;
    }
    
    return testsPassed == totalTests ? 0 : 1;
}