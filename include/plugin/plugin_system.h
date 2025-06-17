// Plugin system for rapid learning of new object types
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include "../core/tensor.h"
#include "../utils/image_processing.h"
#include "../layers/linear.h"
#include "../layers/conv2d.h"

namespace tacs {

// Metadata structure for plugin objects
struct PluginMetadata {
    std::string className;
    std::vector<float> bbox;  // [x, y, w, h]
    std::string imagePath;
    int64_t timestamp;
    std::unordered_map<std::string, std::string> attributes;
};

// Feature descriptor for object recognition
struct FeatureDescriptor {
    std::vector<float> rgbHistogram;     // 256x3 bins
    std::vector<float> hsvHistogram;     // 180+256+256 bins
    std::vector<float> edgeHistogram;    // 8 orientation bins
    std::vector<float> shapeFeatures;    // aspect ratio, area, perimeter
    std::vector<float> textureFeatures;  // LBP or similar
};

// Plugin module for new object detection
class PluginModule {
public:
    PluginModule(const std::string& className, int featureDim);
    ~PluginModule() = default;

    // Train on new samples
    void addTrainingSample(const FeatureDescriptor& features, bool positive);
    void train(int maxIterations = 1000, float learningRate = 0.01f);
    
    // Inference
    float predict(const FeatureDescriptor& features) const;
    std::vector<float> getProbabilities(const FeatureDescriptor& features) const;
    
    // Model management
    void saveWeights(const std::string& path) const;
    void loadWeights(const std::string& path);
    
    std::string getClassName() const { return className_; }
    bool isTrained() const { return trained_; }
    float getAccuracy() const { return accuracy_; }
    
private:
    std::string className_;
    int featureDim_;
    bool trained_;
    float accuracy_;
    
    // Shallow FC network components
    std::unique_ptr<LinearLayer> fc1_;
    std::unique_ptr<LinearLayer> fc2_;
    std::unique_ptr<LinearLayer> fc3_;
    
    // Training data
    std::vector<FeatureDescriptor> positiveFeatures_;
    std::vector<FeatureDescriptor> negativeFeatures_;
    
    // Weight initialization
    void initializeWeights();
    
    // Friend classes for access
    friend class FastPluginTrainer;
    friend class PluginSystem;
};

// Main plugin system manager
class PluginSystem {
public:
    PluginSystem();
    ~PluginSystem() = default;

    // Plugin management
    bool registerPlugin(const std::string& className);
    bool unregisterPlugin(const std::string& className);
    bool hasPlugin(const std::string& className) const;
    
    // Training interface
    bool trainFromImage(const std::string& imagePath, const std::string& metaPath);
    bool trainFromBatch(const std::vector<std::string>& imagePaths, 
                       const std::vector<std::string>& metaPaths);
    
    // Feature extraction
    FeatureDescriptor extractFeatures(const Image& image, const Rect& bbox) const;
    
    // Inference
    std::vector<std::pair<std::string, float>> detectPluginObjects(
        const Image& image, 
        const std::vector<Rect>& candidates) const;
    
    // Hot-swapping and runtime updates
    bool savePluginWeights(const std::string& className, const std::string& weightsPath);
    bool hotSwapModule(const std::string& className, const std::string& weightsPath);
    void updateTACSNetHeads(void* tacsnetPtr);
    
    // Database operations
    bool saveToDatabase(const std::string& dbPath);
    bool loadFromDatabase(const std::string& dbPath);
    
    // Performance metrics
    struct TrainingMetrics {
        std::chrono::milliseconds trainingTime;
        float finalAccuracy;
        int iterations;
        size_t memorySizeBytes;
    };
    
    TrainingMetrics getLastTrainingMetrics() const { return lastMetrics_; }
    
private:
    std::unordered_map<std::string, std::unique_ptr<PluginModule>> plugins_;
    mutable std::mutex pluginMutex_;
    TrainingMetrics lastMetrics_;
    
    // Feature extraction helpers
    std::vector<float> computeRGBHistogram(const Image& roi) const;
    std::vector<float> computeHSVHistogram(const Image& roi) const;
    std::vector<float> computeEdgeHistogram(const Image& roi) const;
    std::vector<float> computeShapeFeatures(const Image& roi) const;
    std::vector<float> computeTextureFeatures(const Image& roi) const;
    
    // JSON parsing
    PluginMetadata parseMetadata(const std::string& jsonPath) const;
    
    // Negative sample mining
    std::vector<Rect> generateNegativeSamples(const Image& image, 
                                              const Rect& positiveBbox,
                                              int numSamples = 10) const;
};

// Integration with TACSNet for modular heads
class TACSNetPluginAdapter {
public:
    TACSNetPluginAdapter();
    ~TACSNetPluginAdapter() = default;
    
    // Add plugin detection head to TACSNet
    void addPluginHead(void* tacsnetPtr, const std::string& className, 
                      int numAnchors = 3);
    
    // Remove plugin head
    void removePluginHead(void* tacsnetPtr, const std::string& className);
    
    // Forward pass through plugin heads
    std::vector<Tensor> forwardPluginHeads(void* tacsnetPtr, 
                                          const Tensor& features,
                                          const std::vector<std::string>& activePlugins);
    
    // Update weights from plugin system
    void syncWeights(void* tacsnetPtr, const PluginSystem& pluginSystem);
    
private:
    struct PluginHead {
        std::string className;
        std::unique_ptr<layers::Conv2D> convLayer;
        int outputChannels;
    };
    
    std::unordered_map<std::string, PluginHead> pluginHeads_;
    mutable std::mutex headMutex_;
};

// Fast training optimizer for <120s constraint
class FastPluginTrainer {
public:
    FastPluginTrainer();
    ~FastPluginTrainer() = default;
    
    // Optimized training with early stopping
    bool rapidTrain(PluginModule* module,
                   const std::vector<FeatureDescriptor>& positives,
                   const std::vector<FeatureDescriptor>& negatives,
                   int timeoutSeconds = 120);
    
    // Active learning sample selection
    std::vector<int> selectHardSamples(const PluginModule* module,
                                      const std::vector<FeatureDescriptor>& features,
                                      int numSamples);
    
    // Data augmentation for small datasets
    std::vector<FeatureDescriptor> augmentFeatures(
        const std::vector<FeatureDescriptor>& original,
        int augmentationFactor = 5);
    
private:
    // Mini-batch SGD with momentum
    void miniBatchUpdate(PluginModule* module,
                        const std::vector<FeatureDescriptor>& features,
                        const std::vector<int>& labels,
                        const std::vector<int>& indices,
                        float learningRate,
                        float momentum);
    
    // Learning rate scheduling
    float adaptiveLearningRate(int epoch, float initialRate) const;
    
    // Early stopping criteria
    bool shouldStop(const std::vector<float>& losses, float threshold = 1e-4f) const;
};

} // namespace tacs