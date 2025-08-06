//
//  test_cpu_backend_implementation.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <memory>

// Mock implementations for testing our CPU backend without full MNN dependencies

namespace MNN {

enum ErrorCode {
    NO_ERROR = 0,
    OUT_OF_MEMORY = 1,
    INVALID_VALUE = 2
};

enum MNNForwardType {
    MNN_FORWARD_CPU = 0
};

// Mock Tensor implementation
class Tensor {
public:
    Tensor(const std::vector<int>& shape) : mShape(shape) {
        mSize = 1;
        for (int dim : shape) {
            mSize *= dim;
        }
        mData.resize(mSize);
    }
    
    template<typename T>
    T* host() { return reinterpret_cast<T*>(mData.data()); }
    
    template<typename T>
    const T* host() const { return reinterpret_cast<const T*>(mData.data()); }
    
    int batch() const { return mShape.size() > 0 ? mShape[0] : 1; }
    int channel() const { return mShape.size() > 1 ? mShape[1] : 1; }
    int height() const { return mShape.size() > 2 ? mShape[2] : 1; }
    int width() const { return mShape.size() > 3 ? mShape[3] : 1; }
    int elementSize() const { return mSize; }
    const std::vector<int>& shape() const { return mShape; }

private:
    std::vector<int> mShape;
    std::vector<float> mData;
    int mSize;
};

// Mock MemChunk
class MemChunk {
public:
    MemChunk() : mPtr(nullptr) {}
    MemChunk(void* ptr) : mPtr(ptr) {}
    
    template<typename T>
    T* ptr() const { return reinterpret_cast<T*>(mPtr); }
    
    bool invalid() const { return mPtr == nullptr; }

private:
    void* mPtr;
};

// Mock BufferAllocator
class BufferAllocator {
public:
    virtual ~BufferAllocator() = default;
    virtual MemChunk alloc(size_t size, bool separate = false) {
        return MemChunk(malloc(size));
    }
    virtual void free(const MemChunk& chunk) {
        if (chunk.ptr<void>()) {
            ::free(chunk.ptr<void>());
        }
    }
};

// Mock Backend
class Backend {
public:
    Backend(MNNForwardType type) : mType(type) {}
    virtual ~Backend() = default;
    
    BufferAllocator* getBufferAllocator() {
        if (!mAllocator) {
            mAllocator.reset(new BufferAllocator);
        }
        return mAllocator.get();
    }

private:
    MNNForwardType mType;
    std::unique_ptr<BufferAllocator> mAllocator;
};

// Mock Execution base class
class Execution {
public:
    Execution(Backend* backend) : mBackend(backend) {}
    virtual ~Execution() = default;
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) = 0;

protected:
    Backend* backend() const { return mBackend; }

private:
    Backend* mBackend;
};

// Mock concurrency macro
#define MNN_CONCURRENCY_BEGIN(var, limit) for (int var = 0; var < limit; ++var) {
#define MNN_CONCURRENCY_END() }

} // namespace MNN

// Include our implementation
#include "source/backend/cpu/CPUInstanceNormGrad.hpp"

using namespace MNN;

class CPUInstanceNormGradTester {
private:
    void fillRandomData(Tensor* tensor, std::mt19937& gen, float min_val = -2.0f, float max_val = 2.0f) {
        std::uniform_real_distribution<float> dis(min_val, max_val);
        float* data = tensor->host<float>();
        for (int i = 0; i < tensor->elementSize(); ++i) {
            data[i] = dis(gen);
        }
    }
    
    void printTensor(const Tensor* tensor, const std::string& name, int maxElements = 8) {
        std::cout << name << " (";
        const auto& shape = tensor->shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << "x";
        }
        std::cout << "): ";
        
        const float* data = tensor->host<float>();
        int size = std::min(maxElements, tensor->elementSize());
        
        for (int i = 0; i < size; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i < size - 1) std::cout << ", ";
        }
        if (size < tensor->elementSize()) {
            std::cout << " ... (+" << (tensor->elementSize() - size) << " more)";
        }
        std::cout << std::endl;
    }

public:
    bool runTest() {
        std::cout << "=== CPUInstanceNormGrad Backend Test ===" << std::endl;
        
        // Test configuration
        const int batch = 2;
        const int channel = 3;
        const int height = 4;
        const int width = 4;
        const float epsilon = 1e-5f;
        
        std::cout << "Configuration: batch=" << batch << ", channel=" << channel 
                  << ", height=" << height << ", width=" << width 
                  << ", epsilon=" << epsilon << std::endl << std::endl;
        
        // Create backend and operation
        Backend backend(MNN_FORWARD_CPU);
        CPUInstanceNormGrad gradOp(&backend, epsilon);
        
        // Create tensors
        auto inputTensor = std::make_unique<Tensor>(std::vector<int>{batch, channel, height, width});
        auto outputGradTensor = std::make_unique<Tensor>(std::vector<int>{batch, channel, height, width});
        auto gammaTensor = std::make_unique<Tensor>(std::vector<int>{channel});
        auto betaTensor = std::make_unique<Tensor>(std::vector<int>{channel});
        
        auto inputGradTensor = std::make_unique<Tensor>(std::vector<int>{batch, channel, height, width});
        auto gammaGradTensor = std::make_unique<Tensor>(std::vector<int>{channel});
        auto betaGradTensor = std::make_unique<Tensor>(std::vector<int>{channel});
        
        // Fill with test data
        std::mt19937 gen(42);
        fillRandomData(inputTensor.get(), gen, -2.0f, 2.0f);
        fillRandomData(outputGradTensor.get(), gen, -1.0f, 1.0f);
        fillRandomData(gammaTensor.get(), gen, 0.5f, 1.5f);
        fillRandomData(betaTensor.get(), gen, -0.5f, 0.5f);
        
        // Print input data
        std::cout << "=== Input Data ===" << std::endl;
        printTensor(inputTensor.get(), "Input");
        printTensor(outputGradTensor.get(), "Output Gradient");
        printTensor(gammaTensor.get(), "Gamma", channel);
        printTensor(betaTensor.get(), "Beta", channel);
        std::cout << std::endl;
        
        // Setup inputs and outputs
        std::vector<Tensor*> inputs = {inputTensor.get(), outputGradTensor.get(), 
                                      gammaTensor.get(), betaTensor.get()};
        std::vector<Tensor*> outputs = {inputGradTensor.get(), gammaGradTensor.get(), 
                                       betaGradTensor.get()};
        
        // Test onResize
        std::cout << "=== Testing onResize ===" << std::endl;
        auto resizeError = gradOp.onResize(inputs, outputs);
        if (resizeError != NO_ERROR) {
            std::cout << "❌ onResize failed with error: " << resizeError << std::endl;
            return false;
        }
        std::cout << "✅ onResize successful" << std::endl << std::endl;
        
        // Test onExecute
        std::cout << "=== Testing onExecute ===" << std::endl;
        auto executeError = gradOp.onExecute(inputs, outputs);
        if (executeError != NO_ERROR) {
            std::cout << "❌ onExecute failed with error: " << executeError << std::endl;
            return false;
        }
        std::cout << "✅ onExecute successful" << std::endl << std::endl;
        
        // Print results
        std::cout << "=== Results ===" << std::endl;
        printTensor(inputGradTensor.get(), "Input Gradient");
        printTensor(gammaGradTensor.get(), "Gamma Gradient", channel);
        printTensor(betaGradTensor.get(), "Beta Gradient", channel);
        std::cout << std::endl;
        
        // Sanity checks
        std::cout << "=== Sanity Checks ===" << std::endl;
        
        bool allFinite = true;
        
        // Check input gradients
        const float* inputGrad = inputGradTensor->host<float>();
        for (int i = 0; i < inputGradTensor->elementSize(); ++i) {
            if (!std::isfinite(inputGrad[i])) {
                std::cout << "❌ Input gradient contains non-finite value at index " << i 
                          << ": " << inputGrad[i] << std::endl;
                allFinite = false;
                break;
            }
        }
        if (allFinite) {
            std::cout << "✅ All input gradients are finite" << std::endl;
        }
        
        // Check gamma gradients
        const float* gammaGrad = gammaGradTensor->host<float>();
        for (int i = 0; i < gammaGradTensor->elementSize(); ++i) {
            if (!std::isfinite(gammaGrad[i])) {
                std::cout << "❌ Gamma gradient contains non-finite value at index " << i 
                          << ": " << gammaGrad[i] << std::endl;
                allFinite = false;
                break;
            }
        }
        if (allFinite) {
            std::cout << "✅ All gamma gradients are finite" << std::endl;
        }
        
        // Check beta gradients
        const float* betaGrad = betaGradTensor->host<float>();
        for (int i = 0; i < betaGradTensor->elementSize(); ++i) {
            if (!std::isfinite(betaGrad[i])) {
                std::cout << "❌ Beta gradient contains non-finite value at index " << i 
                          << ": " << betaGrad[i] << std::endl;
                allFinite = false;
                break;
            }
        }
        if (allFinite) {
            std::cout << "✅ All beta gradients are finite" << std::endl;
        }
        
        // Check reasonable magnitudes
        float maxInputGrad = 0.0f, maxGammaGrad = 0.0f, maxBetaGrad = 0.0f;
        
        for (int i = 0; i < inputGradTensor->elementSize(); ++i) {
            maxInputGrad = std::max(maxInputGrad, std::abs(inputGrad[i]));
        }
        
        for (int i = 0; i < gammaGradTensor->elementSize(); ++i) {
            maxGammaGrad = std::max(maxGammaGrad, std::abs(gammaGrad[i]));
        }
        
        for (int i = 0; i < betaGradTensor->elementSize(); ++i) {
            maxBetaGrad = std::max(maxBetaGrad, std::abs(betaGrad[i]));
        }
        
        std::cout << "Max gradient magnitudes:" << std::endl;
        std::cout << "  Input: " << maxInputGrad << std::endl;
        std::cout << "  Gamma: " << maxGammaGrad << std::endl;
        std::cout << "  Beta: " << maxBetaGrad << std::endl;
        
        bool reasonableMagnitudes = (maxInputGrad < 1000.0f) && 
                                   (maxGammaGrad < 1000.0f) && 
                                   (maxBetaGrad < 1000.0f);
        
        if (reasonableMagnitudes) {
            std::cout << "✅ All gradients have reasonable magnitudes" << std::endl;
        } else {
            std::cout << "⚠️  Some gradients have very large magnitudes" << std::endl;
        }
        
        // Simple mathematical property check
        // Beta gradient should equal sum of output gradients for each channel
        std::cout << "\n=== Mathematical Property Check ===" << std::endl;
        
        const float* outputGradData = outputGradTensor->host<float>();
        bool betaCheckPassed = true;
        
        for (int c = 0; c < channel; ++c) {
            float expectedBetaGrad = 0.0f;
            for (int b = 0; b < batch; ++b) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int idx = ((b * channel + c) * height + h) * width + w;
                        expectedBetaGrad += outputGradData[idx];
                    }
                }
            }
            
            float actualBetaGrad = betaGrad[c];
            float error = std::abs(expectedBetaGrad - actualBetaGrad);
            
            std::cout << "Channel " << c << " beta gradient: expected=" << expectedBetaGrad 
                      << ", actual=" << actualBetaGrad << ", error=" << error << std::endl;
            
            if (error > 1e-4f) {
                betaCheckPassed = false;
            }
        }
        
        if (betaCheckPassed) {
            std::cout << "✅ Beta gradient mathematical property check passed" << std::endl;
        } else {
            std::cout << "❌ Beta gradient mathematical property check failed" << std::endl;
        }
        
        bool overallSuccess = allFinite && reasonableMagnitudes && betaCheckPassed;
        
        std::cout << "\n=== Final Result ===" << std::endl;
        std::cout << "Test " << (overallSuccess ? "PASSED ✅" : "FAILED ❌") << std::endl;
        
        return overallSuccess;
    }
};

int main() {
    CPUInstanceNormGradTester tester;
    bool result = tester.runTest();
    return result ? 0 : 1;
}