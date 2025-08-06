//
//  test_cpu_instancenorm_grad.cpp
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
#include "source/backend/cpu/CPUInstanceNormGrad.hpp"
#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"

using namespace MNN;

// Helper function to create a test tensor
Tensor* createTensor(const std::vector<int>& shape, Tensor::DimensionType dimType = Tensor::CAFFE) {
    Tensor* tensor = Tensor::create(shape, halide_type_of<float>(), nullptr, dimType);
    return tensor;
}

// Helper function to fill tensor with random data
void fillRandomData(Tensor* tensor, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    float* data = tensor->host<float>();
    int size = tensor->elementSize();
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Helper function to print tensor data
void printTensor(const Tensor* tensor, const std::string& name, int maxElements = 10) {
    std::cout << name << " (shape: ";
    auto& shape = tensor->shape();
    for (int i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << "x";
    }
    std::cout << "):\n";
    
    const float* data = tensor->host<float>();
    int size = std::min(maxElements, tensor->elementSize());
    
    for (int i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i];
        if (i < size - 1) std::cout << ", ";
        if ((i + 1) % 8 == 0) std::cout << "\n";
    }
    if (size < tensor->elementSize()) {
        std::cout << " ... (+" << (tensor->elementSize() - size) << " more)";
    }
    std::cout << "\n\n";
}

// Numerical gradient computation for verification
void computeNumericalGradient(const std::vector<float>& input, 
                             const std::vector<float>& gamma,
                             const std::vector<float>& beta,
                             const std::vector<float>& outputGrad,
                             int batch, int channel, int spatial,
                             float epsilon, float h,
                             std::vector<float>& inputGrad,
                             std::vector<float>& gammaGrad,
                             std::vector<float>& betaGrad) {
    
    auto forwardPass = [&](const std::vector<float>& x, 
                          const std::vector<float>& g, 
                          const std::vector<float>& b) -> std::vector<float> {
        std::vector<float> output(x.size());
        
        for (int n = 0; n < batch; ++n) {
            for (int c = 0; c < channel; ++c) {
                // Compute mean and variance for this instance-channel
                float sum = 0.0f;
                float sumSq = 0.0f;
                int idx_base = (n * channel + c) * spatial;
                
                for (int s = 0; s < spatial; ++s) {
                    float val = x[idx_base + s];
                    sum += val;
                    sumSq += val * val;
                }
                
                float mean = sum / spatial;
                float variance = sumSq / spatial - mean * mean;
                float stdInv = 1.0f / std::sqrt(variance + epsilon);
                
                // Apply normalization
                for (int s = 0; s < spatial; ++s) {
                    int idx = idx_base + s;
                    float normalized = (x[idx] - mean) * stdInv;
                    output[idx] = normalized * g[c] + b[c];
                }
            }
        }
        return output;
    };
    
    // Compute input gradients
    inputGrad.resize(input.size());
    for (int i = 0; i < input.size(); ++i) {
        std::vector<float> inputPlus = input;
        std::vector<float> inputMinus = input;
        inputPlus[i] += h;
        inputMinus[i] -= h;
        
        auto outputPlus = forwardPass(inputPlus, gamma, beta);
        auto outputMinus = forwardPass(inputMinus, gamma, beta);
        
        float grad = 0.0f;
        for (int j = 0; j < outputGrad.size(); ++j) {
            grad += outputGrad[j] * (outputPlus[j] - outputMinus[j]) / (2.0f * h);
        }
        inputGrad[i] = grad;
    }
    
    // Compute gamma gradients
    gammaGrad.resize(gamma.size());
    for (int c = 0; c < channel; ++c) {
        std::vector<float> gammaPlus = gamma;
        std::vector<float> gammaMinus = gamma;
        gammaPlus[c] += h;
        gammaMinus[c] -= h;
        
        auto outputPlus = forwardPass(input, gammaPlus, beta);
        auto outputMinus = forwardPass(input, gammaMinus, beta);
        
        float grad = 0.0f;
        for (int j = 0; j < outputGrad.size(); ++j) {
            grad += outputGrad[j] * (outputPlus[j] - outputMinus[j]) / (2.0f * h);
        }
        gammaGrad[c] = grad;
    }
    
    // Compute beta gradients
    betaGrad.resize(beta.size());
    for (int c = 0; c < channel; ++c) {
        std::vector<float> betaPlus = beta;
        std::vector<float> betaMinus = beta;
        betaPlus[c] += h;
        betaMinus[c] -= h;
        
        auto outputPlus = forwardPass(input, gamma, betaPlus);
        auto outputMinus = forwardPass(input, gamma, betaMinus);
        
        float grad = 0.0f;
        for (int j = 0; j < outputGrad.size(); ++j) {
            grad += outputGrad[j] * (outputPlus[j] - outputMinus[j]) / (2.0f * h);
        }
        betaGrad[c] = grad;
    }
}

class MockCPUBackend : public Backend {
public:
    MockCPUBackend() : Backend(MNN_FORWARD_CPU) {}
    virtual ~MockCPUBackend() = default;
    
    virtual MemObj* onAcquire(const Tensor* tensor, StorageType storageType) override {
        return nullptr;
    }
    
    virtual bool onClearBuffer() override { return true; }
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) override {}
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override {
        return std::make_pair(0.0f, false);
    }
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op) override {
        return nullptr;
    }
    virtual void onExecuteBegin() const override {}
    virtual void onExecuteEnd() const override {}
    virtual bool onReleaseBuffer(const Tensor* tensor, StorageType storageType) override { return true; }
    virtual void onResizeBegin() override {}
    virtual void onResizeEnd() override {}
    
    // Mock buffer allocator
    class MockBufferAllocator : public BufferAllocator {
    public:
        virtual MemChunk alloc(size_t size, bool separate = false) override {
            void* ptr = malloc(size);
            return MemChunk(ptr);
        }
        virtual void free(const MemChunk& chunk) override {
            if (chunk.ptr()) {
                ::free(chunk.ptr());
            }
        }
        virtual void release(bool allRelease = true) override {}
        virtual void barrierBegin() override {}
        virtual void barrierEnd() override {}
    };
    
    MockBufferAllocator* getBufferAllocator() {
        if (!mAllocator) {
            mAllocator.reset(new MockBufferAllocator);
        }
        return mAllocator.get();
    }
    
private:
    std::unique_ptr<MockBufferAllocator> mAllocator;
};

int main() {
    std::cout << "=== CPUInstanceNormGrad Test ===" << std::endl;
    
    // Test parameters
    const int batch = 2;
    const int channel = 3;
    const int height = 4;
    const int width = 4;
    const int spatial = height * width;
    const float epsilon = 1e-5f;
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Batch: " << batch << std::endl;
    std::cout << "  Channel: " << channel << std::endl;
    std::cout << "  Height: " << height << std::endl;
    std::cout << "  Width: " << width << std::endl;
    std::cout << "  Epsilon: " << epsilon << std::endl << std::endl;
    
    // Create mock backend
    MockCPUBackend backend;
    
    // Create tensors
    std::vector<int> inputShape = {batch, channel, height, width};
    std::vector<int> paramShape = {channel};
    
    auto inputTensor = createTensor(inputShape);
    auto outputGradTensor = createTensor(inputShape);
    auto gammaTensor = createTensor(paramShape);
    auto betaTensor = createTensor(paramShape);
    
    auto inputGradTensor = createTensor(inputShape);
    auto gammaGradTensor = createTensor(paramShape);
    auto betaGradTensor = createTensor(paramShape);
    
    // Fill with test data
    fillRandomData(inputTensor, -2.0f, 2.0f);
    fillRandomData(outputGradTensor, -1.0f, 1.0f);
    fillRandomData(gammaTensor, 0.5f, 1.5f);
    fillRandomData(betaTensor, -0.5f, 0.5f);
    
    // Print input data
    printTensor(inputTensor, "Input", 16);
    printTensor(outputGradTensor, "Output Gradient", 16);
    printTensor(gammaTensor, "Gamma", channel);
    printTensor(betaTensor, "Beta", channel);
    
    // Create CPUInstanceNormGrad
    CPUInstanceNormGrad gradOp(&backend, epsilon);
    
    // Setup inputs and outputs
    std::vector<Tensor*> inputs = {inputTensor, outputGradTensor, gammaTensor, betaTensor};
    std::vector<Tensor*> outputs = {inputGradTensor, gammaGradTensor, betaGradTensor};
    
    // Resize
    auto resizeError = gradOp.onResize(inputs, outputs);
    if (resizeError != NO_ERROR) {
        std::cout << "Error in onResize: " << resizeError << std::endl;
        return -1;
    }
    
    // Execute
    auto executeError = gradOp.onExecute(inputs, outputs);
    if (executeError != NO_ERROR) {
        std::cout << "Error in onExecute: " << executeError << std::endl;
        return -1;
    }
    
    std::cout << "=== CPUInstanceNormGrad Results ===" << std::endl;
    printTensor(inputGradTensor, "Input Gradient", 16);
    printTensor(gammaGradTensor, "Gamma Gradient", channel);
    printTensor(betaGradTensor, "Beta Gradient", channel);
    
    // Numerical verification (for small test case)
    if (batch * channel * spatial <= 64) { // Only for small cases to avoid long computation
        std::cout << "=== Numerical Verification ===" << std::endl;
        
        // Convert tensor data to vectors
        std::vector<float> inputVec(inputTensor->host<float>(), 
                                   inputTensor->host<float>() + inputTensor->elementSize());
        std::vector<float> outputGradVec(outputGradTensor->host<float>(), 
                                        outputGradTensor->host<float>() + outputGradTensor->elementSize());
        std::vector<float> gammaVec(gammaTensor->host<float>(), 
                                   gammaTensor->host<float>() + gammaTensor->elementSize());
        std::vector<float> betaVec(betaTensor->host<float>(), 
                                  betaTensor->host<float>() + betaTensor->elementSize());
        
        std::vector<float> numInputGrad, numGammaGrad, numBetaGrad;
        computeNumericalGradient(inputVec, gammaVec, betaVec, outputGradVec,
                               batch, channel, spatial, epsilon, 1e-4f,
                               numInputGrad, numGammaGrad, numBetaGrad);
        
        // Compare results
        float maxInputError = 0.0f, maxGammaError = 0.0f, maxBetaError = 0.0f;
        
        const float* analyticalInputGrad = inputGradTensor->host<float>();
        const float* analyticalGammaGrad = gammaGradTensor->host<float>();
        const float* analyticalBetaGrad = betaGradTensor->host<float>();
        
        for (int i = 0; i < numInputGrad.size(); ++i) {
            float error = std::abs(analyticalInputGrad[i] - numInputGrad[i]);
            maxInputError = std::max(maxInputError, error);
        }
        
        for (int i = 0; i < numGammaGrad.size(); ++i) {
            float error = std::abs(analyticalGammaGrad[i] - numGammaGrad[i]);
            maxGammaError = std::max(maxGammaError, error);
        }
        
        for (int i = 0; i < numBetaGrad.size(); ++i) {
            float error = std::abs(analyticalBetaGrad[i] - numBetaGrad[i]);
            maxBetaError = std::max(maxBetaError, error);
        }
        
        std::cout << "Maximum errors (analytical vs numerical):" << std::endl;
        std::cout << "  Input gradient: " << std::scientific << maxInputError << std::endl;
        std::cout << "  Gamma gradient: " << std::scientific << maxGammaError << std::endl;
        std::cout << "  Beta gradient: " << std::scientific << maxBetaError << std::endl;
        
        const float tolerance = 1e-3f;
        bool passed = (maxInputError < tolerance) && 
                     (maxGammaError < tolerance) && 
                     (maxBetaError < tolerance);
        
        std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "Tolerance: " << tolerance << std::endl;
        }
    }
    
    // Cleanup
    delete inputTensor;
    delete outputGradTensor;
    delete gammaTensor;
    delete betaTensor;
    delete inputGradTensor;
    delete gammaGradTensor;
    delete betaGradTensor;
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}