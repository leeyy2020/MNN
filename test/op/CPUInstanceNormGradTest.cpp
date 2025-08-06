//
//  CPUInstanceNormGradTest.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include <vector>
#include <random>
#include <cmath>
#include "MNNTestSuite.h"
#include "TestUtils.h"
#include "source/backend/cpu/CPUInstanceNormGrad.hpp"
#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"

using namespace MNN;

class CPUInstanceNormGradTest : public MNNTestCase {
private:
    // Helper function to create test tensors
    Tensor* createTensor(const std::vector<int>& shape, Tensor::DimensionType dimType = Tensor::CAFFE) {
        return Tensor::create(shape, halide_type_of<float>(), nullptr, dimType);
    }
    
    // Helper function to fill tensor with test data
    void fillTensor(Tensor* tensor, float value) {
        float* data = tensor->host<float>();
        int size = tensor->elementSize();
        for (int i = 0; i < size; ++i) {
            data[i] = value;
        }
    }
    
    // Helper function to fill tensor with random data
    void fillRandomTensor(Tensor* tensor, int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        float* data = tensor->host<float>();
        int size = tensor->elementSize();
        for (int i = 0; i < size; ++i) {
            data[i] = dis(gen);
        }
    }
    
    // Simple mock backend for testing
    class MockCPUBackend : public Backend {
    public:
        MockCPUBackend() : Backend(MNN_FORWARD_CPU) {}
        virtual ~MockCPUBackend() = default;
        
        virtual MemObj* onAcquire(const Tensor*, StorageType) override { return nullptr; }
        virtual bool onClearBuffer() override { return true; }
        virtual void onCopyBuffer(const Tensor*, const Tensor*) override {}
        virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>&, const std::vector<Tensor*>&, const MNN::Op*) override {
            return std::make_pair(0.0f, false);
        }
        virtual Execution* onCreate(const std::vector<Tensor*>&, const std::vector<Tensor*>&, const MNN::Op*) override {
            return nullptr;
        }
        virtual void onExecuteBegin() const override {}
        virtual void onExecuteEnd() const override {}
        virtual bool onReleaseBuffer(const Tensor*, StorageType) override { return true; }
        virtual void onResizeBegin() override {}
        virtual void onResizeEnd() override {}
        
        // Mock buffer allocator
        class MockBufferAllocator : public BufferAllocator {
        public:
            virtual MemChunk alloc(size_t size, bool = false) override {
                return MemChunk(malloc(size));
            }
            virtual void free(const MemChunk& chunk) override {
                if (chunk.ptr()) ::free(chunk.ptr());
            }
            virtual void release(bool = true) override {}
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

public:
    virtual ~CPUInstanceNormGradTest() = default;
    
    virtual bool run(int precision) {
        return testBasicFunctionality() && testGradientComputation();
    }
    
private:
    bool testBasicFunctionality() {
        // Test basic functionality - can the operation run without crashing?
        const int batch = 1, channel = 2, height = 3, width = 3;
        const float epsilon = 1e-5f;
        
        MockCPUBackend backend;
        CPUInstanceNormGrad gradOp(&backend, epsilon);
        
        // Create tensors
        auto inputTensor = createTensor({batch, channel, height, width});
        auto outputGradTensor = createTensor({batch, channel, height, width});
        auto gammaTensor = createTensor({channel});
        auto betaTensor = createTensor({channel});
        
        auto inputGradTensor = createTensor({batch, channel, height, width});
        auto gammaGradTensor = createTensor({channel});
        auto betaGradTensor = createTensor({channel});
        
        // Fill with simple test data
        fillTensor(inputTensor, 1.0f);
        fillTensor(outputGradTensor, 1.0f);
        fillTensor(gammaTensor, 1.0f);
        fillTensor(betaTensor, 0.0f);
        
        std::vector<Tensor*> inputs = {inputTensor, outputGradTensor, gammaTensor, betaTensor};
        std::vector<Tensor*> outputs = {inputGradTensor, gammaGradTensor, betaGradTensor};
        
        // Test resize
        auto resizeError = gradOp.onResize(inputs, outputs);
        if (resizeError != NO_ERROR) {
            MNN_ERROR("Resize failed with error: %d\n", resizeError);
            return false;
        }
        
        // Test execute
        auto executeError = gradOp.onExecute(inputs, outputs);
        if (executeError != NO_ERROR) {
            MNN_ERROR("Execute failed with error: %d\n", executeError);
            return false;
        }
        
        // Basic sanity check - gradients should not be NaN or infinite
        const float* inputGrad = inputGradTensor->host<float>();
        const float* gammaGrad = gammaGradTensor->host<float>();
        const float* betaGrad = betaGradTensor->host<float>();
        
        for (int i = 0; i < inputGradTensor->elementSize(); ++i) {
            if (!std::isfinite(inputGrad[i])) {
                MNN_ERROR("Input gradient contains non-finite values at index %d: %f\n", i, inputGrad[i]);
                return false;
            }
        }
        
        for (int i = 0; i < gammaGradTensor->elementSize(); ++i) {
            if (!std::isfinite(gammaGrad[i])) {
                MNN_ERROR("Gamma gradient contains non-finite values at index %d: %f\n", i, gammaGrad[i]);
                return false;
            }
        }
        
        for (int i = 0; i < betaGradTensor->elementSize(); ++i) {
            if (!std::isfinite(betaGrad[i])) {
                MNN_ERROR("Beta gradient contains non-finite values at index %d: %f\n", i, betaGrad[i]);
                return false;
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
        
        return true;
    }
    
    bool testGradientComputation() {
        // Test specific gradient computation properties
        const int batch = 1, channel = 1, height = 2, width = 2;
        const float epsilon = 1e-5f;
        
        MockCPUBackend backend;
        CPUInstanceNormGrad gradOp(&backend, epsilon);
        
        // Create tensors
        auto inputTensor = createTensor({batch, channel, height, width});
        auto outputGradTensor = createTensor({batch, channel, height, width});
        auto gammaTensor = createTensor({channel});
        auto betaTensor = createTensor({channel});
        
        auto inputGradTensor = createTensor({batch, channel, height, width});
        auto gammaGradTensor = createTensor({channel});
        auto betaGradTensor = createTensor({channel});
        
        // Set up a simple test case
        float* inputData = inputTensor->host<float>();
        inputData[0] = 1.0f; inputData[1] = 2.0f;
        inputData[2] = 3.0f; inputData[3] = 4.0f;
        
        float* outputGradData = outputGradTensor->host<float>();
        outputGradData[0] = 0.1f; outputGradData[1] = 0.2f;
        outputGradData[2] = 0.3f; outputGradData[3] = 0.4f;
        
        fillTensor(gammaTensor, 1.0f);
        fillTensor(betaTensor, 0.0f);
        
        std::vector<Tensor*> inputs = {inputTensor, outputGradTensor, gammaTensor, betaTensor};
        std::vector<Tensor*> outputs = {inputGradTensor, gammaGradTensor, betaGradTensor};
        
        // Run the operation
        auto resizeError = gradOp.onResize(inputs, outputs);
        if (resizeError != NO_ERROR) return false;
        
        auto executeError = gradOp.onExecute(inputs, outputs);
        if (executeError != NO_ERROR) return false;
        
        // Check that beta gradient equals sum of output gradients (since gamma = 1)
        const float* betaGrad = betaGradTensor->host<float>();
        float expectedBetaGrad = 0.1f + 0.2f + 0.3f + 0.4f; // Sum of output gradients
        
        if (std::abs(betaGrad[0] - expectedBetaGrad) > 1e-5f) {
            MNN_ERROR("Beta gradient mismatch: expected %f, got %f\n", expectedBetaGrad, betaGrad[0]);
            return false;
        }
        
        // Check that input gradients are finite and reasonable
        const float* inputGrad = inputGradTensor->host<float>();
        for (int i = 0; i < 4; ++i) {
            if (!std::isfinite(inputGrad[i])) {
                MNN_ERROR("Input gradient %d is not finite: %f\n", i, inputGrad[i]);
                return false;
            }
            if (std::abs(inputGrad[i]) > 100.0f) {  // Reasonable bounds check
                MNN_ERROR("Input gradient %d seems too large: %f\n", i, inputGrad[i]);
                return false;
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
        
        return true;
    }
};

MNNTestSuiteRegister(CPUInstanceNormGradTest, "op/cpu_instance_norm_grad");