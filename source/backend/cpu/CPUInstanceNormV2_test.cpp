//
//  CPUInstanceNormV2_test.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include "CPUInstanceNormV2.hpp"
#include "CPUBackend.hpp"
#include "core/TensorUtils.hpp"
#include <iostream>
#include <vector>
#include <random>

namespace MNN {

// Simple test function for CPUInstanceNormV2
void testInstanceNormV2() {
    std::cout << "Testing CPUInstanceNormV2..." << std::endl;
    
    // Create a simple CPU backend for testing
    BackendConfig config;
    config.precision = BackendConfig::Precision_Normal;
    config.power = BackendConfig::Power_Normal;
    
    auto backend = std::shared_ptr<Backend>(new CPUBackend(config, nullptr));
    
    // Test parameters
    const int batch = 2;
    const int channel = 3;
    const int height = 4;
    const int width = 4;
    const int spatial = height * width;
    const float epsilon = 1e-5f;
    
    // Create input tensor
    auto input = Tensor::create<float>({batch, channel, height, width}, nullptr, Tensor::CAFFE);
    auto output = Tensor::create<float>({batch, channel, height, width}, nullptr, Tensor::CAFFE);
    auto gamma = Tensor::create<float>({channel}, nullptr, Tensor::CAFFE);
    auto beta = Tensor::create<float>({channel}, nullptr, Tensor::CAFFE);
    
    // Initialize input data with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    auto inputPtr = input->host<float>();
    auto gammaPtr = gamma->host<float>();
    auto betaPtr = beta->host<float>();
    
    // Fill input with random data
    for (int i = 0; i < batch * channel * spatial; ++i) {
        inputPtr[i] = dis(gen);
    }
    
    // Initialize gamma (scale) and beta (bias)
    for (int c = 0; c < channel; ++c) {
        gammaPtr[c] = 1.0f;  // Scale factor
        betaPtr[c] = 0.0f;   // Bias
    }
    
    // Create Instance Norm execution
    auto instanceNorm = std::unique_ptr<CPUInstanceNormV2>(new CPUInstanceNormV2(backend.get(), epsilon, true));
    
    // Setup inputs and outputs
    std::vector<Tensor*> inputs = {input.get(), gamma.get(), beta.get()};
    std::vector<Tensor*> outputs = {output.get()};
    
    // Resize and execute
    auto resizeResult = instanceNorm->onResize(inputs, outputs);
    if (resizeResult != NO_ERROR) {
        std::cout << "Resize failed!" << std::endl;
        return;
    }
    
    auto executeResult = instanceNorm->onExecute(inputs, outputs);
    if (executeResult != NO_ERROR) {
        std::cout << "Execute failed!" << std::endl;
        return;
    }
    
    // Verify results (simple sanity check)
    auto outputPtr = output->host<float>();
    bool hasValidOutput = false;
    
    for (int i = 0; i < batch * channel * spatial; ++i) {
        if (!std::isnan(outputPtr[i]) && std::isfinite(outputPtr[i])) {
            hasValidOutput = true;
            break;
        }
    }
    
    if (hasValidOutput) {
        std::cout << "Test PASSED: Output contains valid values" << std::endl;
        
        // Print some sample values
        std::cout << "Sample input values: ";
        for (int i = 0; i < std::min(5, batch * channel * spatial); ++i) {
            std::cout << inputPtr[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Sample output values: ";
        for (int i = 0; i < std::min(5, batch * channel * spatial); ++i) {
            std::cout << outputPtr[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Test FAILED: Output contains invalid values" << std::endl;
    }
}

} // namespace MNN

// Main function for testing
int main() {
    MNN::testInstanceNormV2();
    return 0;
}