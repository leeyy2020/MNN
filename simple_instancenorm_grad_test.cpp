//
//  simple_instancenorm_grad_test.cpp
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

// Simple test implementation of InstanceNorm gradient computation
// This tests the mathematical correctness of our algorithm

class SimpleInstanceNormGradTest {
private:
    // Forward pass implementation
    void instanceNormForward(const std::vector<float>& input,
                           const std::vector<float>& gamma,
                           const std::vector<float>& beta,
                           std::vector<float>& output,
                           std::vector<float>& mean,
                           std::vector<float>& variance,
                           std::vector<float>& normalized,
                           int batch, int channel, int spatial, float epsilon) {
        
        output.resize(input.size());
        mean.resize(batch * channel);
        variance.resize(batch * channel);
        normalized.resize(input.size());
        
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int idx = b * channel + c;
                int base = (b * channel + c) * spatial;
                
                // Compute mean
                float sum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    sum += input[base + s];
                }
                mean[idx] = sum / spatial;
                
                // Compute variance
                float varSum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    float diff = input[base + s] - mean[idx];
                    varSum += diff * diff;
                }
                variance[idx] = varSum / spatial;
                
                // Normalize and apply affine transform
                float stdInv = 1.0f / std::sqrt(variance[idx] + epsilon);
                for (int s = 0; s < spatial; ++s) {
                    int pos = base + s;
                    normalized[pos] = (input[pos] - mean[idx]) * stdInv;
                    output[pos] = normalized[pos] * gamma[c] + beta[c];
                }
            }
        }
    }
    
    // Backward pass implementation
    void instanceNormBackward(const std::vector<float>& input,
                            const std::vector<float>& outputGrad,
                            const std::vector<float>& gamma,
                            const std::vector<float>& mean,
                            const std::vector<float>& variance,
                            const std::vector<float>& normalized,
                            std::vector<float>& inputGrad,
                            std::vector<float>& gammaGrad,
                            std::vector<float>& betaGrad,
                            int batch, int channel, int spatial, float epsilon) {
        
        inputGrad.resize(input.size());
        gammaGrad.resize(channel, 0.0f);
        betaGrad.resize(channel, 0.0f);
        
        // Compute parameter gradients
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int base = (b * channel + c) * spatial;
                for (int s = 0; s < spatial; ++s) {
                    int pos = base + s;
                    gammaGrad[c] += outputGrad[pos] * normalized[pos];
                    betaGrad[c] += outputGrad[pos];
                }
            }
        }
        
        // Compute input gradients
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int idx = b * channel + c;
                int base = (b * channel + c) * spatial;
                
                float stdInv = 1.0f / std::sqrt(variance[idx] + epsilon);
                
                // Compute intermediate values
                float dySum = 0.0f;
                float dyNormSum = 0.0f;
                
                for (int s = 0; s < spatial; ++s) {
                    int pos = base + s;
                    dySum += outputGrad[pos];
                    dyNormSum += outputGrad[pos] * normalized[pos];
                }
                
                dySum /= spatial;
                dyNormSum /= spatial;
                
                // Compute input gradient
                for (int s = 0; s < spatial; ++s) {
                    int pos = base + s;
                    float dxNorm = outputGrad[pos] - dySum - normalized[pos] * dyNormSum;
                    inputGrad[pos] = gamma[c] * stdInv * dxNorm;
                }
            }
        }
    }
    
    // Numerical gradient computation for verification
    float computeNumericalGradient(const std::vector<float>& input,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& beta,
                                 const std::vector<float>& outputGrad,
                                 int paramIndex, int paramType, // 0=input, 1=gamma, 2=beta
                                 int batch, int channel, int spatial, float epsilon, float h) {
        
        auto computeLoss = [&](const std::vector<float>& inp,
                              const std::vector<float>& g,
                              const std::vector<float>& b) -> float {
            std::vector<float> output, mean, variance, normalized;
            instanceNormForward(inp, g, b, output, mean, variance, normalized, batch, channel, spatial, epsilon);
            
            float loss = 0.0f;
            for (int i = 0; i < output.size(); ++i) {
                loss += output[i] * outputGrad[i];
            }
            return loss;
        };
        
        float gradNumerical = 0.0f;
        
        if (paramType == 0) { // input gradient
            std::vector<float> inputPlus = input;
            std::vector<float> inputMinus = input;
            inputPlus[paramIndex] += h;
            inputMinus[paramIndex] -= h;
            
            float lossPlus = computeLoss(inputPlus, gamma, beta);
            float lossMinus = computeLoss(inputMinus, gamma, beta);
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        } else if (paramType == 1) { // gamma gradient
            std::vector<float> gammaPlus = gamma;
            std::vector<float> gammaMinus = gamma;
            gammaPlus[paramIndex] += h;
            gammaMinus[paramIndex] -= h;
            
            float lossPlus = computeLoss(input, gammaPlus, beta);
            float lossMinus = computeLoss(input, gammaMinus, beta);
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        } else if (paramType == 2) { // beta gradient
            std::vector<float> betaPlus = beta;
            std::vector<float> betaMinus = beta;
            betaPlus[paramIndex] += h;
            betaMinus[paramIndex] -= h;
            
            float lossPlus = computeLoss(input, gamma, betaPlus);
            float lossMinus = computeLoss(input, gamma, betaMinus);
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        }
        
        return gradNumerical;
    }

public:
    bool runTest() {
        std::cout << "=== Simple InstanceNorm Gradient Test ===" << std::endl;
        
        // Test configuration
        const int batch = 2;
        const int channel = 2;
        const int height = 3;
        const int width = 3;
        const int spatial = height * width;
        const float epsilon = 1e-5f;
        const float h = 1e-4f; // numerical gradient step size
        
        std::cout << "Configuration: batch=" << batch << ", channel=" << channel 
                  << ", spatial=" << spatial << ", epsilon=" << epsilon << std::endl;
        
        // Generate test data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> inputDis(-2.0f, 2.0f);
        std::uniform_real_distribution<float> gradDis(-1.0f, 1.0f);
        std::uniform_real_distribution<float> paramDis(0.5f, 1.5f);
        
        std::vector<float> input(batch * channel * spatial);
        std::vector<float> outputGrad(batch * channel * spatial);
        std::vector<float> gamma(channel);
        std::vector<float> beta(channel);
        
        for (int i = 0; i < input.size(); ++i) {
            input[i] = inputDis(gen);
            outputGrad[i] = gradDis(gen);
        }
        
        for (int i = 0; i < channel; ++i) {
            gamma[i] = paramDis(gen);
            beta[i] = paramDis(gen) - 1.0f;
        }
        
        // Forward pass
        std::vector<float> output, mean, variance, normalized;
        instanceNormForward(input, gamma, beta, output, mean, variance, normalized, 
                          batch, channel, spatial, epsilon);
        
        // Backward pass
        std::vector<float> inputGrad, gammaGrad, betaGrad;
        instanceNormBackward(input, outputGrad, gamma, mean, variance, normalized,
                           inputGrad, gammaGrad, betaGrad, batch, channel, spatial, epsilon);
        
        // Verify gradients numerically
        bool allPassed = true;
        const float tolerance = 5e-3f; // Relaxed tolerance for numerical gradient verification
        
        std::cout << "\n=== Gradient Verification ===" << std::endl;
        
        // Check a few input gradients
        int numInputChecks = std::min(8, (int)input.size());
        float maxInputError = 0.0f;
        for (int i = 0; i < numInputChecks; ++i) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, i, 0, 
                                                     batch, channel, spatial, epsilon, h);
            float analytical = inputGrad[i];
            float error = std::abs(numerical - analytical);
            maxInputError = std::max(maxInputError, error);
            
            if (i < 4) { // Print first few
                std::cout << "Input grad [" << i << "]: analytical=" << std::fixed << std::setprecision(6) 
                          << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
            }
        }
        
        // Check gamma gradients
        float maxGammaError = 0.0f;
        for (int c = 0; c < channel; ++c) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, c, 1, 
                                                     batch, channel, spatial, epsilon, h);
            float analytical = gammaGrad[c];
            float error = std::abs(numerical - analytical);
            maxGammaError = std::max(maxGammaError, error);
            
            std::cout << "Gamma grad [" << c << "]: analytical=" << std::fixed << std::setprecision(6) 
                      << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
        }
        
        // Check beta gradients
        float maxBetaError = 0.0f;
        for (int c = 0; c < channel; ++c) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, c, 2, 
                                                     batch, channel, spatial, epsilon, h);
            float analytical = betaGrad[c];
            float error = std::abs(numerical - analytical);
            maxBetaError = std::max(maxBetaError, error);
            
            std::cout << "Beta grad [" << c << "]: analytical=" << std::fixed << std::setprecision(6) 
                      << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
        }
        
        std::cout << "\n=== Error Summary ===" << std::endl;
        std::cout << "Max input gradient error: " << std::scientific << maxInputError << std::endl;
        std::cout << "Max gamma gradient error: " << std::scientific << maxGammaError << std::endl;
        std::cout << "Max beta gradient error: " << std::scientific << maxBetaError << std::endl;
        std::cout << "Tolerance: " << std::scientific << tolerance << std::endl;
        
        bool inputPassed = maxInputError < tolerance;
        bool gammaPassed = maxGammaError < tolerance;
        bool betaPassed = maxBetaError < tolerance;
        
        allPassed = inputPassed && gammaPassed && betaPassed;
        
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Input gradients: " << (inputPassed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Gamma gradients: " << (gammaPassed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Beta gradients: " << (betaPassed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Overall: " << (allPassed ? "PASSED" : "FAILED") << std::endl;
        
        return allPassed;
    }
};

int main() {
    SimpleInstanceNormGradTest test;
    bool result = test.runTest();
    return result ? 0 : 1;
}