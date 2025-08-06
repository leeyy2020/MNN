//
//  test_instancenorm_grad_algorithm.cpp
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

// Direct implementation of the CPUInstanceNormGrad algorithm for testing
class InstanceNormGradAlgorithm {
public:
    void computeGradients(const float* input, const float* outputGrad, 
                         const float* gamma, const float* beta,
                         float* inputGrad, float* gammaGrad, float* betaGrad,
                         int batch, int channel, int spatial, float epsilon) {
        
        // Allocate temporary buffers
        std::vector<float> mean(batch * channel);
        std::vector<float> variance(batch * channel);
        std::vector<float> normalized(batch * channel * spatial);
        
        // Step 1: Compute mean and variance for each instance-channel
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int idx = b * channel + c;
                const float* inputSlice = input + (b * channel + c) * spatial;
                
                // Compute mean
                float sum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    sum += inputSlice[s];
                }
                mean[idx] = sum / spatial;
                
                // Compute variance
                float varSum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    float diff = inputSlice[s] - mean[idx];
                    varSum += diff * diff;
                }
                variance[idx] = varSum / spatial;
            }
        }
        
        // Step 2: Compute normalized values
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int idx = b * channel + c;
                const float* inputSlice = input + (b * channel + c) * spatial;
                float* normalizedSlice = &normalized[(b * channel + c) * spatial];
                
                float stdInv = 1.0f / std::sqrt(variance[idx] + epsilon);
                for (int s = 0; s < spatial; ++s) {
                    normalizedSlice[s] = (inputSlice[s] - mean[idx]) * stdInv;
                }
            }
        }
        
        // Step 3: Compute input gradients
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                int idx = b * channel + c;
                const float* outputGradSlice = outputGrad + (b * channel + c) * spatial;
                const float* normalizedSlice = &normalized[(b * channel + c) * spatial];
                float* inputGradSlice = inputGrad + (b * channel + c) * spatial;
                
                float stdInv = 1.0f / std::sqrt(variance[idx] + epsilon);
                
                // Compute intermediate values
                float dySum = 0.0f;
                float dyNormSum = 0.0f;
                
                for (int s = 0; s < spatial; ++s) {
                    dySum += outputGradSlice[s];
                    dyNormSum += outputGradSlice[s] * normalizedSlice[s];
                }
                
                dySum /= spatial;
                dyNormSum /= spatial;
                
                // Compute input gradient
                for (int s = 0; s < spatial; ++s) {
                    float dxNorm = outputGradSlice[s] - dySum - normalizedSlice[s] * dyNormSum;
                    if (gamma != nullptr) {
                        dxNorm *= gamma[c];
                    }
                    inputGradSlice[s] = dxNorm * stdInv;
                }
            }
        }
        
        // Step 4: Compute gamma gradients (if needed)
        if (gammaGrad != nullptr) {
            std::fill(gammaGrad, gammaGrad + channel, 0.0f);
            
            for (int b = 0; b < batch; ++b) {
                for (int c = 0; c < channel; ++c) {
                    const float* outputGradSlice = outputGrad + (b * channel + c) * spatial;
                    const float* normalizedSlice = &normalized[(b * channel + c) * spatial];
                    
                    float sum = 0.0f;
                    for (int s = 0; s < spatial; ++s) {
                        sum += outputGradSlice[s] * normalizedSlice[s];
                    }
                    gammaGrad[c] += sum;
                }
            }
        }
        
        // Step 5: Compute beta gradients (if needed)
        if (betaGrad != nullptr) {
            std::fill(betaGrad, betaGrad + channel, 0.0f);
            
            for (int b = 0; b < batch; ++b) {
                for (int c = 0; c < channel; ++c) {
                    const float* outputGradSlice = outputGrad + (b * channel + c) * spatial;
                    
                    float sum = 0.0f;
                    for (int s = 0; s < spatial; ++s) {
                        sum += outputGradSlice[s];
                    }
                    betaGrad[c] += sum;
                }
            }
        }
    }
};

class InstanceNormGradTester {
private:
    void fillRandomData(std::vector<float>& data, std::mt19937& gen, float min_val, float max_val) {
        std::uniform_real_distribution<float> dis(min_val, max_val);
        for (auto& val : data) {
            val = dis(gen);
        }
    }
    
    void printVector(const std::vector<float>& data, const std::string& name, int maxElements = 8) {
        std::cout << name << ": ";
        int size = std::min(maxElements, (int)data.size());
        
        for (int i = 0; i < size; ++i) {
            std::cout << std::fixed << std::setprecision(4) << data[i];
            if (i < size - 1) std::cout << ", ";
        }
        if (size < data.size()) {
            std::cout << " ... (+" << (data.size() - size) << " more)";
        }
        std::cout << std::endl;
    }
    
    // Numerical gradient computation for verification
    float computeNumericalGradient(const std::vector<float>& input,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& beta,
                                 const std::vector<float>& outputGrad,
                                 int paramIndex, int paramType,
                                 int batch, int channel, int spatial, float epsilon, float h) {
        
        auto forwardPass = [&](const std::vector<float>& inp,
                              const std::vector<float>& g,
                              const std::vector<float>& b) -> std::vector<float> {
            std::vector<float> output(inp.size());
            
            for (int n = 0; n < batch; ++n) {
                for (int c = 0; c < channel; ++c) {
                    // Compute mean and variance
                    float sum = 0.0f;
                    float sumSq = 0.0f;
                    int idx_base = (n * channel + c) * spatial;
                    
                    for (int s = 0; s < spatial; ++s) {
                        float val = inp[idx_base + s];
                        sum += val;
                        sumSq += val * val;
                    }
                    
                    float mean = sum / spatial;
                    float variance = sumSq / spatial - mean * mean;
                    float stdInv = 1.0f / std::sqrt(variance + epsilon);
                    
                    // Apply normalization
                    for (int s = 0; s < spatial; ++s) {
                        int idx = idx_base + s;
                        float normalized = (inp[idx] - mean) * stdInv;
                        output[idx] = normalized * g[c] + b[c];
                    }
                }
            }
            return output;
        };
        
        float gradNumerical = 0.0f;
        
        if (paramType == 0) { // input gradient
            std::vector<float> inputPlus = input;
            std::vector<float> inputMinus = input;
            inputPlus[paramIndex] += h;
            inputMinus[paramIndex] -= h;
            
            auto outputPlus = forwardPass(inputPlus, gamma, beta);
            auto outputMinus = forwardPass(inputMinus, gamma, beta);
            
            float lossPlus = 0.0f, lossMinus = 0.0f;
            for (int i = 0; i < outputGrad.size(); ++i) {
                lossPlus += outputPlus[i] * outputGrad[i];
                lossMinus += outputMinus[i] * outputGrad[i];
            }
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        } else if (paramType == 1) { // gamma gradient
            std::vector<float> gammaPlus = gamma;
            std::vector<float> gammaMinus = gamma;
            gammaPlus[paramIndex] += h;
            gammaMinus[paramIndex] -= h;
            
            auto outputPlus = forwardPass(input, gammaPlus, beta);
            auto outputMinus = forwardPass(input, gammaMinus, beta);
            
            float lossPlus = 0.0f, lossMinus = 0.0f;
            for (int i = 0; i < outputGrad.size(); ++i) {
                lossPlus += outputPlus[i] * outputGrad[i];
                lossMinus += outputMinus[i] * outputGrad[i];
            }
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        } else if (paramType == 2) { // beta gradient
            std::vector<float> betaPlus = beta;
            std::vector<float> betaMinus = beta;
            betaPlus[paramIndex] += h;
            betaMinus[paramIndex] -= h;
            
            auto outputPlus = forwardPass(input, gamma, betaPlus);
            auto outputMinus = forwardPass(input, gamma, betaMinus);
            
            float lossPlus = 0.0f, lossMinus = 0.0f;
            for (int i = 0; i < outputGrad.size(); ++i) {
                lossPlus += outputPlus[i] * outputGrad[i];
                lossMinus += outputMinus[i] * outputGrad[i];
            }
            gradNumerical = (lossPlus - lossMinus) / (2.0f * h);
        }
        
        return gradNumerical;
    }

public:
    bool runTest() {
        std::cout << "=== InstanceNorm Gradient Algorithm Test ===" << std::endl;
        
        // Test configuration
        const int batch = 2;
        const int channel = 2;
        const int height = 3;
        const int width = 3;
        const int spatial = height * width;
        const float epsilon = 1e-5f;
        
        std::cout << "Configuration: batch=" << batch << ", channel=" << channel 
                  << ", spatial=" << spatial << ", epsilon=" << epsilon << std::endl << std::endl;
        
        // Create test data
        std::mt19937 gen(42);
        
        std::vector<float> input(batch * channel * spatial);
        std::vector<float> outputGrad(batch * channel * spatial);
        std::vector<float> gamma(channel);
        std::vector<float> beta(channel);
        
        std::vector<float> inputGrad(batch * channel * spatial);
        std::vector<float> gammaGrad(channel);
        std::vector<float> betaGrad(channel);
        
        // Fill with random data
        fillRandomData(input, gen, -2.0f, 2.0f);
        fillRandomData(outputGrad, gen, -1.0f, 1.0f);
        fillRandomData(gamma, gen, 0.5f, 1.5f);
        fillRandomData(beta, gen, -0.5f, 0.5f);
        
        // Print input data
        std::cout << "=== Input Data ===" << std::endl;
        printVector(input, "Input");
        printVector(outputGrad, "Output Gradient");
        printVector(gamma, "Gamma", channel);
        printVector(beta, "Beta", channel);
        std::cout << std::endl;
        
        // Run gradient computation
        InstanceNormGradAlgorithm algorithm;
        algorithm.computeGradients(input.data(), outputGrad.data(), 
                                 gamma.data(), beta.data(),
                                 inputGrad.data(), gammaGrad.data(), betaGrad.data(),
                                 batch, channel, spatial, epsilon);
        
        // Print results
        std::cout << "=== Results ===" << std::endl;
        printVector(inputGrad, "Input Gradient");
        printVector(gammaGrad, "Gamma Gradient", channel);
        printVector(betaGrad, "Beta Gradient", channel);
        std::cout << std::endl;
        
        // Verification
        std::cout << "=== Verification ===" << std::endl;
        
        bool allFinite = true;
        for (float val : inputGrad) {
            if (!std::isfinite(val)) {
                allFinite = false;
                break;
            }
        }
        for (float val : gammaGrad) {
            if (!std::isfinite(val)) {
                allFinite = false;
                break;
            }
        }
        for (float val : betaGrad) {
            if (!std::isfinite(val)) {
                allFinite = false;
                break;
            }
        }
        
        if (allFinite) {
            std::cout << "✅ All gradients are finite" << std::endl;
        } else {
            std::cout << "❌ Some gradients are not finite" << std::endl;
            return false;
        }
        
        // Numerical gradient verification (sample a few)
        const float h = 1e-4f;
        const float tolerance = 5e-3f;
        
        std::cout << "\nNumerical gradient verification:" << std::endl;
        
        // Check a few input gradients
        bool gradCheckPassed = true;
        for (int i = 0; i < std::min(4, (int)input.size()); ++i) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, i, 0,
                                                     batch, channel, spatial, epsilon, h);
            float analytical = inputGrad[i];
            float error = std::abs(numerical - analytical);
            
            std::cout << "Input grad [" << i << "]: analytical=" << std::fixed << std::setprecision(6)
                      << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
            
            if (error > tolerance) gradCheckPassed = false;
        }
        
        // Check gamma gradients
        for (int c = 0; c < channel; ++c) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, c, 1,
                                                     batch, channel, spatial, epsilon, h);
            float analytical = gammaGrad[c];
            float error = std::abs(numerical - analytical);
            
            std::cout << "Gamma grad [" << c << "]: analytical=" << std::fixed << std::setprecision(6)
                      << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
            
            if (error > tolerance) gradCheckPassed = false;
        }
        
        // Check beta gradients
        for (int c = 0; c < channel; ++c) {
            float numerical = computeNumericalGradient(input, gamma, beta, outputGrad, c, 2,
                                                     batch, channel, spatial, epsilon, h);
            float analytical = betaGrad[c];
            float error = std::abs(numerical - analytical);
            
            std::cout << "Beta grad [" << c << "]: analytical=" << std::fixed << std::setprecision(6)
                      << analytical << ", numerical=" << numerical << ", error=" << error << std::endl;
            
            if (error > tolerance) gradCheckPassed = false;
        }
        
        if (gradCheckPassed) {
            std::cout << "✅ Numerical gradient verification passed" << std::endl;
        } else {
            std::cout << "❌ Numerical gradient verification failed" << std::endl;
        }
        
        std::cout << "\n=== Final Result ===" << std::endl;
        bool success = allFinite && gradCheckPassed;
        std::cout << "Test " << (success ? "PASSED ✅" : "FAILED ❌") << std::endl;
        
        return success;
    }
};

int main() {
    InstanceNormGradTester tester;
    bool result = tester.runTest();
    return result ? 0 : 1;
}