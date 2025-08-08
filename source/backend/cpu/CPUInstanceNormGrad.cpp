//
//  CPUInstanceNormGrad.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include "CPUInstanceNormGrad.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"
#include <cmath>
#include <cstring>

namespace MNN {

CPUInstanceNormGrad::CPUInstanceNormGrad(Backend* backend, float epsilon) : Execution(backend) {
    mEpsilon = epsilon;
}

ErrorCode CPUInstanceNormGrad::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Input layout:
    // inputs[0]: input tensor (forward input)
    // inputs[1]: output gradient (dy)
    // inputs[2]: gamma (optional)
    // inputs[3]: beta (optional)
    
    // Output layout:
    // outputs[0]: input gradient (dx)
    // outputs[1]: gamma gradient (optional)
    // outputs[2]: beta gradient (optional)
    
    MNN_ASSERT(inputs.size() >= 2);
    MNN_ASSERT(outputs.size() >= 1);
    
    const Tensor* input = inputs[0];
    
    // Assume input format is NCHW
    mBatch = input->batch();
    mChannel = input->channel();
    mSpatial = input->height() * input->width();
    
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    
    // Allocate temporary buffers
    auto meanAlloc = bufferAlloc->alloc(mBatch * mChannel * sizeof(float));
    auto varAlloc = bufferAlloc->alloc(mBatch * mChannel * sizeof(float));
    auto normalizedAlloc = bufferAlloc->alloc(input->elementSize() * sizeof(float));
    
    if (meanAlloc.invalid() || varAlloc.invalid() || normalizedAlloc.invalid()) {
        return OUT_OF_MEMORY;
    }
    
    mTempMean = meanAlloc;
    mTempVar = varAlloc;
    mTempNormalized = normalizedAlloc;
    
    // Free allocations (they will be kept until execution)
    bufferAlloc->free(meanAlloc);
    bufferAlloc->free(varAlloc);
    bufferAlloc->free(normalizedAlloc);
    
    return NO_ERROR;
}

ErrorCode CPUInstanceNormGrad::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const float* inputPtr = inputs[0]->host<float>();
    const float* outputGradPtr = inputs[1]->host<float>();
    
    const float* gammaPtr = nullptr;
    const float* betaPtr = nullptr;
    
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        gammaPtr = inputs[2]->host<float>();
    }
    if (inputs.size() > 3 && inputs[3] != nullptr) {
        betaPtr = inputs[3]->host<float>();
    }
    
    float* inputGradPtr = outputs[0]->host<float>();
    float* gammaGradPtr = nullptr;
    float* betaGradPtr = nullptr;
    
    if (outputs.size() > 1 && outputs[1] != nullptr) {
        gammaGradPtr = outputs[1]->host<float>();
    }
    if (outputs.size() > 2 && outputs[2] != nullptr) {
        betaGradPtr = outputs[2]->host<float>();
    }
    
    computeGradients(inputPtr, outputGradPtr, gammaPtr, betaPtr,
                    inputGradPtr, gammaGradPtr, betaGradPtr,
                    mBatch, mChannel, mSpatial);
    
    return NO_ERROR;
}

void CPUInstanceNormGrad::computeGradients(const float* input, const float* outputGrad, 
                                          const float* gamma, const float* beta,
                                          float* inputGrad, float* gammaGrad, float* betaGrad,
                                          int batch, int channel, int spatial) {
    
    float* meanPtr = reinterpret_cast<float*>(mTempMean.ptr());
    float* varPtr = reinterpret_cast<float*>(mTempVar.ptr());
    float* normalizedPtr = reinterpret_cast<float*>(mTempNormalized.ptr());
    
    // Step 1: Compute mean and variance for each instance-channel
    MNN_CONCURRENCY_BEGIN(b, batch) {
        for (int c = 0; c < channel; ++c) {
            int idx = b * channel + c;
            const float* inputSlice = input + (b * channel + c) * spatial;
            
            // Compute mean
            float sum = 0.0f;
            for (int s = 0; s < spatial; ++s) {
                sum += inputSlice[s];
            }
            meanPtr[idx] = sum / spatial;
            
            // Compute variance
            float varSum = 0.0f;
            for (int s = 0; s < spatial; ++s) {
                float diff = inputSlice[s] - meanPtr[idx];
                varSum += diff * diff;
            }
            varPtr[idx] = varSum / spatial;
        }
    }
    MNN_CONCURRENCY_END();
    
    // Step 2: Compute normalized values
    MNN_CONCURRENCY_BEGIN(b, batch) {
        for (int c = 0; c < channel; ++c) {
            int idx = b * channel + c;
            const float* inputSlice = input + (b * channel + c) * spatial;
            float* normalizedSlice = normalizedPtr + (b * channel + c) * spatial;
            
            float stdInv = 1.0f / sqrtf(varPtr[idx] + mEpsilon);
            for (int s = 0; s < spatial; ++s) {
                normalizedSlice[s] = (inputSlice[s] - meanPtr[idx]) * stdInv;
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    // Step 3: Compute input gradients
    MNN_CONCURRENCY_BEGIN(b, batch) {
        for (int c = 0; c < channel; ++c) {
            int idx = b * channel + c;
            const float* outputGradSlice = outputGrad + (b * channel + c) * spatial;
            const float* normalizedSlice = normalizedPtr + (b * channel + c) * spatial;
            float* inputGradSlice = inputGrad + (b * channel + c) * spatial;
            
            float stdInv = 1.0f / sqrtf(varPtr[idx] + mEpsilon);
            
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
    MNN_CONCURRENCY_END();
    
    // Step 4: Compute gamma gradients (if needed)
    if (gammaGrad != nullptr) {
        memset(gammaGrad, 0, channel * sizeof(float));
        
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                const float* outputGradSlice = outputGrad + (b * channel + c) * spatial;
                const float* normalizedSlice = normalizedPtr + (b * channel + c) * spatial;
                
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
        memset(betaGrad, 0, channel * sizeof(float));
        
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

class CPUInstanceNormGradCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        float epsilon = 1e-5f;
        
        // Try to get epsilon from op parameters if available
        if (op->main_type() == OpParameter_BatchNorm) {
            auto param = op->main_as_BatchNorm();
            if (param->epsilon() > 0) {
                epsilon = param->epsilon();
            }
        }
        
        return new CPUInstanceNormGrad(backend, epsilon);
    }
};

REGISTER_CPU_OP_CREATOR(CPUInstanceNormGradCreator, OpType_InstanceNormGrad);

} // namespace MNN