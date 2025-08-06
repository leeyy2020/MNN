//
//  CPUInstanceNormV2.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include <limits>
#include "CPUInstanceNormV2.hpp"
#include "CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "math/Vec.hpp"
#include <cmath>

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {

CPUInstanceNormV2::CPUInstanceNormV2(Backend* backend, float epsilon, bool multiThread)
    : Execution(backend), mEpsilon(epsilon), mSupportMultiThread(multiThread) {
    // Do nothing
}

void CPUInstanceNormV2::_scheduleComputeMeanVar(int batch, int channel, int spatial) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    
    mPreFunctions.emplace_back(std::make_pair([batch, channel, spatial, this](
        int tId, const float* input, const float* gamma, const float* beta, float* output) {
        
        auto meanPtr = (float*)mTempMean.ptr();
        auto varPtr = (float*)mTempVar.ptr();
        
        // Compute mean and variance for each batch and channel
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < channel; ++c) {
                const float* channelData = input + b * channel * spatial + c * spatial;
                int idx = b * channel + c;
                
                // Compute mean
                float sum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    sum += channelData[s];
                }
                float mean = sum / spatial;
                meanPtr[idx] = mean;
                
                // Compute variance
                float varSum = 0.0f;
                for (int s = 0; s < spatial; ++s) {
                    float diff = channelData[s] - mean;
                    varSum += diff * diff;
                }
                float variance = varSum / spatial;
                varPtr[idx] = variance;
            }
        }
    }, 1));
}

void CPUInstanceNormV2::_scheduleNormalization(int batch, int channel, int spatial) {
    int numberThread = mSupportMultiThread ? static_cast<CPUBackend*>(backend())->threadNumber() : 1;
    
    mPreFunctions.emplace_back(std::make_pair([batch, channel, spatial, this](
        int tId, const float* input, const float* gamma, const float* beta, float* output) {
        
        auto meanPtr = (const float*)mTempMean.ptr();
        auto varPtr = (const float*)mTempVar.ptr();
        
        MNN_CONCURRENCY_BEGIN(bcIdx, batch * channel) {
            int b = bcIdx / channel;
            int c = bcIdx % channel;
            int idx = b * channel + c;
            
            const float* channelInput = input + b * channel * spatial + c * spatial;
            float* channelOutput = output + b * channel * spatial + c * spatial;
            
            float mean = meanPtr[idx];
            float variance = varPtr[idx];
            float invStd = 1.0f / sqrtf(variance + mEpsilon);
            
            float scale = gamma ? gamma[c] : 1.0f;
            float bias = beta ? beta[c] : 0.0f;
            
#ifdef MNN_USE_NEON
            if (spatial >= 4) {
                float32x4_t vMean = vdupq_n_f32(mean);
                float32x4_t vInvStd = vdupq_n_f32(invStd);
                float32x4_t vScale = vdupq_n_f32(scale);
                float32x4_t vBias = vdupq_n_f32(bias);
                
                int spatial4 = spatial & (~3);
                for (int s = 0; s < spatial4; s += 4) {
                    float32x4_t vInput = vld1q_f32(channelInput + s);
                    float32x4_t vNorm = vmulq_f32(vsubq_f32(vInput, vMean), vInvStd);
                    float32x4_t vOutput = vmlaq_f32(vBias, vNorm, vScale);
                    vst1q_f32(channelOutput + s, vOutput);
                }
                
                // Handle remaining elements
                for (int s = spatial4; s < spatial; ++s) {
                    float normalized = (channelInput[s] - mean) * invStd;
                    channelOutput[s] = normalized * scale + bias;
                }
            } else {
#endif
                for (int s = 0; s < spatial; ++s) {
                    float normalized = (channelInput[s] - mean) * invStd;
                    channelOutput[s] = normalized * scale + bias;
                }
#ifdef MNN_USE_NEON
            }
#endif
        }
        MNN_CONCURRENCY_END();
    }, numberThread));
}

ErrorCode CPUInstanceNormV2::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* input = inputs[0];
    Tensor* output = outputs[0];
    
    mPreFunctions.clear();
    
    // Assume input format is NCHW
    int batch = input->batch();
    int channel = input->channel();
    int height = input->height();
    int width = input->width();
    int spatial = height * width;
    
    mBatch = batch;
    mChannel = channel;
    mSpatial = spatial;
    
    auto bufferAlloc = static_cast<CPUBackend*>(backend())->getBufferAllocator();
    
    // Allocate temporary buffers for mean and variance
    auto meanAlloc = bufferAlloc->alloc(batch * channel * sizeof(float));
    auto varAlloc = bufferAlloc->alloc(batch * channel * sizeof(float));
    
    if (meanAlloc.invalid() || varAlloc.invalid()) {
        return OUT_OF_MEMORY;
    }
    
    mTempMean = meanAlloc;
    mTempVar = varAlloc;
    
    // Schedule computation
    _scheduleComputeMeanVar(batch, channel, spatial);
    _scheduleNormalization(batch, channel, spatial);
    
    // Free temporary allocations (they will be kept until execution)
    bufferAlloc->free(meanAlloc);
    bufferAlloc->free(varAlloc);
    
    return NO_ERROR;
}

ErrorCode CPUInstanceNormV2::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputPtr = inputs[0]->host<float>();
    auto outputPtr = outputs[0]->host<float>();
    
    const float* gammaPtr = nullptr;
    const float* betaPtr = nullptr;
    
    // Check if gamma (scale) is provided
    if (inputs.size() > 1 && inputs[1] != nullptr) {
        gammaPtr = inputs[1]->host<float>();
    }
    
    // Check if beta (bias) is provided
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        betaPtr = inputs[2]->host<float>();
    }
    
    execute(inputPtr, gammaPtr, betaPtr, outputPtr);
    return NO_ERROR;
}

void CPUInstanceNormV2::execute(const float* input, const float* gamma, const float* beta, float* output) {
    for (auto& f : mPreFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.second) {
            f.first(tId, input, gamma, beta, output);
        }
        MNN_CONCURRENCY_END();
    }
}

class CPUInstanceNormV2Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        // Default epsilon value
        float epsilon = 1e-5f;
        
        // Try to get epsilon from op parameters if available
        if (op->main_type() == OpParameter_BatchNorm) {
            auto param = op->main_as_BatchNorm();
            if (param->epsilon() > 0) {
                epsilon = param->epsilon();
            }
        }
        
        return new CPUInstanceNormV2(backend, epsilon, true);
    }
};

// Register the creator - using a different name to avoid conflicts
REGISTER_CPU_OP_CREATOR(CPUInstanceNormV2Creator, OpType_InstanceNorm);

} // namespace MNN