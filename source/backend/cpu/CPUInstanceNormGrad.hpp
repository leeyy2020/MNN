//
//  CPUInstanceNormGrad.hpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#ifndef CPUInstanceNormGrad_hpp
#define CPUInstanceNormGrad_hpp

#include "core/Execution.hpp"
#include "core/AutoStorage.h"
#include "core/BufferAllocator.hpp"
#include <vector>
#include <functional>

namespace MNN {

class CPUInstanceNormGrad : public Execution {
public:
    CPUInstanceNormGrad(Backend* backend, float epsilon = 1e-5f);
    virtual ~CPUInstanceNormGrad() = default;
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    void computeGradients(const float* input, const float* outputGrad, 
                         const float* gamma, const float* beta,
                         float* inputGrad, float* gammaGrad, float* betaGrad,
                         int batch, int channel, int spatial);
    
    float mEpsilon;
    
    // Dimensions
    int mBatch = 0;
    int mChannel = 0; 
    int mSpatial = 0;
    
    // Temporary buffers
    MemChunk mTempMean;
    MemChunk mTempVar;
    MemChunk mTempNormalized;
};

} // namespace MNN

#endif /* CPUInstanceNormGrad_hpp */