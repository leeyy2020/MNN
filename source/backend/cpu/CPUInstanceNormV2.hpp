//
//  CPUInstanceNormV2.hpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#ifndef CPUInstanceNormV2_hpp
#define CPUInstanceNormV2_hpp

#include "core/Execution.hpp"
#include "core/AutoStorage.h"
#include "core/BufferAllocator.hpp"
#include <vector>
#include <functional>

namespace MNN {

class CPUInstanceNormV2 : public Execution {
public:
    CPUInstanceNormV2(Backend* backend, float epsilon, bool multiThread = true);
    virtual ~CPUInstanceNormV2() = default;
    
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    
    void execute(const float* input, const float* gamma, const float* beta, float* output);

private:
    void _scheduleComputeMeanVar(int batch, int channel, int spatial);
    void _scheduleNormalization(int batch, int channel, int spatial);
    
    float mEpsilon;
    bool mSupportMultiThread;
    
    // Dimensions
    int mBatch = 0;
    int mChannel = 0; 
    int mSpatial = 0;
    
    // Temporary buffers
    MemChunk mTempMean;
    MemChunk mTempVar;
    
    // Pre-computed functions for execution
    std::vector<std::pair<std::function<void(int, const float*, const float*, const float*, float*)>, int>> mPreFunctions;
};

} // namespace MNN

#endif /* CPUInstanceNormV2_hpp */