//
//  InstanceNormGrad.cpp
//  MNN
//
//  Created by MNN on 2024/12/18.
//  Copyright © 2024, Alibaba Group Holding Limited
//

#include "InstanceNormGrad.hpp"
#include "core/Macro.h"
using namespace std;
namespace MNN {
using namespace MNN::Express;

class InstanceNormGrad : public OpGrad {
public:
    InstanceNormGrad() {
        mType = LINEAR;
    }
    
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        std::vector<VARP> res(inputs.size(), nullptr);
        auto outputDiff = backwardOutput[0];
        
        // Get input tensor
        auto input = inputs[0];
        
        // Get gamma (scale) and beta (bias) if they exist
        VARP gamma = nullptr;
        VARP beta = nullptr;
        
        if (inputs.size() > 1 && inputs[1].get() != nullptr) {
            gamma = inputs[1];
        }
        if (inputs.size() > 2 && inputs[2].get() != nullptr) {
            beta = inputs[2];
        }
        
        // Create inputs for InstanceNormGrad backend op
        std::vector<VARP> gradInputs;
        gradInputs.push_back(input);      // forward input
        gradInputs.push_back(outputDiff); // output gradient
        if (gamma.get() != nullptr) {
            gradInputs.push_back(gamma);  // gamma parameter
        } else {
            gradInputs.push_back(nullptr);
        }
        if (beta.get() != nullptr) {
            gradInputs.push_back(beta);   // beta parameter  
        } else {
            gradInputs.push_back(nullptr);
        }
        
        // Get epsilon from op parameters
        float epsilon = 1e-5f;
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        if (forwardOp->main.type == OpParameter_BatchNorm) {
            auto param = forwardOp->main.AsBatchNorm();
            if (param->epsilon > 0) {
                epsilon = param->epsilon;
            }
        }
        
        // Create InstanceNormGrad op with BatchNorm parameters (reusing the parameter structure)
        std::unique_ptr<OpT> gradOp(new OpT);
        gradOp->type = OpType_InstanceNormGrad;
        gradOp->main.type = OpParameter_BatchNorm;
        gradOp->main.value = new BatchNormT;
        auto gradParam = gradOp->main.AsBatchNorm();
        gradParam->epsilon = epsilon;
        
        // Create the backend InstanceNormGrad operation
        auto gradVar = Variable::create(Expr::create(gradOp.get(), gradInputs));
        
        // Extract results - the backend op should return gradients in order:
        // [input_grad, gamma_grad, beta_grad]
        res[0] = Variable::create(gradVar->expr().first, 0); // input gradient
        
        if (inputs.size() > 1 && gamma.get() != nullptr) {
            res[1] = Variable::create(gradVar->expr().first, 1); // gamma gradient
        }
        
        if (inputs.size() > 2 && beta.get() != nullptr) {
            res[2] = Variable::create(gradVar->expr().first, 2); // beta gradient
        }
        
        return res;
    }
};

static void _create() {
    static InstanceNormGrad _c;
    OpGrad::insert(OpType_InstanceNorm, &_c);
}

REGISTER_GRAD(InstanceNormGrad_cpp, _create);

} // namespace MNN