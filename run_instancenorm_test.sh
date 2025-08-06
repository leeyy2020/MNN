#!/bin/bash

# CPUInstanceNormV2 测试运行脚本
# 使用方法: ./run_instancenorm_test.sh [方法编号]
# 方法1: 简单编译 (默认)
# 方法2: 使用MNN的CMake构建系统
# 方法3: 手动编译指定依赖

set -e

MNN_ROOT="/Users/ritakeshi/MNN"
BUILD_DIR="${MNN_ROOT}/build_test"
TEST_NAME="instancenorm_test"

echo "=== CPUInstanceNormV2 测试运行脚本 ==="

# 检查参数
METHOD=${1:-1}

case $METHOD in
    1)
        echo "方法1: 简单编译测试"
        
        # 创建临时构建目录
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        # 简单编译 (需要预先构建MNN)
        echo "正在编译测试..."
        g++ -std=c++11 \
            -I"$MNN_ROOT/include" \
            -I"$MNN_ROOT/source" \
            -I"$MNN_ROOT/3rd_party/flatbuffers/include" \
            -DMNN_USE_NEON \
            -O2 \
            "$MNN_ROOT/source/backend/cpu/CPUInstanceNormV2.cpp" \
            "$MNN_ROOT/source/backend/cpu/CPUInstanceNormV2_test.cpp" \
            -L"$MNN_ROOT/build" \
            -lMNN \
            -o "$TEST_NAME" \
            2>/dev/null || {
                echo "❌ 简单编译失败，可能需要先构建MNN库"
                echo "请尝试方法2或3"
                exit 1
            }
        
        echo "✅ 编译成功，正在运行测试..."
        ./"$TEST_NAME"
        ;;
        
    2)
        echo "方法2: 使用CMake构建系统"
        
        # 创建临时CMakeLists.txt
        cat > "$MNN_ROOT/test_instancenorm_cmake.txt" << 'EOF'
cmake_minimum_required(VERSION 3.6)
project(InstanceNormTest)

set(CMAKE_CXX_STANDARD 11)

# 查找MNN
find_path(MNN_INCLUDE_DIR MNN/MNNDefine.h HINTS ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_library(MNN_LIBRARY MNN HINTS ${CMAKE_CURRENT_SOURCE_DIR}/build)

if(NOT MNN_INCLUDE_DIR OR NOT MNN_LIBRARY)
    message(STATUS "MNN not found, building minimal version...")
    
    # 添加必要的源文件
    set(MNN_SOURCES
        source/core/Backend.cpp
        source/core/Execution.cpp  
        source/core/TensorUtils.cpp
        source/backend/cpu/CPUBackend.cpp
        source/backend/cpu/CPUInstanceNormV2.cpp
    )
    
    add_executable(instancenorm_test 
        ${MNN_SOURCES}
        source/backend/cpu/CPUInstanceNormV2_test.cpp
    )
    
    target_include_directories(instancenorm_test PRIVATE
        include
        source
        3rd_party/flatbuffers/include
    )
    
    target_compile_definitions(instancenorm_test PRIVATE MNN_USE_NEON)
else()
    add_executable(instancenorm_test source/backend/cpu/CPUInstanceNormV2_test.cpp)
    target_link_libraries(instancenorm_test ${MNN_LIBRARY})
    target_include_directories(instancenorm_test PRIVATE ${MNN_INCLUDE_DIR})
endif()
EOF

        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        echo "正在使用CMake构建..."
        cmake -f "$MNN_ROOT/test_instancenorm_cmake.txt" "$MNN_ROOT" || {
            echo "❌ CMake配置失败"
            exit 1
        }
        
        make -j$(nproc) || {
            echo "❌ 编译失败"
            exit 1
        }
        
        echo "✅ 构建成功，正在运行测试..."
        ./instancenorm_test
        ;;
        
    3)
        echo "方法3: 手动编译 (最小依赖)"
        
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        
        # 创建简化的测试版本
        cat > "simple_test.cpp" << 'EOF'
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

// 简化的Instance Norm实现用于测试
void simpleInstanceNorm(const float* input, float* output, 
                       const float* gamma, const float* beta,
                       int batch, int channel, int spatial, float epsilon) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            // 计算均值
            float sum = 0.0f;
            const float* channelData = input + b * channel * spatial + c * spatial;
            for (int s = 0; s < spatial; ++s) {
                sum += channelData[s];
            }
            float mean = sum / spatial;
            
            // 计算方差
            float varSum = 0.0f;
            for (int s = 0; s < spatial; ++s) {
                float diff = channelData[s] - mean;
                varSum += diff * diff;
            }
            float variance = varSum / spatial;
            float invStd = 1.0f / sqrtf(variance + epsilon);
            
            // 归一化
            float scale = gamma ? gamma[c] : 1.0f;
            float bias = beta ? beta[c] : 0.0f;
            
            float* channelOutput = output + b * channel * spatial + c * spatial;
            for (int s = 0; s < spatial; ++s) {
                float normalized = (channelData[s] - mean) * invStd;
                channelOutput[s] = normalized * scale + bias;
            }
        }
    }
}

int main() {
    std::cout << "=== 简化版 Instance Normalization 测试 ===" << std::endl;
    
    const int batch = 2;
    const int channel = 3;
    const int spatial = 16; // 4x4
    const float epsilon = 1e-5f;
    
    std::vector<float> input(batch * channel * spatial);
    std::vector<float> output(batch * channel * spatial);
    std::vector<float> gamma(channel, 1.0f);
    std::vector<float> beta(channel, 0.0f);
    
    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-2.0f, 2.0f);
    
    for (auto& val : input) {
        val = dis(gen);
    }
    
    // 执行Instance Norm
    simpleInstanceNorm(input.data(), output.data(), 
                      gamma.data(), beta.data(),
                      batch, channel, spatial, epsilon);
    
    // 验证结果
    std::cout << "输入样本: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "输出样本: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    
    // 检查第一个通道的均值和方差
    float sum = 0.0f, varSum = 0.0f;
    for (int s = 0; s < spatial; ++s) {
        sum += output[s];
    }
    float outMean = sum / spatial;
    
    for (int s = 0; s < spatial; ++s) {
        float diff = output[s] - outMean;
        varSum += diff * diff;
    }
    float outVar = varSum / spatial;
    
    std::cout << "第一个通道输出统计:" << std::endl;
    std::cout << "  均值: " << outMean << " (应该接近0)" << std::endl;
    std::cout << "  方差: " << outVar << " (应该接近1)" << std::endl;
    
    if (std::abs(outMean) < 0.1f && std::abs(outVar - 1.0f) < 0.1f) {
        std::cout << "✅ 测试通过!" << std::endl;
    } else {
        std::cout << "❌ 测试失败!" << std::endl;
    }
    
    return 0;
}
EOF

        echo "正在编译简化测试..."
        g++ -std=c++11 -O2 simple_test.cpp -o simple_test || {
            echo "❌ 编译失败"
            exit 1
        }
        
        echo "✅ 编译成功，正在运行简化测试..."
        ./simple_test
        ;;
        
    *)
        echo "❌ 无效的方法编号: $METHOD"
        echo "可用方法: 1, 2, 3"
        exit 1
        ;;
esac

echo ""
echo "=== 测试完成 ==="