# CPUInstanceNormV2 实现说明

## 概述

`CPUInstanceNormV2` 是一个参照 `CPUMatMul` 结构设计的新的 Instance Normalization 实现，提供了高效的CPU计算支持。

## 设计特点

### 1. 架构设计
- **参照CPUMatMul结构**: 采用了与矩阵乘法相似的设计模式
- **多阶段执行**: 使用 `mPreFunctions` 来组织计算流程
- **内存管理**: 使用 `MemChunk` 进行临时缓冲区管理

### 2. 核心功能

#### Instance Normalization公式
```
output[b,c,h,w] = (input[b,c,h,w] - mean[b,c]) * gamma[c] / sqrt(var[b,c] + epsilon) + beta[c]
```

其中：
- `mean[b,c]` 和 `var[b,c]` 是在空间维度(H×W)上计算的均值和方差
- `gamma[c]` 是可学习的缩放参数
- `beta[c]` 是可学习的偏移参数
- `epsilon` 是防止除零的小常数

#### 计算流程
1. **均值和方差计算**: `_scheduleComputeMeanVar()` - 为每个batch和channel计算统计信息
2. **归一化计算**: `_scheduleNormalization()` - 应用归一化公式

### 3. 性能优化

#### 多线程支持
- 支持多线程并行计算
- 可通过构造函数参数控制是否启用多线程

#### SIMD优化
- 支持ARM NEON指令集加速
- 向量化处理空间维度的计算
- 对于空间大小≥4的情况使用4路并行

#### 内存优化
- 使用临时缓冲区存储中间结果
- 通过BufferAllocator进行高效内存管理
- 在onResize阶段预分配所需内存

## 文件结构

```
CPUInstanceNormV2.hpp          # 头文件，定义接口
CPUInstanceNormV2.cpp          # 主要实现
CPUInstanceNormV2_test.cpp     # 测试代码
```

## 使用方法

### 基本使用
```cpp
// 创建Instance Norm执行器
float epsilon = 1e-5f;
bool multiThread = true;
auto instanceNorm = new CPUInstanceNormV2(backend, epsilon, multiThread);

// 设置输入输出
std::vector<Tensor*> inputs = {input, gamma, beta};  // gamma和beta可选
std::vector<Tensor*> outputs = {output};

// 执行
instanceNorm->onResize(inputs, outputs);
instanceNorm->onExecute(inputs, outputs);
```

### 输入格式
- **input**: 形状为 [N, C, H, W] 的4D张量
- **gamma**: 形状为 [C] 的1D张量（可选，缩放参数）
- **beta**: 形状为 [C] 的1D张量（可选，偏移参数）

## 与原版本对比

| 特性 | CPUInstanceNorm | CPUInstanceNormV2 |
|------|-----------------|-------------------|
| 架构设计 | 简单直接 | 参照CPUMatMul，更模块化 |
| 多线程 | 基础支持 | 高级多线程调度 |
| 内存管理 | 基础 | 使用BufferAllocator |
| 执行模式 | 直接执行 | 预计算+执行分离 |
| SIMD优化 | NEON支持 | 增强的NEON支持 |

## 编译和测试

### 编译
确保在MNN项目中包含新文件：
```cmake
# 添加到CMakeLists.txt
set(MNN_CPU_SRC
    # ... 其他文件
    ${CMAKE_CURRENT_LIST_DIR}/CPUInstanceNormV2.cpp
)
```

### 测试
```cpp
// 取消注释测试文件中的main函数
int main() {
    MNN::testInstanceNormV2();
    return 0;
}
```

## 技术细节

### 内存布局
- 输入: NCHW格式
- 临时缓冲区: 连续存储每个(batch, channel)的统计信息
- 输出: 与输入相同的NCHW格式

### 计算复杂度
- 时间复杂度: O(N×C×H×W)
- 空间复杂度: O(N×C) （临时缓冲区）

### 线程安全
- onResize阶段: 非线程安全，需要外部同步
- onExecute阶段: 线程安全的并行执行

## 扩展性

该实现提供了良好的扩展性：
- 可以轻松添加新的优化策略
- 支持不同数据类型的扩展
- 便于添加更多SIMD指令集支持

## 注意事项

1. **数据格式**: 当前实现假设输入为NCHW格式
2. **参数可选**: gamma和beta参数可以为nullptr，此时使用默认值
3. **内存对齐**: 使用了MNN的内存对齐机制
4. **错误处理**: 包含了完整的错误检查和处理

这个实现展示了如何将CPUMatMul的优秀设计模式应用到其他算子中，提供了高性能和良好的可维护性。