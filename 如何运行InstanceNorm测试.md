# 如何运行 CPUInstanceNormV2 测试

## 快速开始 🚀

我们提供了多种运行测试的方法：

### 🎯 推荐方式 (最快)
```bash
cd /Users/ritakeshi/MNN
./run_instancenorm_test.sh 3
```
这会运行一个简化但完整的Instance Normalization测试，无需构建MNN库。

### 🔧 完整测试 (需要构建)
```bash
cd /Users/ritakeshi/MNN
./quick_build_and_test.sh
```
这会构建最小化的MNN库并运行完整的CPUInstanceNormV2测试。

## 运行方法详解

### 方法1: 简单编译 (需要预先构建MNN)

```bash
./run_instancenorm_test.sh 1
```

**适用场景**: 已经构建了完整的MNN库
**优点**: 使用完整的MNN框架，测试最真实
**缺点**: 需要先构建整个MNN项目

### 方法2: 使用CMake构建系统

```bash
./run_instancenorm_test.sh 2
```

**适用场景**: 希望使用CMake管理构建过程
**优点**: 自动处理依赖关系
**缺点**: 构建时间较长

### 方法3: 手动编译 (推荐) ✅

```bash
./run_instancenorm_test.sh 3
```

**适用场景**: 快速验证算法正确性
**优点**: 
- 编译快速，无需外部依赖
- 包含完整的Instance Normalization实现
- 自动验证结果正确性
**缺点**: 简化版实现，不包含所有优化

## 测试结果解读

### 成功运行的输出示例：
```
=== 简化版 Instance Normalization 测试 ===
输入样本: -0.129846 -1.885 1.78527 -0.0207847 -0.686938 
输出样本: -0.053272 -1.53398 1.56239 0.0387362 -0.523254 
第一个通道输出统计:
  均值: 2.98023e-08 (应该接近0)
  方差: 0.999993 (应该接近1)
✅ 测试通过!
```

### 关键指标：
- **均值接近0**: Instance Norm会将每个通道的均值归一化到0
- **方差接近1**: Instance Norm会将每个通道的方差归一化到1
- **输出值有意义**: 不包含NaN或无穷大值

## 手动运行测试

如果你想手动编译和运行：

### 1. 编译简化版测试
```bash
cd /Users/ritakeshi/MNN/build_test
g++ -std=c++11 -O2 simple_test.cpp -o simple_test
./simple_test
```

### 2. 编译完整版测试 (需要MNN库)
```bash
cd /Users/ritakeshi/MNN
g++ -std=c++11 \
    -I"include" \
    -I"source" \
    -I"3rd_party/flatbuffers/include" \
    -DMNN_USE_NEON \
    -O2 \
    "source/backend/cpu/CPUInstanceNormV2.cpp" \
    "source/backend/cpu/CPUInstanceNormV2_test.cpp" \
    -L"build" \
    -lMNN \
    -o instancenorm_test

./instancenorm_test
```

## 性能测试

要测试性能，可以修改测试参数：

```cpp
// 在测试文件中修改这些值
const int batch = 8;      // 增加batch size
const int channel = 64;   // 增加通道数
const int height = 224;   // 增加图像尺寸
const int width = 224;
```

## 故障排除

### 编译错误
1. **找不到头文件**: 确保在MNN根目录运行
2. **链接错误**: 使用方法3，它不需要链接MNN库
3. **权限错误**: 运行 `chmod +x run_instancenorm_test.sh`

### 运行时错误
1. **段错误**: 检查输入张量的内存分配
2. **NaN结果**: 检查epsilon值是否太小
3. **性能问题**: 在Release模式下编译 (`-O2`)

## 验证算法正确性

Instance Normalization的数学公式：
```
output[b,c,h,w] = (input[b,c,h,w] - mean[b,c]) * gamma[c] / sqrt(var[b,c] + epsilon) + beta[c]
```

其中：
- `mean[b,c]` 和 `var[b,c]` 在空间维度 (H×W) 上计算
- 每个batch和channel都有独立的统计信息

### 手动验证步骤：
1. 检查输出的均值是否接近0
2. 检查输出的方差是否接近1
3. 验证数值稳定性（无NaN或Inf）

## 下一步

测试通过后，你可以：
1. 集成到MNN的构建系统中
2. 添加更多的单元测试
3. 进行性能基准测试
4. 与其他框架的实现进行对比

## 联系和支持

如果遇到问题，检查：
1. 编译器版本 (支持C++11)
2. 系统架构 (ARM64/x86_64)
3. 依赖库版本

测试脚本会自动处理大部分常见问题，建议优先使用方法3进行快速验证。