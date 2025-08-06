#!/bin/bash

# 快速构建MNN并测试CPUInstanceNormV2
# 这个脚本会构建最小的MNN库来支持我们的测试

set -e

MNN_ROOT="/Users/ritakeshi/MNN"
BUILD_DIR="${MNN_ROOT}/build_minimal"

echo "=== 快速构建MNN并测试CPUInstanceNormV2 ==="

cd "$MNN_ROOT"

# 清理并创建构建目录
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "正在配置CMake..."
cmake .. \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_DEMO=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_QUANTOOLS=OFF \
    -DMNN_EVALUATION=OFF \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_SUPPORT_BF16=OFF \
    -DMNN_ARM82=OFF \
    -DMNN_OPENCL=OFF \
    -DMNN_VULKAN=OFF \
    -DMNN_METAL=OFF \
    -DMNN_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64

echo "正在构建MNN库 (仅CPU后端)..."
make -j$(sysctl -n hw.ncpu) MNN

echo "✅ MNN库构建完成"

# 现在编译我们的测试
echo "正在编译CPUInstanceNormV2测试..."

# 检查库文件是否存在
if [[ ! -f "libMNN.dylib" && ! -f "libMNN.so" ]]; then
    echo "❌ MNN库文件未找到"
    exit 1
fi

# 编译测试程序
g++ -std=c++11 \
    -I"$MNN_ROOT/include" \
    -I"$MNN_ROOT/source" \
    -I"$MNN_ROOT/3rd_party/flatbuffers/include" \
    -DMNN_USE_NEON \
    -O2 \
    "$MNN_ROOT/source/backend/cpu/CPUInstanceNormV2.cpp" \
    "$MNN_ROOT/source/backend/cpu/CPUInstanceNormV2_test.cpp" \
    -L"$BUILD_DIR" \
    -lMNN \
    -o instancenorm_test

echo "✅ 测试程序编译完成"

# 设置库路径
export DYLD_LIBRARY_PATH="$BUILD_DIR:$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH"

echo "正在运行CPUInstanceNormV2测试..."
./instancenorm_test

echo ""
echo "=== 测试完成 ==="
echo "构建目录: $BUILD_DIR"
echo "测试程序: $BUILD_DIR/instancenorm_test"