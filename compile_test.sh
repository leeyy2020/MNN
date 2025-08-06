#!/bin/bash

# Compile test for CPUInstanceNormGrad
echo "Compiling CPUInstanceNormGrad test..."

# Set compiler flags
CXX_FLAGS="-std=c++11 -O2 -I. -I./include -I./source -I./schema/current"

# Add source files
SOURCES="test_cpu_instancenorm_grad.cpp source/backend/cpu/CPUInstanceNormGrad.cpp"

# Try to compile
g++ $CXX_FLAGS $SOURCES -o test_cpu_instancenorm_grad

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running test..."
    ./test_cpu_instancenorm_grad
else
    echo "Compilation failed!"
    exit 1
fi