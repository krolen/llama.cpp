#!/bin/bash

# 1. Set CUDA Environment Variables
# These tell CMake exactly where to find the 5090-compatible compiler
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_PATH/bin/nvcc

# 2. Safety Check
if ! command -v nvcc &> /dev/null
then
    echo "Error: nvcc (CUDA Compiler) not found at $CUDA_PATH/bin/nvcc"
    echo "Please check if CUDA 13.1 is installed correctly."
    exit 1
fi

echo "--- Starting Clean Build for RTX 5090 ---"

# 3. Clean previous failed attempts
rm -rf build
mkdir build
cd build

# 4. Run CMake with Blackwell (sm_100) Optimizations
# We explicitly set CUDAToolkit_ROOT to solve your 'library root' error
cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT=$CUDA_PATH \
  -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF

# 5. Compile using all available CPU cores
echo "--- Compiling Binaries ---"
cmake --build . --config Release -j$(nproc)

echo "--- Build Successful ---"
echo "Your optimized binaries are now in: $(pwd)/bin"
