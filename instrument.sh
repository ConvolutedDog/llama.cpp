#!/bin/bash

# Source code of llama.cpp
LLAMACPP_PATH=$(pwd)
# Path of build
BUILD_PATH=$LLAMACPP_PATH/build
# Path of the executable llama-cli
LLAMA_CLI=$BUILD_PATH/bin/llama-cli
# Path of llama-7B model
MODEL_PATH=$LLAMACPP_PATH/DownloadedModels/llama-7B
# GPU ID of GV100
GPU_ID=4

echo "Using llama.cpp path: $LLAMACPP_PATH"
echo "Using model path: $MODEL_PATH"

if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" != "$GPU_ID" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to $GPU_ID."
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# Compile
if [ -d $BUILD_PATH ] && [ -f $LLAMA_CLI ]; then
  echo "Project already built."
else
  echo "Project not built. Running CMake..."
  cmake -B $BUILD_PATH -DGGML_CUDA=ON -DGGML_CUDA_FORCE_MMQ=true --fresh
fi
cmake --build $BUILD_PATH -j --config Release

# Run a simple case
# $LLAMA_CLI -m $MODEL_PATH/ggml-model-f16.gguf -p "I believe the meaning of life is" -n 128

# Use NVBit to extract the kernels and their params
NVBIT_DOWNLOAD_VER="1.5.5"
NVBIT_DOWNLOAD_ARCH="x86_64" # x86_64 or aarch64
NVBIT_DOWNLOAD_NAME="nvbit-Linux-${NVBIT_DOWNLOAD_ARCH}-${NVBIT_DOWNLOAD_VER}.tar.bz2"
NVBIT_DOWNLOAD_BASEURL="https://github.com/NVlabs/NVBit/releases/download"
NVBIT_DOWNLOAD_URL="${NVBIT_DOWNLOAD_BASEURL}/${NVBIT_DOWNLOAD_VER}/${NVBIT_DOWNLOAD_NAME}"
NVBIT_EXTRACTED_PATH=nvbit_release

# Check if the archive exists
if [ ! -f $NVBIT_DOWNLOAD_NAME ]; then
  echo "Downloading $NVBIT_DOWNLOAD_NAME..."
  wget $NVBIT_DOWNLOAD_URL
  if [ $? -ne 0 ]; then
    echo "Error downloading $NVBIT_DOWNLOAD_NAME. Exiting."
    exit 1
  fi
else
  echo "$NVBIT_DOWNLOAD_NAME already exists."
fi

# Check if the extracted directory exists
if [ ! -d "$NVBIT_EXTRACTED_PATH" ]; then
  echo "Extracting $NVBIT_DOWNLOAD_NAME..."
  tar -jxvf $NVBIT_DOWNLOAD_NAME
  if [ $? -ne 0 ]; then
    echo "Error extracting $NVBIT_DOWNLOAD_NAME. Exiting."
    exit 1
  fi
else
  echo "$NVBIT_EXTRACTED_PATH already exists."
fi

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
  echo "Error: nvcc not found in PATH. Please ensure CUDA is properly installed and configured."
  exit 1
fi

# Get nvcc version
nvcc_version=$(nvcc --version | grep release | awk '{print $5}' | sed 's/,$//')

# Check if the version is 11.4
if [[ "$nvcc_version" != "11.4" ]]; then
  echo "Error: nvcc version is $nvcc_version, but 11.4 is required."
  exit 1
fi

# Proceed with compilation if nvcc version is 11.4
echo "nvcc version is 11.4. Proceeding with compilation..."
make ARCH=70 -j -C $NVBIT_EXTRACTED_PATH/tools

# Instrumention
LD_PRELOAD=$NVBIT_EXTRACTED_PATH/tools/llama_kernels/llama_kernels.so \
$LLAMA_CLI -m $MODEL_PATH/ggml-model-q4_0.gguf -p "I believe the meaning of life is" -n 1 -ngl 1
