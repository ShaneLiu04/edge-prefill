#!/bin/bash
# Build the patched edge-prefill with the CUDA backend (Turing sm_75).
# Requires: nvcc from nvidia-cuda-toolkit plus gcc-12 (CUDA 12.0 rejects
# gcc-13). Source is copied to ext4 so the /mnt/c live tree stays pristine
# and compilation does not fight the 9P filesystem.
set -eu
cd /mnt/d/prefill

if [ ! -d /root/llama-src-cuda ]; then
  cp -r /mnt/c/Users/L30086046/AppData/Local/Temp/opencode/llama-src /root/llama-src-cuda
  rm -rf /root/llama-src-cuda/build-edge /root/llama-src-cuda/.git
fi

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DEDGE_PREFILL_LLAMA_DIR=/root/llama-src-cuda \
  -DEDGE_PREFILL_WITH_CUDA=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build-cuda -j8
echo "CUDA BUILD OK"
