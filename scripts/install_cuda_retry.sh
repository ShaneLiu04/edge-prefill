#!/bin/bash
# Retry CUDA toolkit + python3 install through the flaky corporate proxy.
# apt caches partial downloads, so every retry makes forward progress;
# alternate proxied and direct (no-proxy) attempts in case direct egress works.
LOG=/tmp/cuda-install.log
echo "=== install start $(date)" >> "$LOG"
PKGS="nvidia-cuda-toolkit python3 python3-pip"
for i in $(seq 1 60); do
  echo "=== attempt $i (proxied) $(date)" >> "$LOG"
  apt-get install -y --fix-missing $PKGS >> "$LOG" 2>&1
  if command -v nvcc >/dev/null 2>&1; then
    echo "NVCC_OK $(nvcc --version | tail -1)" >> "$LOG"
    break
  fi
  echo "=== attempt $i (direct) $(date)" >> "$LOG"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
      apt-get install -y --fix-missing $PKGS >> "$LOG" 2>&1
  if command -v nvcc >/dev/null 2>&1; then
    echo "NVCC_OK_DIRECT $(nvcc --version | tail -1)" >> "$LOG"
    break
  fi
  sleep 20
done
python3 --version >> "$LOG" 2>&1
echo "=== install done $(date)" >> "$LOG"
