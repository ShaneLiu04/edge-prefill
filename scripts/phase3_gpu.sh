#!/bin/bash
# Phase 3: after phase 2 (CPU experiments) finishes, run the GPU dense
# baseline cleanly on the Windows side (no CPU contention from WSL).
while pgrep -f "phase2_experiments[.]sh" > /dev/null; do
  sleep 60
done
sleep 30
powershell.exe -ExecutionPolicy Bypass -File 'D:\prefill\scripts\run_gpu_baseline.ps1' \
  > /tmp/explogs/gpu-baseline.log 2>&1
echo "[$(date +%H:%M:%S)] GPU baseline done" >> /tmp/explogs/progress2.log
