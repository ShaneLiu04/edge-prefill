#!/bin/bash
# Orchestrator: sequential execution of the remaining experiment chain with
# explicit stage boundaries (replaces the wait-loop watchers, whose
# wait-for-process logic raced when the target had not started yet).
#
#   phase2 (running) -> GPU dense baseline -> phase4 -> phase5 (waits nvcc)
set -u
LOG=/tmp/explogs/orchestrator.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 1. wait for phase 2 (it is running right now) ----
while pgrep -f "phase2_experiments[.]sh" > /dev/null; do sleep 60; done
sleep 30
log "phase2 finished"

# ---- 2. clean GPU dense baseline (discard the contention-tainted CSV) ----
rm -f /mnt/d/prefill/benchmarks/results/exp-gpu-dense/*.csv
log "GPU dense baseline start"
powershell.exe -ExecutionPolicy Bypass -File 'D:\prefill\scripts\run_gpu_baseline.ps1' \
  > /tmp/explogs/gpu-baseline-clean.log 2>&1
log "GPU dense baseline done"

# ---- 3. deep-dive CPU matrix ----
log "phase4 start"
bash /mnt/d/prefill/scripts/phase4_experiments.sh >> "$LOG" 2>&1
log "phase4 done"

# ---- 4. GPU sparse experiments (phase5 waits for nvcc internally) ----
log "phase5 start (waits for nvcc if needed)"
bash /mnt/d/prefill/scripts/phase5_gpu_sparse.sh >> "$LOG" 2>&1
log "ALL PHASES DONE"
