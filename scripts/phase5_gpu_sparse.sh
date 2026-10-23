#!/bin/bash
# Phase 5: GPU sparse experiments with our patched CUDA build. Waits for
# (a) nvcc to appear (apt install finishing) and (b) phase 4 to finish, so
# the CUDA build and the GPU timings never contend with CPU benchmarks.
set -u
LOGS=/tmp/explogs
while ! command -v nvcc > /dev/null 2>&1; do sleep 120; done
while pgrep -f "phase4_experiments[.]sh" > /dev/null; do sleep 60; done
sleep 30

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/progress5.log"; }

log "building CUDA binary"
bash /mnt/d/prefill/scripts/build_cuda.sh > "$LOGS/build-cuda.log" 2>&1 || {
  log "CUDA build FAILED, see $LOGS/build-cuda.log"
  exit 1
}

cd /mnt/d/prefill
BIN=./build-cuda/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
PROFILE=benchmarks/results/exp-heads/profile.json

mkconf() { # $1 path, $2 data_dir, $3 results_dir, $4 extra lines
  cat > "$1" <<EOF
model = $MODEL
data_dir = $2
results_dir = $3
max_new_tokens = 16
warmup = 1
threads = 4
n_gpu_layers = 99
$4
EOF
}

log "GPU 8K selection"
mkconf /tmp/p5-8k.conf /tmp/exp1 benchmarks/results/exp-gpu-8k-sel "mode = balanced
budget = 0.5"
$BIN bench --config /tmp/p5-8k.conf --repeats 3 > "$LOGS/gpu-8k-sel.log" 2>&1

log "GPU 8K selection+attn24"
mkconf /tmp/p5-8ka.conf /tmp/exp1 benchmarks/results/exp-gpu-8k-attn "mode = balanced
budget = 0.5
attn_top_k = 24"
$BIN bench --config /tmp/p5-8ka.conf --repeats 3 > "$LOGS/gpu-8k-attn.log" 2>&1

log "GPU 16K selection+attn24"
mkconf /tmp/p5-16k.conf /tmp/explen/len16k benchmarks/results/exp-gpu-16k "mode = balanced
budget = 0.5
attn_top_k = 24"
$BIN bench --config /tmp/p5-16k.conf --repeats 3 > "$LOGS/gpu-16k.log" 2>&1

log "GPU 32K selection+attn24"
mkconf /tmp/p5-32k.conf /tmp/deep/len32k benchmarks/results/exp-gpu-32k "mode = balanced
budget = 0.5
attn_top_k = 24"
$BIN bench --config /tmp/p5-32k.conf --repeats 3 > "$LOGS/gpu-32k.log" 2>&1

if [ -f "$PROFILE" ]; then
  log "GPU 8K head-adaptive"
  mkconf /tmp/p5-head.conf /tmp/exp1 benchmarks/results/exp-gpu-8k-head "mode = balanced
budget = 0.5
head_adaptive = true
head_top_k = 128
head_profile = $PROFILE"
  $BIN bench --config /tmp/p5-head.conf --repeats 3 > "$LOGS/gpu-8k-head.log" 2>&1
fi

log "PHASE5 DONE"
