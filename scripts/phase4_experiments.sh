#!/bin/bash
# Phase 4: deep-dive CPU experiments. Runs after phase 3 (GPU baseline)
# so every timing window has the machine to itself.
set -u
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
LOGS=/tmp/explogs
PROFILE=benchmarks/results/exp-heads/profile.json

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/progress4.log"; }

mkconf() { # $1 path, $2 data_dir, $3 results_dir, $4 extra lines
  cat > "$1" <<EOF
model = $MODEL
data_dir = $2
results_dir = $3
max_new_tokens = 16
warmup = 1
$4
EOF
}

# ---------- 1. 32K headline: dense vs selection+attn at the model limit ----------
log "32K combined (balanced 0.5 + attn 24)"
mkconf /tmp/p4-32k.conf /tmp/deep/len32k benchmarks/results/exp-len32k "mode = balanced
budget = 0.5
attn_top_k = 24"
$BIN bench --config /tmp/p4-32k.conf --repeats 2 > "$LOGS/p4-32k.log" 2>&1

# ---------- 2. budget x attn-k grid at 8K (2D pareto surface) ----------
for B in 0.25 0.7; do
  for K in 24 48; do
    log "grid budget=$B attn=$K"
    mkconf /tmp/p4-grid.conf /tmp/exp1 benchmarks/results/exp-grid-b$B-k$K "mode = balanced
budget = $B
attn_top_k = $K"
    $BIN bench --config /tmp/p4-grid.conf --repeats 2 > "$LOGS/grid-b$B-k$K.log" 2>&1
  done
done

# ---------- 3. extreme budgets: locate the quality cliff ----------
for B in 0.1 0.15; do
  log "extreme budget=$B"
  mkconf /tmp/p4-ext.conf /tmp/exp1 benchmarks/results/exp-ext-$B "mode = balanced
budget = $B"
  $BIN bench --config /tmp/p4-ext.conf --repeats 2 > "$LOGS/ext-$B.log" 2>&1
done

# ---------- 4. decode impact: smaller KV -> faster generation ----------
log "decode impact (128 new tokens)"
mkconf /tmp/p4-dec.conf /tmp/exp1 benchmarks/results/exp-decode "mode = balanced
budget = 0.5
max_new_tokens = 128"
$BIN bench --config /tmp/p4-dec.conf --repeats 2 > "$LOGS/decode.log" 2>&1

# ---------- 5. multi-needle robustness at three depths ----------
log "multi-needle robustness"
mkconf /tmp/p4-mn.conf /tmp/deep/multineedle benchmarks/results/exp-multineedle "mode = balanced
budget = 0.5
attn_top_k = 48"
$BIN bench --config /tmp/p4-mn.conf --repeats 2 > "$LOGS/multineedle.log" 2>&1

# ---------- 6. perplexity across budgets (quality without generative noise) ----------
for B in 0.25 0.1; do
  log "ppl budget=$B"
  mkconf /tmp/p4-ppl.conf benchmarks/data/ppl benchmarks/results/exp-ppl-b$B "mode = balanced
budget = $B"
  $BIN bench --config /tmp/p4-ppl.conf --repeats 2 > "$LOGS/ppl-b$B.log" 2>&1
done

# ---------- 7. head-adaptive at 16K (per-head sparsity, long context) ----------
if [ -f "$PROFILE" ]; then
  log "head-adaptive 16K"
  mkconf /tmp/p4-h16.conf /tmp/explen/len16k benchmarks/results/exp-head-16k "mode = balanced
budget = 0.5
head_adaptive = true
head_top_k = 128
head_profile = $PROFILE"
  $BIN bench --config /tmp/p4-h16.conf --repeats 2 > "$LOGS/head-16k.log" 2>&1
else
  log "skip head-16k: no profile"
fi

log "PHASE4 DONE"
