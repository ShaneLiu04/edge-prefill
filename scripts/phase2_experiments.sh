#!/bin/bash
# Phase 2: head profile, fixed len2k rerun, and AR003 head-adaptive benches.
# Run AFTER run_experiments.sh prints ALL DONE (avoids CPU contention).
set -u
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
LOGS=/tmp/explogs
PROFILE=benchmarks/results/exp-heads/profile.json

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/progress2.log"; }

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

# ---------- 1. head profile (exp-heads dir was missing in phase 1) ----------
log "classify-heads start"
mkdir -p benchmarks/results/exp-heads
$BIN classify-heads --model $MODEL --output $PROFILE --threads 16 > "$LOGS/heads2.log" 2>&1
log "classify-heads: $(tail -1 $LOGS/heads2.log)"

# ---------- 2. len2k-sel rerun on the non-flash binary ----------
log "len2k-sel rerun"
mkconf /tmp/p2-len2k.conf /tmp/explen/len2k benchmarks/results/exp-len-len2k-sel "mode = balanced
budget = 0.5"
$BIN bench --config /tmp/p2-len2k.conf --repeats 2 > "$LOGS/p2-len2k.log" 2>&1

# ---------- 3. head-adaptive sweep on 8K (budget 0.5) ----------
for K in 64 128 256; do
  log "head-adaptive k=$K"
  mkconf /tmp/p2-head.conf /tmp/exp1 benchmarks/results/exp-headk-$K "mode = balanced
budget = 0.5
head_adaptive = true
head_top_k = $K
head_profile = $PROFILE"
  $BIN bench --config /tmp/p2-head.conf --repeats 2 > "$LOGS/headk-$K.log" 2>&1
done

# ---------- 4. head-adaptive quality breadth ----------
log "qa head-adaptive"
mkconf /tmp/p2-qa.conf benchmarks/data/qa2 benchmarks/results/exp-qa-head "mode = balanced
budget = 0.5
head_adaptive = true
head_top_k = 128
head_profile = $PROFILE"
$BIN bench --config /tmp/p2-qa.conf --repeats 2 > "$LOGS/qa-head.log" 2>&1

log "ppl head-adaptive"
mkconf /tmp/p2-ppl.conf benchmarks/data/ppl benchmarks/results/exp-ppl-head "mode = balanced
budget = 0.5
head_adaptive = true
head_top_k = 128
head_profile = $PROFILE"
$BIN bench --config /tmp/p2-ppl.conf --repeats 2 > "$LOGS/ppl-head.log" 2>&1

log "PHASE2 DONE"
