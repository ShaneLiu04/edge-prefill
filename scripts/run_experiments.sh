#!/bin/bash
# EdgePrefill experiment suite (CPU, Qwen2.5-1.5B q4_k_m, 16 cores).
# Runs the full interpretability + performance matrix sequentially so
# timings never contend; every section logs to /tmp/explogs.
set -u
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
LOGS=/tmp/explogs
mkdir -p "$LOGS"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/progress.log"; }

# ---------- 0. data prep: length-scaling contexts ----------
log "data prep"
mkdir -p /tmp/explen/len2k/contexts /tmp/explen/len4k/contexts /tmp/explen/len8k/contexts /tmp/explen/len16k/contexts
NI=/mnt/d/prefill/benchmarks/data/niah
head -c 13000 $NI/contexts/niah-01.txt > /tmp/explen/len2k/contexts/niah-01.txt
head -c 26000 $NI/contexts/niah-01.txt > /tmp/explen/len4k/contexts/niah-01.txt
cp $NI/contexts/niah-01.txt /tmp/explen/len8k/contexts/niah-01.txt
cat $NI/contexts/niah-01.txt $NI/contexts/niah-02.txt > /tmp/explen/len16k/contexts/niah-01.txt
for d in len2k len4k len8k len16k; do
  head -2 $NI/index.tsv > /tmp/explen/$d/index.tsv
done
wc -c /tmp/explen/*/contexts/niah-01.txt >> "$LOGS/progress.log" 2>&1

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

# ---------- 1. head classification (interpretability) ----------
log "classify-heads start"
$BIN classify-heads --model $MODEL --output benchmarks/results/exp-heads/profile.json --threads 16 > "$LOGS/heads.log" 2>&1
log "classify-heads done: $(tail -1 $LOGS/heads.log)"

# ---------- 2. length sweep: selection-only vs selection+attn ----------
for L in len2k len4k len8k len16k; do
  for V in sel attn; do
    log "length sweep $L $V"
    if [ "$V" = sel ]; then
      EXTRA="mode = balanced
budget = 0.5"
    else
      EXTRA="mode = balanced
budget = 0.5
attn_top_k = 24"
    fi
    mkconf /tmp/exp-$L-$V.conf /tmp/explen/$L benchmarks/results/exp-len-$L-$V "$EXTRA"
    $BIN bench --config /tmp/exp-$L-$V.conf --repeats 2 > "$LOGS/len-$L-$V.log" 2>&1
  done
done

# ---------- 3. budget sweep (8K) ----------
for B in 0.25 0.4 0.7; do
  log "budget sweep $B"
  mkconf /tmp/exp-budget.conf /tmp/exp1 benchmarks/results/exp-budget-$B "mode = balanced
budget = $B"
  $BIN bench --config /tmp/exp-budget.conf --repeats 2 > "$LOGS/budget-$B.log" 2>&1
done

# ---------- 4. attention top-k sweep (8K, budget 0.5) ----------
for K in 24 48 96; do
  log "attn-k sweep $K"
  mkconf /tmp/exp-attnk.conf /tmp/exp1 benchmarks/results/exp-attnk-$K "mode = balanced
budget = 0.5
attn_top_k = $K"
  $BIN bench --config /tmp/exp-attnk.conf --repeats 2 > "$LOGS/attnk-$K.log" 2>&1
done

# ---------- 5. latency mode (auto FFN skip, budget 0.4) ----------
log "latency mode"
mkconf /tmp/exp-latency.conf /tmp/exp1 benchmarks/results/exp-latency "mode = latency"
$BIN bench --config /tmp/exp-latency.conf --repeats 2 > "$LOGS/latency.log" 2>&1

# ---------- 6. QA + PPL quality breadth ----------
log "qa run"
mkconf /tmp/exp-qa.conf benchmarks/data/qa2 benchmarks/results/exp-qa "mode = balanced
budget = 0.5
attn_top_k = 48"
$BIN bench --config /tmp/exp-qa.conf --repeats 2 > "$LOGS/qa.log" 2>&1

log "ppl run"
mkconf /tmp/exp-ppl.conf benchmarks/data/ppl benchmarks/results/exp-ppl "mode = balanced
budget = 0.5
attn_top_k = 48"
$BIN bench --config /tmp/exp-ppl.conf --repeats 2 > "$LOGS/ppl.log" 2>&1

# ---------- 7. selection dumps (interpretability) ----------
log "selection dumps"
mkdir -p benchmarks/results/exp-sel-dump benchmarks/results/exp-sel-dump-attn
mkconf /tmp/exp-dump1.conf /tmp/explen/len8k benchmarks/results/exp-dump "mode = balanced
budget = 0.5"
EDGE_PREFILL_DUMP_SELECTION=benchmarks/results/exp-sel-dump \
  $BIN bench --config /tmp/exp-dump1.conf --repeats 1 > "$LOGS/dump.log" 2>&1
mkconf /tmp/exp-dump2.conf /tmp/explen/len8k benchmarks/results/exp-dump-attn "mode = balanced
budget = 0.5
attn_top_k = 48"
EDGE_PREFILL_DUMP_SELECTION=benchmarks/results/exp-sel-dump-attn \
  $BIN bench --config /tmp/exp-dump2.conf --repeats 1 > "$LOGS/dump2.log" 2>&1

log "ALL DONE"
