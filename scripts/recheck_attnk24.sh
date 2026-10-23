#!/bin/bash
# Re-run the exact exp-attnk-24 experiment (8K, budget 0.5, attn 24)
# with the fixed binary to see if q=0.44 recovers.
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

cat > /tmp/exp-attnk24-recheck.conf <<EOF
model = $MODEL
data_dir = /tmp/exp1
results_dir = benchmarks/results/exp-attnk-24-fixed
max_new_tokens = 16
warmup = 1
mode = balanced
budget = 0.5
attn_top_k = 24
EOF

timeout 1200 $BIN bench --config /tmp/exp-attnk24-recheck.conf --repeats 2 > /tmp/attnk24-recheck.log 2>&1
echo "exit=$?"
tail -5 /tmp/attnk24-recheck.log
echo "=== quality CSV ==="
cat benchmarks/results/exp-attnk-24-fixed/*.csv 2>/dev/null | head -8
