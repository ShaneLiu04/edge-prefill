#!/bin/bash
# Numerical-equivalence test: allow ALL blocks through the gather (k=200)
# -> output must exactly equal selection-only.
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

CTX=/tmp/exp1/contexts/niah-01.txt
Q="What is the secret passcode for vault 1?"
cat "$CTX" > /tmp/eq-prompt.txt
echo "" >> /tmp/eq-prompt.txt
echo "$Q" >> /tmp/eq-prompt.txt

cat > /tmp/eq.conf <<EOF
model = $MODEL
max_new_tokens = 16
threads = 16
EOF

echo "=== selection-only ==="
timeout 600 $BIN run --config /tmp/eq.conf --prompt-file /tmp/eq-prompt.txt > /tmp/eq-sel.log 2>&1
echo "exit=$?"
grep -a "generated" /tmp/eq-sel.log | head -1
grep -a "\[edge-prefill\] prompt" /tmp/eq-sel.log

echo "=== all-blocks gather (attn-top-k 200) ==="
timeout 600 $BIN run --config /tmp/eq.conf --prompt-file /tmp/eq-prompt.txt --attn-top-k 200 --verbose > /tmp/eq-full.log 2>&1
echo "exit=$?"
grep -a "attn-top-k-blocks" /tmp/eq-full.log | head -1
grep -a "generated" /tmp/eq-full.log | head -1
grep -a "\[edge-prefill\] prompt" /tmp/eq-full.log
