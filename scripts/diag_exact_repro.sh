#!/bin/bash
# Exact reproduction of the original 4K k62 repro (question included),
# old binary vs new binary, multi-chunk default config.
CTX=/tmp/explen/len4k/contexts/niah-01.txt
Q="What is the secret passcode for vault 1?"
cat "$CTX" > /tmp/diag-prompt-q.txt
echo "" >> /tmp/diag-prompt-q.txt
echo "$Q" >> /tmp/diag-prompt-q.txt
wc -c /tmp/diag-prompt-q.txt

cd /mnt/d/prefill
cat > /tmp/diag-default.conf <<EOF
model = /mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
max_new_tokens = 24
threads = 16
EOF

echo "=== NEW binary, k62, multi-chunk default ==="
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-default.conf --prompt-file /tmp/diag-prompt-q.txt --attn-top-k 62 --verbose > /tmp/diag-k62q-new.log 2>&1
echo "exit=$?"
grep -a "generated\|spider" /tmp/diag-k62q-new.log | head -2
grep -a "\[edge-prefill\] prompt" /tmp/diag-k62q-new.log

echo "=== NEW binary, selection-only, multi-chunk default ==="
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-default.conf --prompt-file /tmp/diag-prompt-q.txt --verbose > /tmp/diag-selq-new.log 2>&1
echo "exit=$?"
grep -a "generated" /tmp/diag-selq-new.log | head -2
grep -a "\[edge-prefill\] prompt" /tmp/diag-selq-new.log

echo "=== NEW binary, attn-24 (the q=0.44 config), multi-chunk ==="
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-default.conf --prompt-file /tmp/diag-prompt-q.txt --attn-top-k 24 --verbose > /tmp/diag-k24q-new.log 2>&1
echo "exit=$?"
grep -a "generated" /tmp/diag-k24q-new.log | head -2
grep -a "\[edge-prefill\] prompt" /tmp/diag-k24q-new.log
