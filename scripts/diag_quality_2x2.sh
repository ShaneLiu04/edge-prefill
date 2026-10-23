#!/bin/bash
cd /mnt/d/prefill
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-dense.conf --prompt-file /tmp/diag-prompt.txt --attn-top-k 62 --verbose > /tmp/diag-fix1.log 2>&1
rc=$?
echo "sparse-k62-single exit=$rc"
grep -a "generated=\|\[edge-prefill\]" /tmp/diag-fix1.log | tail -3
grep -a "GGML_ASSERT" /tmp/diag-fix1.log | head -2

echo "=== control: selection-only (no attn gather), single-ubatch config ==="
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-dense.conf --prompt-file /tmp/diag-prompt.txt --verbose > /tmp/diag-sel-only.log 2>&1
rc=$?
echo "sel-only exit=$rc"
grep -a "generated=\|\[edge-prefill\]" /tmp/diag-sel-only.log | tail -3

echo "=== k62 with DEFAULT config (multi-chunk 512) ==="
cat > /tmp/diag-default.conf <<EOF
model = /mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
max_new_tokens = 8
threads = 16
EOF
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-default.conf --prompt-file /tmp/diag-prompt.txt --attn-top-k 62 --verbose > /tmp/diag-k62-mc.log 2>&1
rc=$?
echo "k62-multi exit=$rc"
grep -a "generated=\|\[edge-prefill\]" /tmp/diag-k62-mc.log | tail -3

echo "=== control: selection-only, DEFAULT config ==="
timeout 300 ./build-wsl/edge-prefill run --config /tmp/diag-default.conf --prompt-file /tmp/diag-prompt.txt --verbose > /tmp/diag-sel-mc.log 2>&1
rc=$?
echo "sel-multi exit=$rc"
grep -a "generated=\|\[edge-prefill\]" /tmp/diag-sel-mc.log | tail -3
