#!/bin/bash
# Decisive 2x2 test for the single-ubatch crash:
#   A) dense (no attn sparsity) + chunk 8192 / n_ubatch 4096
#   B) sparse (attn-top-k 62) + chunk 8192 / n_ubatch 4096  (known crash)
# Same 4K prompt as the earlier repro runs.
cd /mnt/d/prefill
BIN=./build-wsl/edge-prefill
MODEL=/mnt/d/edge-llm/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

CTX=/tmp/explen/len4k/contexts/niah-01.txt
Q=$(grep -o "The secret[^	]*" /mnt/d/prefill/benchmarks/data/niah/index.tsv | head -1)
echo "question: $Q"
PROMPT=$(cat "$CTX")
printf '%s\n%s\n' "$PROMPT" "$Q" > /tmp/diag-prompt.txt
wc -c /tmp/diag-prompt.txt

mkconf() {
  cat > "$1" <<EOF
model = $MODEL
chunk_size = 8192
n_ubatch = 4096
max_new_tokens = 8
threads = 16
EOF
}

mkconf /tmp/diag-dense.conf

echo "=== A) DENSE single-ubatch ==="
timeout 300 $BIN run --config /tmp/diag-dense.conf --prompt-file /tmp/diag-prompt.txt --verbose > /tmp/diag-dense.log 2>&1
echo "exit=$?"
grep -a "generated=\|GGML_ASSERT\|\[edge-prefill\]" /tmp/diag-dense.log | head -4

echo "=== B) SPARSE k=62 single-ubatch ==="
timeout 300 $BIN run --config /tmp/diag-dense.conf --prompt-file /tmp/diag-prompt.txt --attn-top-k 62 --verbose > /tmp/diag-sparse.log 2>&1
echo "exit=$?"
grep -a "generated=\|GGML_ASSERT\|\[edge-prefill\]" /tmp/diag-sparse.log | head -4
