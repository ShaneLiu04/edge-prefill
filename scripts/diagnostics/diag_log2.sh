#!/bin/bash
LOG=/tmp/diag-single.log
echo "=== edge_prefill banner / config ==="
grep -an "edge_prefill\|EdgePrefill\|attn\|top_k\|prefill_mode\|chunk" "$LOG" | grep -av "llama_graph\|inp_" | head -20
echo "=== command context: n_ubatch / n_batch ==="
grep -an "n_ubatch\|n_batch\|n_ctx\|compute buffer" "$LOG" | head -12
echo "=== main: decode progress ==="
grep -an "decode\|prefill tokens\|eval time" "$LOG" | head -10
