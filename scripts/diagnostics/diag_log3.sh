#!/bin/bash
ls -la /tmp/diag*.log
for f in /tmp/diag*.log; do
  echo "== $f"
  grep -a "attn_top_k\|sparsity\|edge_prefill\|EdgePrefill\|top_k" "$f" | head -4
done
