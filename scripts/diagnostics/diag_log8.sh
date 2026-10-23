#!/bin/bash
LOG=/tmp/diag-inst2.log
echo "=== all get_n_kv lines ==="
grep -a "EdgePrefill get_n_kv" "$LOG"
echo "=== all edge-prefill/set_input/chunk lines ==="
grep -a "chunk \[\|EdgePrefill set_input\|sparsity" "$LOG"
echo "=== assert? ==="
grep -ac GGML_ASSERT "$LOG"
