#!/bin/bash
LOG=/tmp/diag-sparse.log
echo "=== ordered graph listings (line:ne of inp_pos) ==="
grep -an "inp_pos" "$LOG" | grep -o "^[0-9]*:.*ne = {[0-9 ]*" | awk -F"ne = {" '{print $1, $2}' | awk 'NR%28==1' | head -30
echo "=== sched_reserve lines with line numbers ==="
grep -an "sched_reserve: reserve\|graph: nodes" "$LOG" | head -20
echo "=== edge-prefill lines ==="
grep -an "edge-prefill" "$LOG"
echo "=== assert line numbers ==="
grep -an "GGML_ASSERT" "$LOG" | head -2
