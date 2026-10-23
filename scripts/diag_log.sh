#!/bin/bash
LOG=/tmp/diag-single.log
echo "=== inp_pos sizes (graph listings) ==="
grep -a "inp_pos" "$LOG" | grep -o "ne = {[0-9 ]*" | sort | uniq -c
echo "=== sched_reserve count ==="
grep -ac sched_reserve "$LOG"
echo "=== sparsity lines ==="
grep -an "attn_top_k\|attention sparsity\|sparsity" "$LOG" | head -5
echo "=== generated line ==="
grep -an "generated" "$LOG" | head -3
echo "=== first 30 lines ==="
head -30 "$LOG"
