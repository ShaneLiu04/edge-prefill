#!/bin/bash
LOG=/tmp/diag-sparse.log
echo "=== self_attn_idx occurrences ==="
grep -a "self_attn_idx" "$LOG" | sort -u | head -4
echo "=== inp_pos sizes ==="
grep -a "inp_pos" "$LOG" | grep -o "ne = {[0-9 ]*" | sort | uniq -c
echo "=== inp_mixed ==="
grep -a "inp_mixed" "$LOG" | sort -u | head -6
echo "=== last 25 lines ==="
tail -25 "$LOG"
