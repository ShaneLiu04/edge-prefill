#!/bin/bash
LOG=/tmp/diag-sparse.log
echo "=== graph #B input tensors (lines 2060-2131) ==="
sed -n '2060,2131p' "$LOG" | grep -a "input tensor" | sed "s/.*input tensor //" | sort -u
echo ""
echo "=== graph #A input tensors (lines 1760-1920) ==="
sed -n '1760,1918p' "$LOG" | grep -a "input tensor" | sed "s/.*input tensor //" | sort -u
