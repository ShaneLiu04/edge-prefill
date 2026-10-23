#!/bin/bash
LOG="$1"
N=$(grep -an GGML_ASSERT "$LOG" | head -1 | cut -d: -f1)
if [ -z "$N" ]; then echo "no assert found"; exit 1; fi
sed -n "${N},$((N+18))p" "$LOG"
