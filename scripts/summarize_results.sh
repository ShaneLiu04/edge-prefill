#!/bin/bash
# Interim summary of every experiment CSV: one line per mode row.
cd /mnt/d/prefill/benchmarks/results
for d in exp-len-len2k-sel exp-len-len4k-sel exp-len-len8k-sel \
         exp-len-len16k-sel exp-len-len16k-attn \
         exp-budget-0.25 exp-budget-0.4 exp-budget-0.7 \
         exp-attnk-24 exp-attnk-48 exp-attnk-96 exp-latency \
         exp-qa exp-ppl; do
  f=$(ls $d/*_raw.csv 2>/dev/null | tail -1)
  [ -z "$f" ] && continue
  awk -F, -v d="$d" 'NR>1 {
    printf "%-22s %-12s ttft=%7.1fs tps=%5.0f q=%.2f tok=%5d sel=%5d k=%d\n", \
      d, $2, $3/1000, $5, $7, $8, $9, $13
  }' "$f"
done
