#!/bin/bash
nvcc --version | tail -2
echo "=== GPU baseline: n_prompt, avg_ts ==="
python3 - <<'EOF'
import csv
with open('/mnt/d/prefill/benchmarks/results/exp-gpu-dense/gpu_dense_1792696846.csv') as f:
    for row in csv.DictReader(filter(lambda l: not l.startswith('#'), f)):
        print(row['n_prompt'], row['flash_attn'], round(float(row['avg_ts']),1), 'tps ->', round(int(row['n_prompt'])/float(row['avg_ts'])*1000), 'ms')
EOF
