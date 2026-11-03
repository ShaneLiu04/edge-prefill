#!/usr/bin/env bash
# AR019: interleaved A/B of the static pipeline batch vs continuous batching.
# Reproduces benchmarks/results/ar019-continuous (walls.csv + o_<arm>_k<K>_r<R>).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1

M=${EDGE_PREFILL_MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
BIN=${EDGE_PREFILL_BIN:-build-cuda/edge-prefill}
DS=${1:-deep/multifact8k}
ROUNDS=${2:-3}
OUT=${EDGE_PREFILL_OUT:-benchmarks/results/ar019-continuous}
PLAN=${EDGE_PREFILL_PLAN:-benchmarks/results/ar016-exec-plan/_conf/explen-len4k-plan.conf}

mkdir -p "$OUT"
TMP=$(mktemp -d)

mk() { # path k extra...
  local path=$1 k=$2; shift 2
  grep -vE '^(snapshot_|decode_batch_|pipeline_|continuous_|warmup|repeats)' "$PLAN" > "$path"
  case "$DS" in
    deep/multifact8k|deep/multifact8k-hard) echo "budget = 0.05" >> "$path" ;;
  esac
  {
    echo "warmup = 1"
    echo "repeats = 1"
    echo "max_new_tokens = 48"
    echo "decode_batch_size = $k"
    echo "decode_batch_ctx = $((k * 3072))"
  } >> "$path"
  for line in "$@"; do echo "$line" >> "$path"; done
}

for k in 4 8; do
  mk "$OUT/pipe_k$k.conf" "$k" "n_gpu_layers = 0" "pipeline_mode = 1" \
     "pipeline_prefill_gpu_layers = 99" "pipeline_ahead = $k"
  mk "$OUT/cont_k$k.conf" "$k" "n_gpu_layers = 0" "pipeline_mode = 1" \
     "pipeline_prefill_gpu_layers = 99" "pipeline_ahead = $k" "continuous_batch = 1"
done

echo "arm,k,round,wall_ms" > "$OUT/ab_walls.csv"
for r in $(seq 1 "$ROUNDS"); do
  for k in 4 8; do
    for arm in pipe cont; do
      t0=$(date +%s%N)
      "$BIN" bench --config "$OUT/${arm}_k$k.conf" --model "$M" \
        --data-dir "benchmarks/data/$DS" --out "$OUT/o_${arm}_k${k}_r${r}" \
        --bench-skip-dense > "$TMP/l_${arm}_k${k}_r${r}.log" 2>&1
      rc=$?
      wall=$(( ($(date +%s%N) - t0) / 1000000 ))
      echo "$arm,$k,$r,$wall" >> "$OUT/ab_walls.csv"
      echo "round=$r k=$k arm=$arm wall=${wall}ms rc=$rc"
    done
  done
done
rm -rf "$TMP"
python3 benchmarks/scripts/ar019_report.py "$OUT" --gate
