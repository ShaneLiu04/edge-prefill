#!/usr/bin/env bash
# AR017 focused A/B: decode cost per arm, interleaved in time so host drift hits
# every arm equally. Each round runs K=1,2,4,8 back to back; the summary takes
# the minimum per arm.
#
#   ./benchmarks/scripts/ar017_ab.sh [dataset] [rounds] [max-new-tokens]
#
# Env: EDGE_PREFILL_MODEL, EDGE_PREFILL_BIN, EDGE_PREFILL_PLAN, EDGE_PREFILL_OUT
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1

M=${EDGE_PREFILL_MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
BIN=${EDGE_PREFILL_BIN:-build/edge-prefill}
PLAN=${EDGE_PREFILL_PLAN:-benchmarks/results/ar016-exec-plan/_conf/explen-len4k-plan.conf}
DEST=${EDGE_PREFILL_OUT:-benchmarks/results/ar017-batched-decode}
DS=${1:-deep/multifact8k}
ROUNDS=${2:-3}
MNT=${3:-32}
OUT=$(mktemp -d)

for r in $(seq 1 "$ROUNDS"); do
    for k in 1 2 4 8; do
        conf=$OUT/c_${k}_${r}.conf
        grep -v '^snapshot_' "$PLAN" > "$conf"
        case "$DS" in
            deep/multifact8k|deep/multifact8k-hard) echo "budget = 0.05" >> "$conf" ;;
        esac
        {
            echo "max_new_tokens = $MNT"
            echo "repeats = 1"
            echo "warmup = 1"
            echo "decode_batch_size = $k"
            [ "$k" -gt 1 ] && echo "decode_batch_ctx = $((k * 1536))"
        } >> "$conf"
        rm -rf "$OUT/o_${k}_${r}"
        "$BIN" bench --config "$conf" --model "$M" \
            --data-dir "benchmarks/data/$DS" --out "$OUT/o_${k}_${r}" \
            --bench-skip-dense > "$OUT/l_${k}_${r}.log" 2>&1
        echo "round=$r k=$k rc=$?"
    done
done

mkdir -p "$DEST"
[ "${AB_APPEND:-0}" = "1" ] || rm -f "$DEST/ab.csv"
python3 benchmarks/scripts/ar017_ab_summary.py "$OUT" "$DS" "$DEST/ab.csv"
rm -rf "$OUT"
