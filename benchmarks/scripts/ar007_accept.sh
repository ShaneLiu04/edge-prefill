#!/usr/bin/env bash
# AR007 acceptance campaign: dense reference, auto-budget (curve) and the
# AR006 fixed budget, measured back to back per dataset in one session.
#
#   benchmarks/scripts/ar007_accept.sh [--bin PATH] [--model GGUF] \
#       [--out DIR] [--datasets explen/len8k,explen/len16k,deep/len32k,deep/multineedle]
#
# Grid rules in the bench loop: mode=dense needs budget=1.0, sparse modes need
# budget<0.95. The auto conf therefore pins NO budget and lets the planner read
# the curve (the 0-sentinel path in main.cpp).
#
# Dense is the expensive part (32K is ~6 min/case on a 16-thread Xeon); reruns
# skip any directory that already has samples, so an interrupted campaign can
# be resumed.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

BIN=build-wsl/edge-prefill
MODEL=${EDGE_PREFILL_TEST_MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
OUT=benchmarks/results/ar007-campaign
CURVE=benchmarks/results/ar007-sweep/budget_curve.txt
DATASETS=explen/len8k,explen/len16k,deep/len32k,deep/multineedle

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
if [[ ! -f "$CURVE" ]]; then
  echo "missing curve $CURVE (run: ./repro.sh budget)" >&2
  exit 1
fi

mk_dense() {
  cat > /tmp/ar007_dense.conf <<EOF
mode = dense
budget = 1.0
attn_top_k = 24
pool_branches = false
query_gating = false
qg_bound = false
embedding_model =
anchor_ratio = 0.35
threads = 16
warmup = 0
max_new_tokens = 16
repeats = 1
EOF
}

mk_auto() {
  cat > /tmp/ar007_auto.conf <<EOF
mode = balanced
attn_top_k = 24
pool_branches = false
query_gating = false
qg_bound = false
embedding_model =
anchor_ratio = 0.35
budget_curve = $CURVE
threads = 16
warmup = 0
max_new_tokens = 16
repeats = 1
EOF
}

mk_base() {
  cat > /tmp/ar007_base.conf <<EOF
mode = balanced
budget = 0.25
attn_top_k = 24
pool_branches = false
query_gating = false
qg_bound = false
embedding_model =
anchor_ratio = 0.35
threads = 16
warmup = 0
max_new_tokens = 16
repeats = 1
EOF
}

run() { # dataset conf tag
  local dir="$OUT/$(basename "$1")-$3"
  mkdir -p "$dir"
  if ls "$dir"/*_raw.csv >/dev/null 2>&1; then echo "[skip] $1 $3"; return 0; fi
  echo "[run ] $1 $3 $(date +%H:%M:%S)"
  # The trap ledger: main.cpp also looks for a curve path relative to cwd, and
  # the bench rewrites bench.conf into the out dir for the record.
  cp "$2" "$dir/bench.conf"
  "$BIN" bench --config "$dir/bench.conf" --model "$MODEL" \
      --data-dir "benchmarks/data/$1" --out "$dir" --bench-skip-dense \
      > "/tmp/ar007_runner_$3.log" 2>&1
  local rc=$?
  local n
  n=$(grep -o 'samples=[0-9]*' "/tmp/ar007_runner_$3.log" | tail -1)
  echo "[done] $1 $3 rc=$rc $n $(date +%H:%M:%S)"
  if [ "$n" = "samples=0" ]; then
    echo "    !! no samples, runner log kept at /tmp/ar007_runner_$3.log"
    return 1
  fi
  for f in "$dir"/*_raw.csv; do
    [ -f "$f" ] || continue
    python3 - "$f" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"    {r['case']:<18} ttft={float(r['ttft_ms']):>8.0f} q={r['quality']:<5} "
          f"sel={r['selected_tokens']:>5} budget={r['budget']:<6} src={r['budget_source']:<9} "
          f"anch={r['anchor_blocks']:>3} mmr={r['mmr_penalized']}")
PY
  done
}

for ds in ${DATASETS//,/ }; do
  mk_dense; run "$ds" /tmp/ar007_dense.conf dense || exit 1
  mk_auto;  run "$ds" /tmp/ar007_auto.conf auto || exit 1
  mk_base;  run "$ds" /tmp/ar007_base.conf ar006-budget || exit 1
done

python3 benchmarks/scripts/ar007_report.py "$OUT"
echo "[acceptance done] $(date +%H:%M:%S)"
