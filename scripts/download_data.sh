#!/bin/bash
# AR004 FR-7 / T014: regenerate the benchmark datasets (niah / qa / ppl).
# The generated contexts and index files are deterministic given the
# pinned seed in gen_*.py; data_sha256 is recorded per benchmark run.
set -eu
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" > /dev/null 2>&1; then
  echo "error: python3 not found" >&2
  exit 2
fi

for gen in gen_niah.py gen_qa.py gen_ppl.py; do
  echo "== $gen"
  "$PYTHON" "benchmarks/scripts/$gen"
done

# Deep-dive variants (32K long context, multi-needle) used by the AR004
# experiment matrix; safe to re-run, output is deterministic.
bash scripts/make_deep_datasets.sh

echo
echo "data SHA256 (record this in your report env):"
find benchmarks/data -type f \( -name '*.txt' -o -name '*.tsv' \) -print0 \
  | sort -z | xargs -0 cat | sha256sum
