#!/usr/bin/env python3
"""Compare an assembled matrix.csv against the expected-results table.

`benchmarks/matrix/expected.csv` carries one row per (dataset, conf) with
the acceptance floors measured on the reference machine:

    dataset,conf,q_min,speedup_min,note

- q_min: hard floor on the mean quality of the grid point.
- speedup_min: hard floor on the *net* speedup (dense / (sparse + scoring
  ms)). Floors are set ~15% below the measured reference so host noise
  does not fail the check, while a real regression does.
- Rows present in the matrix but absent from the table are reported as
  warnings (new cells need a recorded expectation before they gate).

Exit code is non-zero when any floor is violated, so `repro.sh` fails
loudly instead of silently drifting.
"""

from __future__ import annotations

import argparse
import csv
import sys


def read_table(path):
    with open(path, newline="") as f:
        return {(r["dataset"], r["conf"]): r for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--datasets", default="",
                    help="comma-separated dataset filter")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    expected = read_table(args.expected)
    with open(args.matrix, newline="") as f:
        rows = list(csv.DictReader(f))

    checked = 0
    failures = []
    warnings = []
    for row in rows:
        dataset = row["dataset"]
        if wanted and dataset not in wanted:
            continue
        key = (dataset, row["conf"])
        want = expected.get(key)
        if want is None:
            warnings.append("no expectation for %s/%s" % key)
            continue
        checked += 1
        q = float(row["quality_mean"])
        speed = float(row["speedup_net_mean"])
        q_min = float(want["q_min"])
        s_min = float(want["speedup_min"])
        ok = True
        if q + 1e-9 < q_min:
            ok = False
            failures.append("%s/%s: quality %.3f < floor %.3f"
                            % (dataset, row["conf"], q, q_min))
        if speed + 1e-9 < s_min:
            ok = False
            failures.append("%s/%s: net speedup %.2fx < floor %.2fx"
                            % (dataset, row["conf"], speed, s_min))
        if args.verbose or not ok:
            print("[%s] %s/%s q=%.3f (>=%.3f) speedup=%.2fx (>=%.2fx)"
                  % ("ok" if ok else "FAIL", dataset, row["conf"], q, q_min,
                     speed, s_min))

    for w in warnings:
        print("[warn] " + w)
    print("checked %d grid points against %s" % (checked, args.expected))
    if failures:
        for f in failures:
            print("[fail] " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
