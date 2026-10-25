#!/usr/bin/env python3
"""AR006: check a campaign directory against an expected table.

expected.csv columns:
    dataset,mode,min_quality,min_speedup
where mode is the result-dir suffix pair name used by assemble_ar006.py
(dense is the reference and is not gated). Speedup is dense_ttft/mode_ttft
measured in the same session, so the table is machine-independent up to
noise; min_speedup is a floor, never a target.

Usage: check_expected_ar006.py RESULTS_DIR EXPECTED_CSV
Exit 1 on any shortfall (CI gate).
"""
import csv
import glob
import os
import sys


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def load(path):
    csvs = sorted(glob.glob(os.path.join(path, "*_raw.csv")))
    if not csvs:
        return None
    with open(csvs[-1]) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {
        "ttft": median([float(r["ttft_ms"]) for r in rows]),
        "q": sum(float(r["quality"]) for r in rows) / len(rows),
        "min_q": min(float(r["quality"]) for r in rows),
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    results, expected = sys.argv[1], sys.argv[2]
    failures = []
    checked = 0
    with open(expected) as f:
        for exp in csv.DictReader(f):
            dataset, mode = exp["dataset"], exp["mode"]
            dense = load(os.path.join(results, dataset + "-dense"))
            got = load(os.path.join(results, f"{dataset}-{mode}"))
            if dense is None or got is None:
                failures.append(f"{dataset}/{mode}: missing raw csv")
                continue
            speedup = dense["ttft"] / got["ttft"] if got["ttft"] else 0.0
            min_q = float(exp["min_quality"])
            min_s = float(exp["min_speedup"])
            ok_q = got["q"] >= min_q
            ok_s = speedup >= min_s
            checked += 1
            print(f"{dataset:<18} {mode:<12} q={got['q']:.2f} (>= {min_q}) "
                  f"min={got['min_q']:.2f} speedup={speedup:.2f}x (>= {min_s}) "
                  f"{'ok' if ok_q and ok_s else 'FAIL'}")
            if not ok_q:
                failures.append(f"{dataset}/{mode}: quality {got['q']:.2f} < {min_q}")
            if not ok_s:
                failures.append(f"{dataset}/{mode}: speedup {speedup:.2f}x < {min_s}")
    print(f"\nchecked={checked} failures={len(failures)}")
    for failure in failures:
        print("  " + failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
