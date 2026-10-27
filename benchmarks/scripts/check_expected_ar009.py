#!/usr/bin/env python3
"""AR009 FR-3 gate: verify the multi-fact coverage ablation against expected.

Reads `<dir>/ablation.csv` (written by ar009_ablation.py) and
`benchmarks/matrix/multifact/expected_ar009.csv`, then checks every expected
row:

  * the cell exists,
  * `q_mean >= q_floor` (the SRS floor, not the measured value),
  * `covered_queries >= covered_floor`,
  * the AR009 cell beats the AR008 cell (coverage_weight=0) by at least
    `min_delta`.

A missing cell is a failure, never skipped: the gate is meant to catch a
silently disabled feature (which is exactly how the first two implementation
rounds failed — every number identical to AR008).

Usage:
  check_expected_ar009.py benchmarks/results/ar009-multifact \
      benchmarks/matrix/multifact/expected_ar009.csv
"""
import csv
import os
import sys


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("usage: check_expected_ar009.py <results-dir> <expected.csv>\n")
        return 2
    results_dir, expected_path = argv[1], argv[2]
    ablation = os.path.join(results_dir, "ablation.csv")
    if not os.path.isfile(ablation):
        sys.stderr.write(f"missing {ablation}\n")
        return 1

    measured = {}
    for row in read_csv(ablation):
        measured[(row["dataset"], float(row["coverage_weight"]))] = row

    expected = read_csv(expected_path)
    if not expected:
        sys.stderr.write(f"empty {expected_path}\n")
        return 1

    failures = []
    for spec in expected:
        dataset = spec["dataset"]
        on = measured.get((dataset, 3.0))
        off = measured.get((dataset, 0.0))
        if on is None or off is None:
            failures.append(f"{dataset}: missing cell (on={on is not None}, "
                            f"off={off is not None})")
            continue
        q_on = float(on["q_mean"])
        q_off = float(off["q_mean"])
        covered = float(on["covered_queries"])
        q_floor = float(spec["q_floor"])
        covered_floor = float(spec["covered_floor"])
        min_delta = float(spec["min_delta"])
        status = []
        if q_on < q_floor:
            status.append(f"q_mean {q_on:.3f} < floor {q_floor:.3f}")
        if covered < covered_floor:
            status.append(f"covered {covered:.2f} < floor {covered_floor:.2f}")
        if q_on - q_off < min_delta:
            status.append(f"delta {q_on - q_off:+.3f} < required {min_delta:.3f}")
        if status:
            failures.append(f"{dataset}: " + "; ".join(status))
        else:
            print(f"PASS {dataset:<28} q {q_off:.3f} -> {q_on:.3f} "
                  f"({q_on - q_off:+.3f}), covered {covered:.2f}")

    if failures:
        for f in failures:
            sys.stderr.write(f"FAIL {f}\n")
        return 1
    print(f"all {len(expected)} AR009 cells within expected bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
