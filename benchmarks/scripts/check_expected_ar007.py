#!/usr/bin/env python3
"""AR007 self-check: compare a campaign directory against expected.csv.

Absolute TTFT is host dependent, so the gate only checks the parts that must
be machine independent:

    quality          >= expected - --q-tolerance
    budget_source    exact (this is the AR007 audit contract)
    selected_tokens  within +-2% (selection is deterministic for a given prompt)

Usage:
  check_expected_ar007.py [campaign-dir] [expected.csv]

Exit code 0 when every row passes, 1 on any mismatch (the mismatch table is
printed to stderr).
"""
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEYS = ("quality", "budget_source", "selected_tokens")


def load_rows(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*_raw.csv")) +
                    glob.glob(os.path.join(path, "raw.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main(argv):
    camp = argv[1] if len(argv) > 1 else "benchmarks/results/ar007-campaign"
    exp_path = argv[2] if len(argv) > 2 else "benchmarks/matrix/budget/expected.csv"
    camp = camp if os.path.isabs(camp) else os.path.join(ROOT, camp)
    exp_path = exp_path if os.path.isabs(exp_path) else os.path.join(ROOT, exp_path)
    q_tol = 0.2
    for arg in argv[3:]:
        if arg.startswith("--q-tolerance="):
            q_tol = float(arg.split("=", 1)[1])

    expected = list(csv.DictReader(open(exp_path, newline="", encoding="utf-8")))
    if not expected:
        sys.stderr.write(f"empty expectation file {exp_path}\n")
        return 1

    cache = {}
    fails = []
    for exp in expected:
        dataset = exp["dataset"]
        config = f"{dataset}-{exp['config']}"
        # Directory convention: <dataset>-<config>, dataset being the last
        # path segment of the data dir (explen/len8k -> len8k).
        dirname = os.path.join(camp, config)
        if dirname not in cache:
            cache[dirname] = load_rows(dirname) if os.path.isdir(dirname) else []
        rows = [r for r in cache[dirname] if r["case"] == exp["case"]]
        if not rows:
            fails.append((config, exp["case"], "missing", "", ""))
            continue
        r = rows[0]
        if float(r["quality"]) + q_tol < float(exp["quality"]):
            fails.append((config, exp["case"], "quality",
                          f'{r["quality"]} < {exp["quality"]}-{q_tol}', ""))
        if r["budget_source"] != exp["budget_source"]:
            fails.append((config, exp["case"], "budget_source",
                          r["budget_source"], exp["budget_source"]))
        if exp.get("selected_tokens"):
            want = int(exp["selected_tokens"])
            got = int(r["selected_tokens"])
            if abs(got - want) > max(2, want // 50):
                fails.append((config, exp["case"], "selected_tokens",
                              str(got), str(want)))

    total = len(expected)
    if fails:
        sys.stderr.write(f"FAIL {len(fails)} mismatch(es) over {total} rows\n")
        for f in fails:
            sys.stderr.write(f"  {f[0]} {f[1]}: {f[2]} got={f[3]} want={f[4]}\n")
        return 1
    print(f"PASS {total}/{total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
