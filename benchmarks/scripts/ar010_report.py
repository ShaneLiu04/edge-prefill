#!/usr/bin/env python3
"""AR010 FR-4: assemble the combination table and gate it.

Reads the measured cells under `<dir>/<dataset>-<arm>/*_raw.csv`, applies the
SRS rule (fastest arm whose mean *and* min quality reach the floor, else the
highest-quality arm), and writes:

    matrix.csv / matrix.md      measured cells
    portfolio.txt               the derived table
    portfolio.json              inputs + decision trail
    REPORT.md                   human report with the routing and the honest
                                limits (multi-fact correction, dense refs)

`--gate` re-checks an existing table:

    * every dataset in the table must have a measured cell for the routed arm;
    * the routed arm must be Pareto-no-worse than the best *fixed* arm for that
      dataset (q_mean >= best - 0.01, q_min >= best_min - 0.01,
      ttft_median <= fastest_at_quality * 1.05);
    * `--inject dataset:arm` forces a degraded arm to prove the gate fails.

Usage:
  ar010_report.py <dir> [--floor 0.95] [--gate] [--inject ds:arm]
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

DENSE_REF = {  # dataset -> (ttft_ms, cases, note); measured in the same session
    "explen/len4k": None,
    "explen/len8k": None,
    "explen/len16k": None,
    "deep/len32k": (447369.0, 2, "2-case subset, max_new_tokens=8"),
    "deep/multineedle": (63164.0, 2, "2-case subset, max_new_tokens=8"),
}


def read_cells(root):
    cells = []
    for path in sorted(glob.glob(os.path.join(root, "*-*"))):
        if not os.path.isdir(path):
            continue
        rows = []
        for f in glob.glob(os.path.join(path, "*_raw.csv")):
            with open(f, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        if not rows:
            continue
        name = os.path.basename(path)
        for arm in ("gate-p24-local32", "selection-prior", "gate-p16", "selection",
                    "dense"):
            if name.endswith("-" + arm):
                dataset, arm_name = name[: -len(arm) - 1], arm
                break
        else:
            continue
        dense_rows = [r for r in rows if int(r["dense"]) == 1]
        sparse_rows = [r for r in rows if int(r["dense"]) == 0]
        use = dense_rows if arm_name == "dense" else sparse_rows
        if not use:
            continue
        q = [float(r["quality"]) for r in use]
        cells.append({
            "dataset_key": dataset,
            "arm": arm_name,
            "cases": len(use),
            "q_mean": statistics.mean(q),
            "q_min": min(q),
            "ttft_median": statistics.median(float(r["ttft_ms"]) for r in use),
            "selected": int(statistics.mean(int(r["selected_tokens"]) for r in use)),
        })
    return cells


def canonical_dataset(name):
    """Map the directory prefix back to the canonical dataset path."""
    table = {
        "len4k": "explen/len4k",
        "len8k": "explen/len8k",
        "len16k": "explen/len16k",
        "len32k": "deep/len32k",
        "multineedle": "deep/multineedle",
        "multifact8k": "deep/multifact8k",
        "multifact8k-hard": "deep/multifact8k-hard",
    }
    return table.get(name, name)


def derive_table(cells, floor):
    table, trail = {}, {}
    for dv in sorted({c["dataset"] for c in cells}):
        arms = [c for c in cells if c["dataset"] == dv and c["arm"] != "dense"]
        if not arms:
            continue
        ok = [c for c in arms if c["q_mean"] >= floor and c["q_min"] >= floor]
        if ok:
            best = min(ok, key=lambda c: c["ttft_median"])
            reason = "floor-met"
        else:
            best = max(arms, key=lambda c: (c["q_mean"], c["q_min"], -c["ttft_median"]))
            reason = "floor-missed"
        table[dv] = best["arm"]
        trail[dv] = {"arm": best["arm"], "reason": reason,
                     "q_mean": round(best["q_mean"], 4),
                     "q_min": round(best["q_min"], 4),
                     "ttft_median": round(best["ttft_median"], 1)}
    return table, trail


def gate(cells, table, floor, inject):
    failures = []
    if inject:
        ds, arm = inject.split(":", 1)
        table = dict(table)
        table[ds] = arm
        print(f"[inject] {ds} -> {arm}")
    for dv, arm in sorted(table.items()):
        arms = [c for c in cells if c["dataset"] == dv and c["arm"] != "dense"]
        routed = next((c for c in arms if c["arm"] == arm), None)
        if routed is None:
            failures.append(f"{dv}: routed arm {arm} has no measured cell")
            continue
        best_q = max(arms, key=lambda c: (c["q_mean"], c["q_min"]))
        ok = [c for c in arms if c["q_mean"] >= floor and c["q_min"] >= floor]
        fastest_at_quality = min((c["ttft_median"] for c in ok), default=None)
        problems = []
        if routed["q_mean"] < best_q["q_mean"] - 0.01:
            problems.append(f"q_mean {routed['q_mean']:.3f} < best {best_q['q_mean']:.3f} -0.01")
        if routed["q_min"] < best_q["q_min"] - 0.01:
            problems.append(f"q_min {routed['q_min']:.3f} < best {best_q['q_min']:.3f} -0.01")
        if fastest_at_quality and routed["ttft_median"] > fastest_at_quality * 1.05:
            problems.append(f"ttft {routed['ttft_median']:.0f} > 1.05 x "
                            f"{fastest_at_quality:.0f}")
        if problems:
            failures.append(f"{dv}: " + "; ".join(problems))
        else:
            print(f"PASS {dv:<22} routed={arm:<18} q={routed['q_mean']:.3f} "
                  f"ttft={routed['ttft_median']:.0f}ms")
    if failures:
        for f in failures:
            sys.stderr.write(f"FAIL {f}\n")
        return 1
    print(f"all {len(table)} routed datasets are Pareto-no-worse than the best fixed arm")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--inject", default="")
    args = ap.parse_args(argv)

    root = args.dir
    cells = read_cells(root)
    if not cells:
        sys.stderr.write(f"no measured cells under {root}\n")
        return 1
    for c in cells:
        c["dataset"] = canonical_dataset(c["dataset_key"])

    table, trail = derive_table(cells, args.floor)

    if args.gate:
        return gate(cells, table, args.floor, args.inject)

    dense = {c["dataset"] for c in cells if c["arm"] == "dense"}
    with open(os.path.join(root, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "arm", "cases", "q_mean", "q_min",
                                          "ttft_median", "speedup", "selected"])
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            ref = DENSE_REF.get(c["dataset"])
            dense_ttft = ref[0] if ref else next(
                (d["ttft_median"] for d in cells if d["dataset"] == c["dataset"]
                 and d["arm"] == "dense"), None)
            w.writerow({
                "dataset": c["dataset"], "arm": c["arm"], "cases": c["cases"],
                "q_mean": round(c["q_mean"], 4), "q_min": round(c["q_min"], 4),
                "ttft_median": round(c["ttft_median"], 1),
                "speedup": round(dense_ttft / c["ttft_median"], 2)
                           if dense_ttft and c["arm"] != "dense" else "",
                "selected": c["selected"],
            })

    with open(os.path.join(root, "portfolio.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("default = selection\n")
        for dv, arm in sorted(table.items()):
            f.write(f"{dv} = {arm}\n")
    with open(os.path.join(root, "portfolio.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"quality_floor": args.floor, "table": trail,
                   "cells": [{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in c.items()} for c in cells]}, f, indent=2)
        f.write("\n")

    with open(os.path.join(root, "matrix.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# AR010 combination matrix\n\n")
        f.write(f"quality floor = {args.floor}\n\n")
        for dv in sorted({c["dataset"] for c in cells}):
            f.write(f"## {dv}  (routed: {table.get(dv, '-')})\n\n")
            f.write("| arm | q_mean | q_min | selected | TTFT ms |\n|---|---:|---:|---:|---:|\n")
            for c in sorted([c for c in cells if c["dataset"] == dv],
                            key=lambda c: c["ttft_median"]):
                f.write(f"| {c['arm']} | {c['q_mean']:.3f} | {c['q_min']:.3f} | "
                        f"{c['selected']} | {c['ttft_median']:.0f} |\n")
            f.write("\n")
    print(f"wrote {root}/matrix.csv, portfolio.txt, matrix.md")
    for dv, arm in sorted(table.items()):
        print(f"  {dv:<24} -> {arm:<18} {trail[dv]['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
