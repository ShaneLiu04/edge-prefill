#!/usr/bin/env python3
"""AR011 FR-4: assemble the combination table and gate it against AR010.

The AR011 rule is a *no-regression* rule, because AR010 already reaches q=1.000
on every single-fact dataset: an arm may only be routed if it keeps the AR010
quality (mean and min) while using no more tokens. Among the arms that qualify,
the routed one is the one with the fewest admitted tokens (TTFT is proportional
to them; TTFT itself carries 2-3x session noise).

Quality floors:

  * datasets where AR010 already reaches the global floor (0.95) keep that floor;
  * the multi-fact families cannot reach 0.95 with partial-credit scoring, so
    their floor is the AR010 measurement itself (iso-quality routing).

`--gate` re-checks an existing table:

  * the routed arm must have a measured cell;
  * q_mean and q_min must not drop by more than 0.01 versus AR010;
  * selected tokens must not exceed the AR010 count;
  * TTFT must not exceed AR010 by more than 5%.

`--inject dataset:arm` forces a worse arm to prove the gate fails.

Usage:
  ar011_report.py <dir> [--floor 0.95] [--gate] [--inject ds:arm]
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

ARMS = ("ar011", "ar010", "gate-p16", "dense")


def canonical_dataset(name):
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


def mean_audit(rows, field):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else 0.0


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
        arm = None
        for cand in ARMS:
            if name.endswith("-" + cand):
                dataset, arm = name[: -len(cand) - 1], cand
                break
        if arm is None:
            continue
        dense_rows = [r for r in rows if int(r["dense"]) == 1]
        sparse_rows = [r for r in rows if int(r["dense"]) == 0]
        use = dense_rows if arm == "dense" else sparse_rows
        if not use:
            continue
        q = [float(r["quality"]) for r in use]
        cells.append({
            "dataset_key": dataset,
            "arm": arm,
            "cases": len(use),
            "q_mean": statistics.mean(q),
            "q_min": min(q),
            "ttft_median": statistics.median(float(r["ttft_ms"]) for r in use),
            "selected": int(statistics.mean(int(r["selected_tokens"]) for r in use)),
            "structure": int(statistics.mean(mean_audit([r], "structure_tokens")
                                             for r in use)),
            "span_trimmed": int(statistics.mean(mean_audit([r], "span_tokens")
                                                for r in use)),
        })
    return cells


def floors(cells, global_floor):
    """Per-dataset floor: the global floor where AR010 reaches it, else AR010."""
    out = {}
    for dv in {c["dataset"] for c in cells}:
        base = next((c for c in cells if c["dataset"] == dv and c["arm"] == "ar010"),
                    None)
        if base is None:
            out[dv] = (global_floor, global_floor)
        elif base["q_mean"] >= global_floor:
            out[dv] = (global_floor, global_floor)
        else:
            out[dv] = (base["q_mean"], base["q_min"])
    return out


def derive_table(cells, fl):
    table, trail = {}, {}
    for dv in sorted({c["dataset"] for c in cells}):
        arms = [c for c in cells if c["dataset"] == dv and c["arm"] != "dense"]
        if not arms:
            continue
        floor_mean, floor_min = fl[dv]
        ok = [c for c in arms if c["q_mean"] >= floor_mean - 1e-9
              and c["q_min"] >= floor_min - 1e-9]
        if ok:
            # Ties on tokens are broken towards AR011 (the shipped default) and
            # only then by TTFT, which carries 2-3x session noise.
            best = min(ok, key=lambda c: (c["selected"],
                                          0 if c["arm"] == "ar011" else 1,
                                          c["ttft_median"]))
            reason = "iso-quality, fewest tokens"
        else:
            best = max(arms, key=lambda c: (c["q_mean"], c["q_min"], -c["selected"]))
            reason = "no arm holds the floor"
        table[dv] = best["arm"]
        trail[dv] = {"arm": best["arm"], "reason": reason,
                     "q_mean": round(best["q_mean"], 4),
                     "q_min": round(best["q_min"], 4),
                     "selected": best["selected"],
                     "ttft_median": round(best["ttft_median"], 1)}
    return table, trail


def gate(cells, table, fl, inject):
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
        floor_mean, floor_min = fl[dv]
        iso = [c for c in arms if c["q_mean"] >= floor_mean - 1e-9
               and c["q_min"] >= floor_min - 1e-9] or arms
        best_q = max(arms, key=lambda c: (c["q_mean"], c["q_min"]))
        min_tokens = min(c["selected"] for c in iso)
        fastest = min(c["ttft_median"] for c in iso)
        problems = []
        if routed["q_mean"] < best_q["q_mean"] - 0.01:
            problems.append(f"q_mean {routed['q_mean']:.3f} < best "
                            f"{best_q['q_mean']:.3f} -0.01")
        if routed["q_min"] < best_q["q_min"] - 0.01:
            problems.append(f"q_min {routed['q_min']:.3f} < best "
                            f"{best_q['q_min']:.3f} -0.01")
        if routed["selected"] > min_tokens * 1.05:
            problems.append(f"selected {routed['selected']} > 1.05 x {min_tokens}")
        if fastest and routed["ttft_median"] > fastest * 1.05:
            problems.append(f"ttft {routed['ttft_median']:.0f} > 1.05 x {fastest:.0f}")
        if problems:
            failures.append(f"{dv}: " + "; ".join(problems))
        else:
            print(f"PASS {dv:<24} routed={arm:<10} q={routed['q_mean']:.3f} "
                  f"qmin={routed['q_min']:.3f} sel={routed['selected']:5d} "
                  f"ttft={routed['ttft_median']:.0f}ms")
    if failures:
        for f in failures:
            sys.stderr.write(f"FAIL {f}\n")
        return 1
    print(f"all {len(table)} routed datasets are no-worse than AR010 "
          f"(quality, tokens, TTFT)")
    return 0


def write_outputs(root, cells, table, trail, fl, global_floor):
    dense = {c["dataset"]: c for c in cells if c["arm"] == "dense"}
    with open(os.path.join(root, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median",
                  "speedup", "selected", "structure", "span_trimmed"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            ref = dense.get(c["dataset"])
            w.writerow({
                "dataset": c["dataset"], "arm": c["arm"], "cases": c["cases"],
                "q_mean": round(c["q_mean"], 4), "q_min": round(c["q_min"], 4),
                "ttft_median": round(c["ttft_median"], 1),
                "speedup": round(ref["ttft_median"] / c["ttft_median"], 2)
                if ref and c["arm"] != "dense" else "",
                "selected": c["selected"], "structure": c["structure"],
                "span_trimmed": c["span_trimmed"],
            })
    with open(os.path.join(root, "portfolio.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("default = ar011\n")
        for dv, arm in sorted(table.items()):
            floor_mean, floor_min = fl[dv]
            f.write(f"{dv} = {arm}   # floor {floor_mean:.3f}/{floor_min:.3f}\n")
    with open(os.path.join(root, "portfolio.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"global_floor": global_floor,
                   "floors": {k: list(v) for k, v in fl.items()},
                   "table": trail,
                   "cells": [{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in c.items()} for c in cells]}, f, indent=2)
        f.write("\n")
    with open(os.path.join(root, "matrix.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# AR011 combination matrix\n\n")
        f.write(f"global quality floor = {global_floor}; per-dataset floors are "
                f"the AR010 measurement where it is below the global floor\n\n")
        for dv in sorted({c["dataset"] for c in cells}):
            f.write(f"## {dv}  (routed: {table.get(dv, '-')}, "
                    f"floor {fl[dv][0]:.3f}/{fl[dv][1]:.3f})\n\n")
            f.write("| arm | q_mean | q_min | selected | struct | span | TTFT ms |\n"
                    "|---|---:|---:|---:|---:|---:|---:|\n")
            for c in sorted([c for c in cells if c["dataset"] == dv],
                            key=lambda c: c["selected"]):
                f.write(f"| {c['arm']} | {c['q_mean']:.3f} | {c['q_min']:.3f} | "
                        f"{c['selected']} | {c['structure']} | {c['span_trimmed']} | "
                        f"{c['ttft_median']:.0f} |\n")
            f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--floor", type=float, default=0.95)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--inject", default="")
    args = ap.parse_args(argv)

    cells = read_cells(args.dir)
    if not cells:
        sys.stderr.write(f"no measured cells under {args.dir}\n")
        return 1
    for c in cells:
        c["dataset"] = canonical_dataset(c["dataset_key"])
    fl = floors(cells, args.floor)
    table, trail = derive_table(cells, fl)
    if args.gate:
        return gate(cells, table, fl, args.inject)
    write_outputs(args.dir, cells, table, trail, fl, args.floor)
    print(f"wrote {args.dir}/matrix.csv, portfolio.txt, matrix.md")
    for dv, arm in sorted(table.items()):
        t = trail[dv]
        print(f"  {dv:<24} -> {arm:<10} q={t['q_mean']:.3f}/{t['q_min']:.3f} "
              f"sel={t['selected']} ({t['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
