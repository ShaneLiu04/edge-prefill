#!/usr/bin/env python3
"""AR012: derive and gate the sentence-complete admission routing.

Arms measured by ar012_matrix.py:

  ar011        baseline (structure budget + single-fact span)
  ar012        sentence completion (multi-fact only, union mode)
  ar012-trim   completion + drop tokens outside the admitted sentences
  gate-p16     AR006 in-graph page gating

Routing rule: the arm with the highest (q_mean, q_min, -selected) among the arms
that do not regress against AR011 on quality. Ties go to ar012 (the shipped
default).

Gate (`--gate`) - "no free cost":

  * q_mean >= ar011 - 0.01 and q_min >= ar011 - 0.01 for every dataset;
  * if the routed arm does not beat AR011 by more than 0.01 on q_mean, it must
    not admit more tokens and must stay within 1.05x its TTFT;
  * if it does beat AR011, it may spend up to 1.25x the tokens (and TTFT) -
    that is the price of admitting complete answers, and it is reported.

`--inject ds:arm` forces a worse arm to prove the gate fails.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

ARMS = ("ar012-trim", "ar012", "ar011", "gate-p16")
# Routable arms. ar012-trim is measured but never routed: dropping the tokens
# outside the admitted sentences measured q 1.000 -> 0.827 on 4K and 32K (it
# removes sentences the model still needs), even though it wins on two cells.
ROUTABLE = ("ar012", "ar011", "gate-p16")


def canonical(name):
    table = {
        "len4k": "explen/len4k", "len8k": "explen/len8k", "len16k": "explen/len16k",
        "len32k": "deep/len32k", "multineedle": "deep/multineedle",
        "multifact8k": "deep/multifact8k",
        "multifact8k-hard": "deep/multifact8k-hard",
    }
    return table.get(name, name)


def read_cells(root):
    cells = []
    for path in sorted(glob.glob(os.path.join(root, "*-*"))):
        if not os.path.isdir(path):
            continue
        rows = []
        for f in glob.glob(os.path.join(path, "*_raw.csv")):
            with open(f, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        rows = [r for r in rows if int(r["dense"]) == 0]
        if not rows:
            continue
        name = os.path.basename(path)
        arm = None
        for cand in ARMS:
            if name.endswith("-" + cand):
                arm, dataset = cand, name[: -len(cand) - 1]
                break
        if arm is None:
            continue
        q = [float(r["quality"]) for r in rows]
        cells.append({
            "dataset": canonical(dataset), "arm": arm, "cases": len(rows),
            "q_mean": statistics.mean(q), "q_min": min(q),
            "ttft_median": statistics.median(float(r["ttft_ms"]) for r in rows),
            "selected": int(statistics.mean(int(r["selected_tokens"]) for r in rows)),
            "sentences": statistics.mean(float(r.get("sentences_completed", 0)) for r in rows),
            "extended": statistics.mean(float(r.get("sentences_extended", 0)) for r in rows),
        })
    return cells


def derive(cells):
    table, trail = {}, {}
    for dv in sorted({c["dataset"] for c in cells}):
        arms = [c for c in cells if c["dataset"] == dv]
        base = next((c for c in arms if c["arm"] == "ar011"), None)
        if base is None:
            continue
        ok = [c for c in arms if c["arm"] in ROUTABLE
              and c["q_mean"] >= base["q_mean"] - 0.01
              and c["q_min"] >= base["q_min"] - 0.01]
        best = max(ok, key=lambda c: (c["q_mean"], c["q_min"], -c["selected"]))
        table[dv] = best["arm"]
        trail[dv] = {"arm": best["arm"], "q_mean": round(best["q_mean"], 4),
                     "q_min": round(best["q_min"], 4), "selected": best["selected"],
                     "ttft_median": round(best["ttft_median"], 1),
                     "vs_base": round(best["q_mean"] - base["q_mean"], 4)}
    return table, trail


def gate(cells, table, inject):
    failures = []
    if inject:
        ds, arm = inject.split(":", 1)
        table = dict(table)
        table[ds] = arm
        print(f"[inject] {ds} -> {arm}")
    for dv, arm in sorted(table.items()):
        arms = [c for c in cells if c["dataset"] == dv]
        routed = next((c for c in arms if c["arm"] == arm), None)
        base = next((c for c in arms if c["arm"] == "ar011"), None)
        if routed is None or base is None:
            failures.append(f"{dv}: missing cell for {arm} or ar011")
            continue
        gain = routed["q_mean"] - base["q_mean"]
        problems = []
        if routed["q_mean"] < base["q_mean"] - 0.01:
            problems.append(f"q_mean {routed['q_mean']:.3f} < ar011 "
                            f"{base['q_mean']:.3f} -0.01")
        if routed["q_min"] < base["q_min"] - 0.01:
            problems.append(f"q_min {routed['q_min']:.3f} < ar011 "
                            f"{base['q_min']:.3f} -0.01")
        token_cap = 1.25 if gain > 0.01 + 1e-9 else 1.05
        if routed["selected"] > base["selected"] * token_cap:
            problems.append(f"selected {routed['selected']} > {token_cap:.2f} x "
                            f"ar011 {base['selected']} (gain {gain:+.3f})")
        if base["ttft_median"] and routed["ttft_median"] > base["ttft_median"] * token_cap:
            problems.append(f"ttft {routed['ttft_median']:.0f} > {token_cap:.2f} x "
                            f"{base['ttft_median']:.0f}")
        if problems:
            failures.append(f"{dv}: " + "; ".join(problems))
        else:
            print(f"PASS {dv:<24} routed={arm:<10} q={routed['q_mean']:.3f} "
                  f"qmin={routed['q_min']:.3f} sel={routed['selected']:5d} "
                  f"(gain {gain:+.3f})")
    if failures:
        for f in failures:
            sys.stderr.write(f"FAIL {f}\n")
        return 1
    print(f"all {len(table)} routed datasets pass the no-free-cost gate")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--inject", default="")
    args = ap.parse_args(argv)
    cells = read_cells(args.dir)
    if not cells:
        sys.stderr.write(f"no measured cells under {args.dir}\n")
        return 2
    table, trail = derive(cells)
    if args.gate:
        return gate(cells, table, args.inject)

    with open(os.path.join(args.dir, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median",
                  "selected", "sentences", "extended"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k in fields})
    with open(os.path.join(args.dir, "portfolio.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("default = ar012\n")
        for dv, arm in sorted(table.items()):
            f.write(f"{dv} = {arm}\n")
    with open(os.path.join(args.dir, "portfolio.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"table": trail, "cells": cells}, f, indent=2)
        f.write("\n")
    with open(os.path.join(args.dir, "matrix.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# AR012 sentence-complete admission matrix\n\n")
        for dv in sorted({c["dataset"] for c in cells}):
            f.write(f"## {dv}  (routed: {table.get(dv, '-')})\n\n")
            f.write("| arm | q_mean | q_min | selected | sentences | ext | TTFT ms |\n"
                    "|---|---:|---:|---:|---:|---:|---:|\n")
            for c in sorted([c for c in cells if c["dataset"] == dv],
                            key=lambda c: c["selected"]):
                f.write(f"| {c['arm']} | {c['q_mean']:.3f} | {c['q_min']:.3f} | "
                        f"{c['selected']} | {c['sentences']:.1f} | {c['extended']:.1f} | "
                        f"{c['ttft_median']:.0f} |\n")
            f.write("\n")
    print(f"wrote {args.dir}/matrix.csv, portfolio.txt, matrix.md")
    for dv, arm in sorted(table.items()):
        t = trail[dv]
        print(f"  {dv:<24} -> {arm:<10} q={t['q_mean']:.3f}/{t['q_min']:.3f} "
              f"sel={t['selected']} ({t['vs_base']:+.3f} vs ar011)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
