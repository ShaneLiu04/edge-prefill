#!/usr/bin/env python3
"""AR010 FR-4: measure the combination matrix after evidence-first admission.

Arms:

  selection      budget curve + evidence-gated admission + prior fill (AR008 semantics)
  selection-fit  same, but the admission stops when the evidence is exhausted
                 (content-sized context: the budget is a ceiling, not a quota)
  gate-p16       AR006 in-graph page gating, 16 pages
  gate-p24-local32  AR006 in-graph page gating, 24 pages + 32 local rows
  dense          full prefill (reference, measured once per dataset)

Outputs (same shapes as ar008-portfolio so the existing checks can read them):

  <out>/<dataset>-<arm>/{bench.conf,raw.csv}
  <out>/matrix.csv  dataset,arm,cases,q_mean,q_min,ttft_median,speedup,selected
  <out>/matrix.md
  <out>/portfolio.txt   derived table (rule: fastest arm at the quality floor)
  <out>/portfolio.json  inputs + decision trail
"""
import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = {
    "selection": {"mode": "balanced", "qg_bound": "false", "fill_priors": 0,
                  "prior_gate": 1, "budget": None},
    "selection-prior": {"mode": "balanced", "qg_bound": "false", "fill_priors": 1,
                        "prior_gate": 1, "budget": None},
    "gate-p16": {"mode": "balanced", "qg_bound": "true", "qg_sel_pages": 16,
                 "qg_local_rows": 0, "fill_priors": 0, "prior_gate": 1, "budget": 0.25},
    "gate-p24-local32": {"mode": "balanced", "qg_bound": "true", "qg_sel_pages": 24,
                         "qg_local_rows": 32, "fill_priors": 0, "prior_gate": 1,
                         "budget": 0.25},
    "dense": {"mode": "dense", "qg_bound": "false", "fill_priors": 1, "prior_gate": 1,
              "budget": 1.0},
}


def sha256_of(path):
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 else "unavailable"


def write_conf(path, arm, curve, threads, max_new_tokens, phrase_weight, coverage_weight,
               flat_budget=None):
    spec = ARMS[arm]
    lines = [f"mode = {spec['mode']}"]
    if flat_budget is not None:
        lines.append(f"budget = {flat_budget}")
    elif spec["budget"] is not None:
        lines.append(f"budget = {spec['budget']}")
    elif curve:
        lines.append(f"budget_curve = {curve}")
    lines += [
        f"attn_top_k = 24",
        "pool_branches = false",
        "query_gating = false",
        f"qg_bound = {spec['qg_bound']}",
    ]
    if "qg_sel_pages" in spec:
        lines.append(f"qg_sel_pages = {spec['qg_sel_pages']}")
        lines.append(f"qg_local_rows = {spec.get('qg_local_rows', 0)}")
    lines += [
        "embedding_model =",
        "anchor_ratio = 0.35",
        f"phrase_weight = {phrase_weight}",
        f"coverage_weight = {coverage_weight}",
        f"prior_gate = {spec['prior_gate']}",
        f"fill_priors = {spec['fill_priors']}",
        f"threads = {threads}",
        "warmup = 0",
        f"max_new_tokens = {max_new_tokens}",
        "repeats = 1",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def run_cell(bin_path, conf, model, data_dir, out_dir, skip_dense):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [bin_path, "bench", "--config", conf, "--model", model,
           "--data-dir", data_dir, "--out", out_dir]
    if skip_dense:
        cmd.append("--bench-skip-dense")
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - started
    if proc.returncode != 0:
        return None, f"rc={proc.returncode} wall={wall:.0f}s {proc.stderr[-200:]}"
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_raw.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def summarize(rows, dense=False):
    want = [r for r in rows if int(r["dense"]) == (1 if dense else 0)]
    if not want:
        return None
    q = sorted(float(r["quality"]) for r in want)
    t = [float(r["ttft_ms"]) for r in want]
    return {
        "cases": len(want),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "ttft_median": statistics.median(t),
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in want)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build-wsl/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k,explen/len16k,"
                                          "deep/len32k,deep/multineedle")
    ap.add_argument("--arms", default="selection,selection-prior,gate-p16,"
                                      "gate-p24-local32,dense")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar010-evidence-first")
    ap.add_argument("--quality-floor", type=float, default=0.95)
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--coverage-weight", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--flat-budget", type=float, default=None,
                    help="override the budget for every dataset (e.g. 0.05 for multi-fact)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    bin_path = args.bin if os.path.isabs(args.bin) else os.path.join(ROOT, args.bin)
    model = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    curve = args.curve if args.curve else ""
    os.makedirs(out, exist_ok=True)

    cells = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        name = dataset.rsplit("/", 1)[-1]
        data_dir = os.path.join(ROOT, "benchmarks/data", dataset)
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            tag = f"{name}-{arm}"
            cell_dir = os.path.join(out, tag)
            conf = os.path.join(cell_dir, "bench.conf")
            existing = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")))
            if existing and not args.force:
                rows = []
                for f in existing:
                    with open(f, newline="", encoding="utf-8") as fh:
                        rows.extend(csv.DictReader(fh))
            else:
                write_conf(conf, arm, os.path.join(ROOT, curve) if curve else "",
                           args.threads, args.max_new_tokens, args.phrase_weight,
                           args.coverage_weight, args.flat_budget)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir,
                                     skip_dense=(arm != "dense"))
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
            stats = summarize(rows, dense=(arm == "dense"))
            if stats is None:
                print(f"[fail] {tag}: no matching samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, "tag": tag, **stats})
            print(f"[cell] {tag:<34} q={stats['q_mean']:.3f} min={stats['q_min']:.3f} "
                  f"sel={stats['selected']:5d} ttft={stats['ttft_median']:8.0f}ms",
                  flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    dense = {c["dataset"]: c for c in cells if c["arm"] == "dense"}
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "arm", "cases", "q_mean", "q_min",
                                          "ttft_median", "speedup", "selected"])
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            d = dense.get(c["dataset"])
            w.writerow({
                "dataset": c["dataset"], "arm": c["arm"], "cases": c["cases"],
                "q_mean": round(c["q_mean"], 4), "q_min": round(c["q_min"], 4),
                "ttft_median": round(c["ttft_median"], 1),
                "speedup": round(d["ttft_median"] / c["ttft_median"], 2) if d and c["arm"] != "dense" else "",
                "selected": c["selected"],
            })

    # table rule: fastest arm reaching the floor (mean and min), else best quality
    table = {}
    trail = {}
    for ds in sorted({c["dataset"] for c in cells}):
        arms = [c for c in cells if c["dataset"] == ds and c["arm"] != "dense"]
        ok = [c for c in arms if c["q_mean"] >= args.quality_floor and c["q_min"] >= args.quality_floor]
        if ok:
            best = min(ok, key=lambda c: c["ttft_median"])
            reason = "floor-met"
        else:
            best = max(arms, key=lambda c: (c["q_mean"], c["q_min"]))
            reason = "floor-missed"
        name = ds.rsplit("/", 1)[-1]
        table[name] = best["arm"]
        trail[name] = {"arm": best["arm"], "reason": reason,
                       "q_mean": round(best["q_mean"], 4), "q_min": round(best["q_min"], 4),
                       "ttft_median": round(best["ttft_median"], 1)}

    with open(os.path.join(out, "portfolio.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("default = selection\n")
        for name, arm in sorted(table.items()):
            f.write(f"{name} = {arm}\n")
    with open(os.path.join(out, "portfolio.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"quality_floor": args.quality_floor, "model_sha256": sha256_of(model),
                   "cells": [{"dataset": c["dataset"], "arm": c["arm"],
                              "q_mean": round(c["q_mean"], 4), "q_min": round(c["q_min"], 4),
                              "ttft_median": round(c["ttft_median"], 1),
                              "selected": c["selected"]} for c in cells],
                   "table": trail}, f, indent=2)
        f.write("\n")

    with open(os.path.join(out, "matrix.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# AR010 combination matrix\n\n")
        f.write(f"model_sha256 = {sha256_of(model)}\n\n")
        for ds in sorted({c["dataset"] for c in cells}):
            f.write(f"## {ds}\n\n| arm | q_mean | q_min | selected | TTFT ms | vs dense |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            d = dense.get(ds)
            for c in sorted([c for c in cells if c["dataset"] == ds], key=lambda c: c["ttft_median"]):
                sp = (d["ttft_median"] / c["ttft_median"]) if d and c["arm"] != "dense" else 1.0
                f.write(f"| {c['arm']} | {c['q_mean']:.3f} | {c['q_min']:.3f} | {c['selected']} | "
                        f"{c['ttft_median']:.0f} | {sp:.2f}x |\n")
            f.write(f"\nrouted: **{table[ds.rsplit('/', 1)[-1]]}** "
                    f"({trail[ds.rsplit('/', 1)[-1]]['reason']})\n\n")
    print(f"wrote {out}/matrix.csv + portfolio.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
