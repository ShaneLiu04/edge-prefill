#!/usr/bin/env python3
"""AR009 FR-2: measure the multi-fact coverage ablation.

For every (dataset, coverage_weight) cell this runs `edge-prefill bench` and
writes

    <out>/<dataset>-cw<weight>/{bench.conf,raw.csv,<ts>_env.json}
    <out>/ablation.csv   dataset,coverage_weight,cases,q_mean,q_min,q_p10,
                        covered_queries,selected,ttft_median

`coverage_weight=0` is the AR008 selection exactly (no sub-query set is built),
so the table is a true before/after, measured with the same binary, model,
threads and datasets. `covered_queries` is the AR009 audit column: how many of
the requested facts the admission actually placed in the context.

Dense is measured once per dataset with --dense so the table can report what
fraction of the full-prompt quality the sparse selection reaches; it is a
reference point, not a candidate.

Usage:
  ar009_ablation.py --bin build-wsl/edge-prefill --model M.gguf \
      --out benchmarks/results/ar009-multifact \
      [--datasets deep/multifact8k,deep/multifact8k-hard] [--weights 0,3]
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


def sha256_of(path):
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 else "unavailable"


def write_conf(path, weight, threads, max_new_tokens, budget, phrase_weight):
    lines = [
        "mode = balanced",
        f"budget = {budget}",
        "attn_top_k = 24",
        "pool_branches = false",
        "query_gating = false",
        "qg_bound = false",
        "embedding_model =",
        "anchor_ratio = 0.35",
        f"phrase_weight = {phrase_weight}",
        f"coverage_weight = {weight}",
        f"threads = {threads}",
        "warmup = 0",
        f"max_new_tokens = {max_new_tokens}",
        "repeats = 1",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return lines


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
        return None, f"rc={proc.returncode} wall={wall:.0f}s {proc.stderr[-300:]}"
    files = sorted(glob.glob(os.path.join(out_dir, "*_raw.csv")))
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def summarize(rows, weight):
    sparse = [r for r in rows if int(r["dense"]) == 0]
    if not sparse:
        return None
    q = sorted(float(r["quality"]) for r in sparse)
    tt = [float(r["ttft_ms"]) for r in sparse]
    covered = [float(r["coverage_queries"]) for r in sparse]
    return {
        "cases": len(sparse),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "q_p10": q[max(0, int(0.1 * (len(q) - 1)))],
        "covered_queries": statistics.mean(covered),
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in sparse)),
        "ttft_median": statistics.median(tt),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build-wsl/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="deep/multifact8k,deep/multifact8k-hard")
    ap.add_argument("--weights", default="0,3")
    ap.add_argument("--budget", type=float, default=0.05)
    ap.add_argument("--out", default="benchmarks/results/ar009-multifact")
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    bin_path = args.bin if os.path.isabs(args.bin) else os.path.join(ROOT, args.bin)
    model = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    weights = [float(w) for w in args.weights.split(",") if w.strip()]

    cells = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        name = dataset.rsplit("/", 1)[-1]
        data_dir = os.path.join(ROOT, "benchmarks/data", dataset)
        for weight in weights:
            tag = f"{name}-cw{weight:g}"
            cell_dir = os.path.join(out, tag)
            conf = os.path.join(cell_dir, "bench.conf")
            existing = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")))
            if existing and not args.force:
                rows = []
                for f in existing:
                    with open(f, newline="", encoding="utf-8") as fh:
                        rows.extend(csv.DictReader(fh))
            else:
                lines = write_conf(conf, weight, args.threads, args.max_new_tokens,
                                   args.budget, args.phrase_weight)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir, True)
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
                os.makedirs(cell_dir, exist_ok=True)
                with open(os.path.join(cell_dir, "config.used"), "w",
                          encoding="utf-8", newline="\n") as fh:
                    fh.write("\n".join(lines) + "\n")
            stats = summarize(rows, weight)
            if stats is None:
                print(f"[fail] {tag}: no sparse samples", flush=True)
                continue
            cells.append({"dataset": dataset, "coverage_weight": weight,
                          "tag": tag, **stats})
            print(f"[cell] {tag:<28} q={stats['q_mean']:.3f} "
                  f"min={stats['q_min']:.3f} covered={stats['covered_queries']:.1f} "
                  f"ttft={stats['ttft_median']:.0f}ms", flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    cols = ["dataset", "coverage_weight", "cases", "q_mean", "q_min", "q_p10",
            "covered_queries", "selected", "ttft_median"]
    with open(os.path.join(out, "ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["coverage_weight"])):
            row = {k: c[k] for k in ("dataset", "coverage_weight", "cases", "selected")}
            row["q_mean"] = round(c["q_mean"], 4)
            row["q_min"] = round(c["q_min"], 4)
            row["q_p10"] = round(c["q_p10"], 4)
            row["covered_queries"] = round(c["covered_queries"], 2)
            row["ttft_median"] = round(c["ttft_median"], 1)
            w.writerow(row)

    # delta report, per dataset
    by_ds = {}
    for c in cells:
        by_ds.setdefault(c["dataset"], {})[c["coverage_weight"]] = c
    baseline = 0.0
    with open(os.path.join(out, "ablation.md"), "w", encoding="utf-8",
              newline="\n") as f:
        f.write("# AR009 multi-fact coverage ablation\n\n")
        f.write(f"model_sha256 = {sha256_of(model)}\n\n")
        f.write(f"budget = {args.budget}, phrase_weight = {args.phrase_weight}, "
                f"max_new_tokens = {args.max_new_tokens}, threads = {args.threads}\n\n")
        f.write("| dataset | coverage_weight | cases | q_mean | q_min | covered facts "
                "| selected tok | TTFT ms |\n|---|---|---|---|---|---|---|---|\n")
        for ds in sorted(by_ds):
            for w_ in sorted(by_ds[ds]):
                c = by_ds[ds][w_]
                f.write(f"| {ds} | {w_:g} | {c['cases']} | {c['q_mean']:.3f} | "
                        f"{c['q_min']:.3f} | {c['covered_queries']:.2f} | "
                        f"{c['selected']} | {c['ttft_median']:.0f} |\n")
        f.write("\n| dataset | q(AR008) | q(AR009) | delta |\n|---|---|---|---|\n")
        for ds in sorted(by_ds):
            if baseline in by_ds[ds] and 3.0 in by_ds[ds]:
                a = by_ds[ds][baseline]["q_mean"]
                b = by_ds[ds][3.0]["q_mean"]
                f.write(f"| {ds} | {a:.3f} | {b:.3f} | {b - a:+.3f} |\n")

    with open(os.path.join(out, "ablation.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump({"model_sha256": sha256_of(model), "budget": args.budget,
                   "phrase_weight": args.phrase_weight,
                   "max_new_tokens": args.max_new_tokens,
                   "threads": args.threads, "cells": cells}, f, indent=2)
        f.write("\n")
    print(f"wrote {out}/ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
