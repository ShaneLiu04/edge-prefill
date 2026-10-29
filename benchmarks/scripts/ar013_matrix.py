#!/usr/bin/env python3
"""AR013 matrix: compact context vs the AR012 admission baseline.

Arms
----
ar012    AR012 shipped arm, identity positions (context grows to the prompt span)
ar013    same admission, block-compacted positions (KV span = admitted tokens)

The 64K dataset is the capability arm: the AR012 configuration cannot run it at
all (a 64K prompt needs 64K positions, the model trains at 32K), so a missing
cell for ar012 is the *expected* result there and the report treats it as such.

Everything is measured on the real GGUF; no synthetic scores.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = {
    "ar012": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
              "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24,
              "sentence_completion": 1, "sentence_trim": 0,
              "position_compaction": 0},
    "ar013": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
              "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24,
              "sentence_completion": 1, "sentence_trim": 0,
              "position_compaction": 1},
}

FLAT_BUDGET = {"deep/multifact8k": 0.05, "deep/multifact8k-hard": 0.05,
               "explen/len64k": 0.08}
NEW_TOKENS = {"deep/multifact8k": 48, "deep/multifact8k-hard": 48}


def write_conf(path, arm, curve, threads, max_new_tokens, phrase_weight,
               coverage_weight, flat_budget, dump_dir="", repeats=3):
    spec = ARMS[arm]
    lines = [f"mode = {spec['mode']}"]
    if flat_budget is not None:
        lines.append(f"budget = {flat_budget}")
    elif curve:
        lines.append(f"budget_curve = {curve}")
    lines += [
        "attn_top_k = 24",
        "pool_branches = false",
        "query_gating = false",
        "qg_bound = false",
        "embedding_model =",
        "anchor_ratio = 0.35",
        f"phrase_weight = {phrase_weight}",
        f"coverage_weight = {coverage_weight}",
        f"prior_gate = {spec['prior_gate']}",
        f"fill_priors = {spec['fill_priors']}",
        f"structure_budget = {spec.get('structure_budget', 0)}",
        f"anchor_span_radius = {spec.get('anchor_span_radius', 0)}",
        f"coverage_reserve = {spec.get('coverage_reserve', 0)}",
        f"span_admission = {spec.get('span_admission', 0)}",
        f"span_min_tokens = {spec.get('span_min_tokens', 24)}",
        f"sentence_completion = {spec.get('sentence_completion', 0)}",
        f"sentence_trim = {spec.get('sentence_trim', 0)}",
        f"span_max_tokens = {spec.get('span_max_tokens', 128)}",
        f"position_compaction = {spec.get('position_compaction', 0)}",
        f"threads = {threads}",
        "warmup = 0",
        f"max_new_tokens = {max_new_tokens}",
        f"repeats = {repeats}",
    ]
    if dump_dir:
        lines.append(f"dump_selection_dir = {dump_dir}")
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
        return None, f"rc={proc.returncode} wall={wall:.0f}s {proc.stderr[-300:]}"
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_raw.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        warn = [ln for ln in proc.stderr.splitlines() if "warn" in ln]
        return None, f"no samples wall={wall:.0f}s {warn[:2]}"
    return rows, None


def summarize(rows):
    want = [r for r in rows if int(r["dense"]) == 0]
    if not want:
        return None
    q = [float(r["quality"]) for r in want]
    t = [float(r["ttft_ms"]) for r in want]
    mean = lambda k: statistics.mean(float(r[k]) for r in want) if k in want[0] else 0.0
    mins = [float(r["ttft_min_ms"]) for r in want
            if "ttft_min_ms" in r and float(r["ttft_min_ms"]) > 0]
    return {
        "cases": len(want),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "ttft_median": statistics.median(t),
        "ttft_min": statistics.median(mins) if mins else 0.0,
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in want)),
        "prompt": int(statistics.mean(int(r["prompt_tokens"]) for r in want)),
        "position_span": int(statistics.mean(int(r["position_span"]) for r in want)),
        "ctx_cells": int(statistics.mean(int(r["ctx_cells"]) for r in want)),
        "kv_alloc_mib": statistics.mean(float(r["kv_alloc_bytes"]) for r in want) / 2**20,
        "kv_used_mib": statistics.mean(float(r["kv_used_bytes"]) for r in want) / 2**20,
        "rss_mib": statistics.mean(float(r["rss_mb"]) for r in want),
        "compaction": want[0].get("compaction_reason", ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k,explen/len16k,"
                                          "deep/len32k,deep/multineedle,"
                                          "deep/multifact8k,deep/multifact8k-hard,"
                                          "explen/len64k")
    ap.add_argument("--arms", default="ar012,ar013")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar013-compact-context")
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--coverage-weight", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repeats behind each TTFT median")
    ap.add_argument("--dump", action="store_true",
                    help="also write each case's admitted text for auditing")
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
        flat = FLAT_BUDGET.get(dataset)
        new_tokens = NEW_TOKENS.get(dataset, args.max_new_tokens)
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            tag = f"{name}-{arm}"
            cell_dir = os.path.join(out, tag)
            conf = os.path.join(cell_dir, "bench.conf")
            dump_dir = os.path.join(cell_dir, "selected") if args.dump else ""
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            existing = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")))
            if existing and not args.force:
                rows = []
                for f in existing:
                    with open(f, newline="", encoding="utf-8") as fh:
                        rows.extend(csv.DictReader(fh))
            else:
                for stale in glob.glob(os.path.join(cell_dir, "*_raw.csv")) + \
                             glob.glob(os.path.join(cell_dir, "*_env.json")) + \
                             glob.glob(os.path.join(cell_dir, "*_report.md")):
                    os.remove(stale)
                write_conf(conf, arm, os.path.join(ROOT, curve) if curve else "",
                           args.threads, new_tokens, args.phrase_weight,
                           args.coverage_weight, flat, dump_dir, args.repeats)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir, True)
                if err:
                    print(f"[skip] {tag}: {err}", flush=True)
                    cells.append({"dataset": dataset, "arm": arm, "tag": tag,
                                  "cases": 0, "q_mean": 0.0, "q_min": 0.0,
                                  "ttft_median": 0.0, "selected": 0, "prompt": 0,
                                  "position_span": 0, "ctx_cells": 0,
                                  "kv_alloc_mib": 0.0, "kv_used_mib": 0.0,
                                  "rss_mib": 0.0, "compaction": "unavailable"})
                    continue
            stats = summarize(rows)
            if stats is None:
                print(f"[fail] {tag}: no sparse samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, "tag": tag, **stats})
            print(f"[cell] {tag:<28} q={stats['q_mean']:.3f} min={stats['q_min']:.3f} "
                  f"sel={stats['selected']:5d} span={stats['position_span']:6d} "
                  f"ctx={stats['ctx_cells']:6d} kv={stats['kv_alloc_mib']:6.1f}MiB "
                  f"ttft={stats['ttft_median']:7.0f}ms min={stats['ttft_min']:7.0f}ms",
                  flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1
    fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median", "ttft_min",
              "selected",
              "prompt", "position_span", "ctx_cells", "kv_alloc_mib", "kv_used_mib",
              "rss_mib", "compaction"]
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k in fields})
    print("wrote", os.path.join(out, "matrix.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
