#!/usr/bin/env python3
"""AR014 matrix: session KV reuse vs recomputing the same conversation log.

Two arms on `deep/multiturn` (8 sessions x 5 turns):
  restart  session_reuse=0 — every turn re-prefills the whole log from position 0
  reuse    session_reuse=1 — only the turn's delta is decoded, the prefix KV is kept

Both arms walk the *same* log (same tokens, same positions), so per-case quality
must be identical; the difference is only the prefill work.

Plus two single-turn spot checks (len4k, multifact8k) that must reproduce the
committed AR013 matrix numbers — a session column is absent there, so the
session path must not engage at all.
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
    "restart": {"session_reuse": 0},
    "reuse": {"session_reuse": 1},
}

# Same shipped admission as AR012/AR013; only session_reuse differs.
BASE_CONF = {
    "mode": "balanced",
    "attn_top_k": 24,
    "pool_branches": "false",
    "query_gating": "false",
    "qg_bound": "false",
    "embedding_model": "",
    "anchor_ratio": 0.35,
    "phrase_weight": 0.5,
    "coverage_weight": 3.0,
    "prior_gate": 1,
    "fill_priors": 0,
    "structure_budget": 64,
    "anchor_span_radius": 0,
    "coverage_reserve": 0,
    "span_admission": 1,
    "span_min_tokens": 24,
    "sentence_completion": 1,
    "sentence_trim": 0,
    "span_max_tokens": 128,
    "position_compaction": 1,
}

FLAT_BUDGET = {"deep/multifact8k": 0.05, "deep/multifact8k-hard": 0.05,
               "deep/multiturn": 0.05}
NEW_TOKENS = {"deep/multifact8k": 48, "deep/multifact8k-hard": 48}


def write_conf(path, arm, curve, threads, repeats, max_new_tokens, flat_budget):
    lines = [f"mode = {BASE_CONF['mode']}"]
    if flat_budget is not None:
        lines.append(f"budget = {flat_budget}")
    else:
        lines.append(f"budget_curve = {curve}")
    for key, value in BASE_CONF.items():
        if key == "mode":
            continue
        lines.append(f"{key} = {value}")
    lines.append(f"session_reuse = {ARMS[arm]['session_reuse']}")
    lines += [f"threads = {threads}", "warmup = 0",
              f"max_new_tokens = {max_new_tokens}", f"repeats = {repeats}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def run_cell(bin_path, conf, model, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [bin_path, "bench", "--config", conf, "--model", model,
           "--data-dir", data_dir, "--out", out_dir, "--bench-skip-dense"]
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
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def summarize(rows):
    want = [r for r in rows if int(r["dense"]) == 0]
    if not want:
        return None
    ttft = [float(r["ttft_ms"]) for r in want]
    mins = [float(r["ttft_min_ms"]) for r in want if float(r.get("ttft_min_ms", 0)) > 0]
    return {
        "turns": len(want),
        "q_mean": statistics.mean(float(r["quality"]) for r in want),
        "ttft_median": statistics.median(ttft),
        "ttft_min": statistics.median(mins) if mins else statistics.median(ttft),
        "appended": int(statistics.mean(int(r["appended_tokens"]) for r in want)),
        "reused": int(statistics.mean(int(r["reused_tokens"]) for r in want)),
        "log_tokens": int(statistics.mean(int(r["session_log_tokens"]) for r in want)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="deep/multiturn")
    ap.add_argument("--arms", default="restart,reuse")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar014-session-kv-reuse")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    bin_path = args.bin if os.path.isabs(args.bin) else os.path.join(ROOT, args.bin)
    model = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    curve = os.path.join(ROOT, args.curve) if args.curve else ""

    cells = []
    per_case = {}
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        name = dataset.rsplit("/", 1)[-1]
        data_dir = os.path.join(ROOT, "benchmarks/data", dataset)
        flat = FLAT_BUDGET.get(dataset)
        new_tokens = NEW_TOKENS.get(dataset, args.max_new_tokens)
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
                for stale in glob.glob(os.path.join(cell_dir, "*_raw.csv")) + \
                             glob.glob(os.path.join(cell_dir, "*_env.json")) + \
                             glob.glob(os.path.join(cell_dir, "*_report.md")):
                    os.remove(stale)
                write_conf(conf, arm, curve, args.threads, args.repeats, new_tokens, flat)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir)
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
            stats = summarize(rows)
            if stats is None:
                print(f"[fail] {tag}: no sparse samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, **stats})
            per_case[(dataset, arm)] = {
                r["case"]: r for r in rows if int(r["dense"]) == 0
            }
            print(f"[cell] {tag:<20} turns={stats['turns']:3d} q={stats['q_mean']:.3f} "
                  f"ttft_med={stats['ttft_median']:8.0f}ms ttft_min={stats['ttft_min']:8.0f}ms "
                  f"reused={stats['reused']:5d} appended={stats['appended']:5d}",
                  flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1
    fields = ["dataset", "arm", "turns", "q_mean", "ttft_median", "ttft_min",
              "reused", "appended", "log_tokens"]
    # A run only recomputes the requested datasets; keep the other datasets'
    # rows instead of dropping them from the summary table.
    matrix_path = os.path.join(out, "matrix.csv")
    fresh_ds = {c["dataset"] for c in cells}
    if os.path.exists(matrix_path):
        with open(matrix_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dataset"] not in fresh_ds:
                    cells.append({k: r.get(k, "") for k in fields})
    cells.sort(key=lambda c: (c["dataset"], c["arm"]))
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k in fields})

    # per-turn table: the point of the AR is the shape of the curve, not a mean
    rows_out = []
    for (dataset, arm), cases in per_case.items():
        for case, r in cases.items():
            rows_out.append({
                "dataset": dataset, "arm": arm, "case": case,
                "turn": case.rsplit("-t", 1)[-1].split("@")[0],
                "quality": round(float(r["quality"]), 6),
                "ttft_min_ms": round(float(r.get("ttft_min_ms", r["ttft_ms"])), 1),
                "reused_tokens": r["reused_tokens"],
                "appended_tokens": r["appended_tokens"],
                "log_tokens": r["session_log_tokens"],
                "log_hash": r["session_log_hash"],
            })
    # A run only recomputes the requested datasets, so merge the per-turn table
    # with what is already there instead of dropping the other datasets' turns.
    turns_path = os.path.join(out, "turns.csv")
    fresh = {(r["dataset"], r["arm"]) for r in rows_out}
    if os.path.exists(turns_path):
        with open(turns_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["dataset"], r["arm"]) not in fresh:
                    rows_out.append({k: r.get(k, "") for k in rows_out[0]})
    rows_out.sort(key=lambda r: (r["dataset"], r["arm"], r["case"]))
    with open(turns_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", os.path.join(out, "matrix.csv"), "and turns.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
