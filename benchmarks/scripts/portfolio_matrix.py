#!/usr/bin/env python3
"""AR008 FR-4: measure the execution-combination matrix and derive the table.

For every (dataset, combination) cell this runs `edge-prefill bench` and writes

    <out>/<dataset>-<combo>/{bench.conf,raw.csv,<ts>_env.json}
    <out>/matrix.csv          dataset,combo,cases,q_mean,q_min,ttft_median,selected
    <out>/matrix.md           human table with speedups against dense
    <out>/portfolio.txt       the planner table (rule below)
    <out>/portfolio.json      the rule's inputs/outputs for the record

The table rule is the one the SRS sells: **the fastest combination whose mean
quality reaches the floor, and if none does, the highest-quality one**. It is
computed from measurements only - a missing cell is reported as `missing` and
never interpolated. `dense` is measured once per dataset (or reused with
--dense-from) and only feeds the speedup column.

Usage:
  portfolio_matrix.py --bin build-wsl/edge-prefill --model M.gguf \
      --out benchmarks/results/ar008-portfolio \
      [--datasets explen/len4k,...] [--combos selection,gate-p16,gate-p24-local32,dense]
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

# Combo definitions: everything is expressed with existing knobs, so a cell is
# exactly what a user could type into a conf by hand.
COMBOS = {
    "selection": {
        "mode": "balanced", "qg_bound": "false", "attn_top_k": 24,
        "budget": None,  # curve decides
    },
    "gate-p16": {
        "mode": "balanced", "qg_bound": "true", "qg_sel_pages": 16,
        "qg_local_rows": 0, "attn_top_k": 24, "budget": 0.25,
    },
    "gate-p24-local32": {
        "mode": "balanced", "qg_bound": "true", "qg_sel_pages": 24,
        "qg_local_rows": 32, "attn_top_k": 24, "budget": 0.25,
    },
    "dense": {
        "mode": "dense", "qg_bound": "false", "attn_top_k": 24, "budget": 1.0,
    },
}


def sha256_of(path):
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 else "unavailable"


def write_conf(path, combo, curve, threads, model, phrase_weight):
    spec = COMBOS[combo]
    lines = [f"mode = {spec['mode']}"]
    if spec.get("budget") is not None:
        lines.append(f"budget = {spec['budget']}")
    elif curve:
        lines.append(f"budget_curve = {curve}")
    lines += [
        f"attn_top_k = {spec.get('attn_top_k', 24)}",
        "pool_branches = false",
        "query_gating = false",
        f"qg_bound = {spec.get('qg_bound', 'false')}",
    ]
    if "qg_sel_pages" in spec:
        lines.append(f"qg_sel_pages = {spec['qg_sel_pages']}")
        lines.append(f"qg_local_rows = {spec.get('qg_local_rows', 0)}")
    lines += [
        "embedding_model =",
        "anchor_ratio = 0.35",
        f"phrase_weight = {phrase_weight}",
        f"threads = {threads}",
        "warmup = 0",
        "max_new_tokens = 16",
        "repeats = 1",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def run_cell(bin_path, conf, model, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    proc = subprocess.run(
        [bin_path, "bench", "--config", conf, "--model", model,
         "--data-dir", data_dir, "--out", out_dir, "--bench-skip-dense"],
        cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - started
    if proc.returncode != 0:
        return None, f"rc={proc.returncode} wall={wall:.0f}s {proc.stderr[-200:]}"
    files = sorted(glob.glob(os.path.join(out_dir, "*_raw.csv")) +
                   glob.glob(os.path.join(out_dir, "raw.csv")))
    rows = []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def summarize(rows):
    q = [float(r["quality"]) for r in rows]
    t = [float(r["ttft_ms"]) for r in rows]
    return {
        "cases": len(rows),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "ttft_median": statistics.median(t),
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in rows)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build-wsl/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k,explen/len16k,"
                                          "deep/len32k,deep/multineedle")
    ap.add_argument("--combos", default="selection,gate-p16,gate-p24-local32,dense")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar008-portfolio")
    ap.add_argument("--quality-floor", type=float, default=0.95)
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    bin_path = args.bin if os.path.isabs(args.bin) else os.path.join(ROOT, args.bin)
    model = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    curve = os.path.join(ROOT, args.curve) if args.curve else ""
    os.makedirs(out, exist_ok=True)
    combos = [c.strip() for c in args.combos.split(",") if c.strip()]
    for c in combos:
        if c not in COMBOS:
            sys.stderr.write(f"unknown combo {c}\n")
            return 2

    cells = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        name = dataset.rsplit("/", 1)[-1]
        data_dir = os.path.join(ROOT, "benchmarks/data", dataset)
        for combo in combos:
            tag = f"{name}-{combo}"
            cell_dir = os.path.join(out, tag)
            conf = os.path.join(cell_dir, "bench.conf")
            existing = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")) +
                              glob.glob(os.path.join(cell_dir, "raw.csv")))
            if existing and not args.force:
                rows = []
                for f in existing:
                    with open(f, newline="", encoding="utf-8") as fh:
                        rows.extend(csv.DictReader(fh))
                err = None if rows else "no samples"
            else:
                write_conf(conf, combo, curve, args.threads, model, args.phrase_weight)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir)
            if err:
                print(f"[fail] {tag}: {err}", flush=True)
                continue
            stats = summarize(rows)
            cell = {"dataset": dataset, "combo": combo, "tag": tag, **stats}
            cells.append(cell)
            print(f"[cell] {tag:<32} q={stats['q_mean']:.3f} "
                  f"min={stats['q_min']:.3f} ttft={stats['ttft_median']:.0f}ms "
                  f"sel={stats['selected']}", flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    dense = {(c["dataset"]): c for c in cells if c["combo"] == "dense"}
    cols = ["dataset", "combo", "cases", "q_mean", "q_min", "ttft_median",
            "speedup", "selected"]
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["combo"])):
            d = dense.get(c["dataset"])
            row = {k: c[k] for k in ("dataset", "combo", "cases", "selected")}
            row["q_mean"] = round(c["q_mean"], 4)
            row["q_min"] = round(c["q_min"], 4)
            row["ttft_median"] = round(c["ttft_median"], 1)
            row["speedup"] = round(d["ttft_median"] / c["ttft_median"], 2) if d else ""
            w.writerow(row)

    # --- table rule: fastest combo at/above the floor, else best quality ----
    table = {}
    rationale = {}
    for dataset in sorted({c["dataset"] for c in cells}):
        group = [c for c in cells if c["dataset"] == dataset and c["combo"] != "dense"]
        if not group:
            continue
        passing = [c for c in group
                   if c["q_mean"] >= args.quality_floor and c["q_min"] >= args.quality_floor]
        if passing:
            best = min(passing, key=lambda c: c["ttft_median"])
            note = "floor-met"
        else:
            best = max(group, key=lambda c: (c["q_mean"], -c["ttft_median"]))
            note = "floor-missed"
        table[dataset] = best["combo"]
        rationale[dataset] = {
            "combo": best["combo"], "note": note,
            "q_mean": round(best["q_mean"], 4), "ttft_median": round(best["ttft_median"], 1),
            "candidates": [{"combo": c["combo"], "q_mean": round(c["q_mean"], 4),
                            "ttft_median": round(c["ttft_median"], 1)} for c in group],
        }

    # Map dataset -> bucket key using the measured prompt length and the
    # selector's own profile fields (specific terms / phrase hits /
    # concentration) recorded in the raw CSV of the routed selection cell.
    keys = {}
    for dataset in table:
        rows = []
        for f in sorted(glob.glob(os.path.join(out, f"{dataset.rsplit('/', 1)[-1]}-selection",
                                               "*_raw.csv"))):
            with open(f, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        if not rows:
            continue
        r = rows[0]
        n_tokens = max(int(x["prompt_tokens"]) for x in rows)
        if n_tokens < 5000:
            bucket = "lt4k"
        elif n_tokens < 12000:
            bucket = "8k"
        elif n_tokens < 24000:
            bucket = "16k"
        else:
            bucket = "32k"
        specific = int(r.get("specific_terms", 0) or 0) if "specific_terms" in r else \
            (2 if int(r.get("phrase_blocks", 0) or 0) > 0 else 0)
        keys[dataset] = f"{'high' if specific >= 2 else 'low'}:{bucket}"

    model_sha = sha256_of(model)
    lines = [
        "# AR008 execution-combination portfolio (generated by portfolio_matrix.py)",
        "# rule: fastest combo whose mean+min quality reach the floor; else best quality",
        "# do not edit by hand: regenerate after any model/dataset/selector change",
        f"model_sha256 = {model_sha}",
        f"generated = {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"quality_floor = {args.quality_floor}",
        "default = selection",
    ]
    seen = set()
    for dataset in sorted(keys) + sorted(set(table) - set(keys)):
        key = keys.get(dataset)
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(f"{key} = {table[dataset]}")
    with open(os.path.join(out, "portfolio.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(out, "portfolio.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"model_sha256": model_sha, "quality_floor": args.quality_floor,
                   "keys": keys, "table": table, "rationale": rationale}, f, indent=2)
        f.write("\n")

    md = ["# AR008 combination matrix", "",
          f"model `{model_sha[:16]}...`, quality floor {args.quality_floor}, "
          f"phrase_weight {args.phrase_weight}.", "",
          "| dataset | combo | cases | q mean | q min | TTFT ms | vs dense | selected |",
          "|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["dataset"], c["combo"])):
        d = dense.get(c["dataset"])
        speed = f"{d['ttft_median'] / c['ttft_median']:.1f}x" if d else "-"
        md.append(f"| {c['dataset']} | {c['combo']} | {c['cases']} | {c['q_mean']:.3f} | "
                  f"{c['q_min']:.3f} | {c['ttft_median']:.0f} | {speed} | {c['selected']} |")
    md += ["", "## Routed combinations", ""]
    for dataset, info in sorted(rationale.items()):
        md.append(f"- {dataset} -> **{info['combo']}** ({info['note']}, "
                  f"q={info['q_mean']}, {info['ttft_median']:.0f} ms)")
    with open(os.path.join(out, "matrix.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")

    print(f"cells={len(cells)} -> {os.path.join(out, 'matrix.md')}, "
          f"portfolio.txt, portfolio.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
