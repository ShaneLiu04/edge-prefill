#!/usr/bin/env python3
"""AR017 matrix: serial decode vs continuous batching (K = 2/4/8).

Each cell is a full bench process so the measured wall time is real end-to-end
work, not an extrapolation. Arms differ only in `decode_batch_size`; admission,
budget and threads come from the AR016 probe plan so the comparison isolates the
decode scheduler.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = [
    "explen/len4k",
    "explen/len8k",
    "deep/multifact8k",
    "deep/multifact8k-hard",
]

BASE_CONF = {
    "mode": "balanced",
    "attn_top_k": 24,
    "anchor_ratio": 0.35,
    "phrase_weight": 0.5,
    "coverage_weight": 3.0,
    "prior_gate": 1,
    "fill_priors": 0,
    "structure_budget": 64,
    "span_admission": 1,
    "span_min_tokens": 24,
    "sentence_completion": 1,
    "sentence_trim": 0,
    "span_max_tokens": 128,
    "position_compaction": 1,
}

FLAT_DATASETS = {"deep/multifact8k", "deep/multifact8k-hard"}


def read_plan(plan_dir):
    plan = {}
    with open(os.path.join(plan_dir, "plan.conf"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                plan[key.strip()] = value.strip()
    return plan


def write_conf(path, dataset, k, plan, threads, repeats, warmup, max_new, ctx):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in BASE_CONF.items():
            f.write(f"{key} = {value}\n")
        if dataset in FLAT_DATASETS:
            f.write("budget = 0.05\n")
        f.write(f"threads = {threads}\n")
        f.write(f"warmup = {warmup}\n")
        f.write(f"repeats = {repeats}\n")
        f.write(f"max_new_tokens = {max_new}\n")
        f.write(f"decode_batch_size = {k}\n")
        if k > 1:
            f.write(f"decode_batch_ctx = {ctx if ctx else k * 1536}\n")
        for key in ("cost_a_ms", "cost_b_ms_per_token", "cost_r2", "cost_n"):
            if key in plan:
                f.write(f"{key} = {plan[key]}\n")


def load_rows(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_raw.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r.get("dense") == "0"]
    return rows


def summarize(rows, wall_ms):
    if not rows:
        return None
    # Decode wall time must not be summed per case for a batched arm: every case
    # in a batch reports the same batch wall, so summing would multiply it by K.
    eff = max(int(r.get("batch_size", 1) or 1) for r in rows)
    n_batches = (len(rows) + eff - 1) // eff
    batch_wall = statistics.median(float(r.get("batch_wall_ms", 0) or 0) for r in rows)
    if eff > 1 and batch_wall > 0:
        decode_wall = batch_wall * n_batches
    else:
        decode_wall = sum(max(0.0, float(r["e2e_ms"]) - float(r["ttft_min_ms"])) for r in rows)
    return {
        "decode_wall_ms": decode_wall,
        "batches": n_batches,
        "cases": len(rows),
        "q_mean": statistics.mean(float(r["quality"]) for r in rows),
        "q_min": min(float(r["quality"]) for r in rows),
        "ttft_min": statistics.median(float(r["ttft_min_ms"]) for r in rows),
        "e2e_median": statistics.median(float(r["e2e_ms"]) for r in rows),
        "e2e_sum": sum(float(r["e2e_ms"]) for r in rows),
        "wall_ms": wall_ms,
        "batch_effective": max(int(r.get("batch_size", 1) or 1) for r in rows),
        "steps_median": statistics.median(int(r.get("batch_steps", 0) or 0) for r in rows),
        "ms_per_step": statistics.median(
            float(r.get("decode_ms_per_token", 0) or 0) for r in rows),
        "threads": int(rows[0].get("n_threads", 0) or 0),
        "device_state": rows[0].get("device_state", ""),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--arms", default="1,2,4,8")
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--out", default="benchmarks/results/ar017-batched-decode")
    ap.add_argument("--threads", type=int, default=0, help="0 = plan")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args(argv)

    datasets = [d for d in args.datasets.split(",") if d]
    arms = [int(a) for a in args.arms.split(",") if a]
    threads = args.threads or int(read_plan(os.path.join(ROOT, args.plan_dir)).get("threads", 0) or 0)
    if threads <= 0:
        sys.stderr.write("plan has no threads; pass --threads\n")
        return 2
    plan = read_plan(os.path.join(ROOT, args.plan_dir))
    out = os.path.join(ROOT, args.out)
    conf_dir = os.path.join(out, "_conf")
    os.makedirs(conf_dir, exist_ok=True)

    cells = []
    per_case = {}
    for dataset in datasets:
        for k in arms:
            tag = f"{dataset.replace('/', '-')}-k{k}{args.tag_suffix}"
            cell_dir = os.path.join(out, tag)
            meta_path = os.path.join(cell_dir, "_wall.json")
            if not args.force and glob.glob(os.path.join(cell_dir, "*_raw.csv")) \
                    and os.path.exists(meta_path):
                print(f"[skip] {tag}", flush=True)
                rows = load_rows(cell_dir)
                with open(meta_path, encoding="utf-8") as f:
                    wall_ms = json.load(f)["wall_ms"]
            else:
                if os.path.isdir(cell_dir):
                    shutil.rmtree(cell_dir)
                os.makedirs(cell_dir, exist_ok=True)
                conf = os.path.join(conf_dir, f"{tag}.conf")
                write_conf(conf, dataset, k, plan, threads, args.repeats,
                           args.warmup, args.max_new_tokens, args.ctx)
                cmd = [os.path.join(ROOT, args.bin), "bench", "--config", conf,
                       "--model", args.model,
                       "--data-dir", os.path.join(ROOT, "benchmarks", "data", dataset),
                       "--out", cell_dir, "--bench-skip-dense"]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                wall_ms = (time.time() - t0) * 1000.0
                if proc.returncode != 0:
                    tail = "\n".join((proc.stdout or "").splitlines()[-4:])
                    print(f"[fail] {tag}: rc={proc.returncode} {tail}", flush=True)
                    continue
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"wall_ms": wall_ms}, f)
                rows = load_rows(cell_dir)
            stats = summarize(rows, wall_ms)
            if stats is None:
                print(f"[fail] {tag}: no sparse rows", flush=True)
                continue
            cells.append({"dataset": dataset, "k": k, **stats})
            per_case[(dataset, k)] = {r["case"]: r for r in rows}
            print(f"[cell] {tag:<32} cases={stats['cases']:3d} q={stats['q_mean']:.3f} "
                  f"ttft={stats['ttft_min']:7.0f} e2e_med={stats['e2e_median']:8.0f} "
                  f"wall={wall_ms/1000.0:7.1f}s batch={stats['batch_effective']} "
                  f"ms/step={stats['ms_per_step']:6.1f}", flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1
    fields = list(cells[0].keys())
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)

    with open(os.path.join(out, "cases.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "case", "k", "quality", "generated_hash", "ttft_min_ms",
                         "e2e_ms", "batch_size", "batch_steps", "decode_ms_per_token"])
        for (dataset, k), rows in sorted(per_case.items()):
            for case, r in sorted(rows.items()):
                writer.writerow([dataset, case, k, r["quality"],
                                 r.get("generated_hash", ""), r["ttft_min_ms"],
                                 r["e2e_ms"], r.get("batch_size", 1),
                                 r.get("batch_steps", 0),
                                 r.get("decode_ms_per_token", 0)])
    print(f"wrote {os.path.join(out, 'matrix.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
