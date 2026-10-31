#!/usr/bin/env python3
"""AR018 matrix: serial vs AR017 batch vs GPU-only vs CPU/GPU pipeline.

Arms run interleaved, round by round, so host drift hits every arm equally
(the AR017 lesson: single-point timings on this machine swing 2-3x).
Each cell is a full bench process; the numbers are real end-to-end work.

  python3 benchmarks/scripts/ar018_matrix.py --plan-dir <probe plan> [--rounds 2]
"""
import argparse
import csv
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = ["explen/len4k", "deep/multifact8k", "deep/multifact8k-hard"]
FLAT = {"deep/multifact8k", "deep/multifact8k-hard"}

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

ARMS = {
    # name: (binary, extra conf)
    "serial": ("build/edge-prefill", {"decode_batch_size": 1, "n_gpu_layers": 0}),
    "batch": ("build/edge-prefill", {"decode_batch_size": 0}),  # K from --k
    "gpuall": ("build-cuda/edge-prefill", {"decode_batch_size": 0, "n_gpu_layers": 99}),
    "pipeline": ("build-cuda/edge-prefill", {
        "decode_batch_size": 0,
        "n_gpu_layers": 0,
        "pipeline_mode": 1,
        "pipeline_prefill_gpu_layers": 99,
    }),
}


def read_plan(plan_dir):
    plan = {}
    path = os.path.join(ROOT, plan_dir, "plan.conf")
    if not os.path.exists(path):
        return plan
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [x.strip() for x in line.split("=", 1)]
        plan[key] = value
    return plan


def write_conf(path, dataset, k, plan, rounds_ahead, max_new):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in BASE_CONF.items():
            f.write(f"{key} = {value}\n")
        if dataset in FLAT:
            f.write("budget = 0.05\n")
        f.write(f"threads = {plan.get('threads', 12)}\n")
        f.write("warmup = 1\nrepeats = 2\n")
        f.write(f"max_new_tokens = {max_new}\n")
        f.write(f"decode_batch_size = {k}\n")
        if k > 1:
            f.write(f"decode_batch_ctx = {k * 3072}\n")
        f.write(f"pipeline_ahead = {rounds_ahead}\n")
        for key in ("cost_a_ms", "cost_b_ms_per_token", "cost_r2", "cost_n"):
            if key in plan:
                f.write(f"{key} = {plan[key]}\n")


def load_rows(cell_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r.get("dense") == "0"]
    return rows


def summarize(rows, wall_ms):
    if not rows:
        return None
    eff = max(int(r.get("batch_size", 1) or 1) for r in rows)
    n_batches = (len(rows) + eff - 1) // eff
    batch_wall = st.median(float(r.get("batch_wall_ms", 0) or 0) for r in rows)
    if eff > 1 and batch_wall > 0:
        decode_wall = batch_wall * n_batches
    else:
        decode_wall = sum(max(0.0, float(r["e2e_ms"]) - float(r["ttft_min_ms"])) for r in rows)
    prefill_sum = sum(float(r.get("ttft_min_ms", 0) or 0) for r in rows)
    return {
        "cases": len(rows),
        "q_mean": st.mean(float(r["quality"]) for r in rows),
        "q_min": min(float(r["quality"]) for r in rows),
        "ttft_min": st.median(float(r["ttft_min_ms"]) for r in rows),
        "e2e_median": st.median(float(r["e2e_ms"]) for r in rows),
        "decode_wall_ms": decode_wall,
        "batch_effective": eff,
        "queue_wait_ms": st.median(float(r.get("queue_wait_ms", 0) or 0) for r in rows),
        "handoff_ms": st.median(float(r.get("handoff_ms", 0) or 0) for r in rows),
        "handoff_bytes": max(int(r.get("handoff_bytes", 0) or 0) for r in rows),
        "pipeline_mode": int(rows[0].get("pipeline_mode", 0) or 0),
        "prefill_device_state": rows[0].get("prefill_device_state", ""),
        "decode_device_state": rows[0].get("device_state", ""),
        # Sum of the two stages over the elapsed wall: 1.0 = the stages ran
        # back to back, >1 = part of the work overlapped, 2.0 = perfect
        # overlap. Start-up (two model loads) counts against the ratio, so a
        # short run is dominated by it.
        "overlap_ratio": (prefill_sum + decode_wall) / wall_ms if wall_ms > 0 else 0.0,
        "wall_ms": wall_ms,
        "hashes": ",".join(sorted({r["generated_hash"] for r in rows})),
        "per_case_q": {r["case"]: float(r["quality"]) for r in rows},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--arms", default="serial,batch,gpuall,pipeline")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="benchmarks/results/ar018-disaggregated")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    plan = read_plan(args.plan_dir)
    datasets = [d for d in args.datasets.split(",") if d]
    arms = [a for a in args.arms.split(",") if a]
    out = os.path.join(ROOT, args.out)
    conf_dir = os.path.join(out, "_conf")
    os.makedirs(conf_dir, exist_ok=True)

    header = ("dataset,arm,round,cases,q_mean,q_min,ttft_min,e2e_median,"
              "decode_wall_ms,batch_effective,queue_wait_ms,handoff_ms,handoff_bytes,"
              "pipeline_mode,prefill_device_state,decode_device_state,overlap_ratio,"
              "wall_ms,hashes")
    csv_path = os.path.join(out, "matrix.csv")
    if args.force and os.path.exists(csv_path):
        os.remove(csv_path)

    for rnd in range(1, args.rounds + 1):
        for dataset in datasets:
            for arm in arms:
                binary, extra = ARMS[arm]
                tag = f"{dataset.replace('/', '-')}-{arm}-r{rnd}"
                cell = os.path.join(out, tag)
                if os.path.isdir(cell):
                    shutil.rmtree(cell)
                os.makedirs(cell, exist_ok=True)
                conf = os.path.join(conf_dir, f"{tag}.conf")
                k = extra["decode_batch_size"] if extra.get("decode_batch_size", 0) else args.k
                write_conf(conf, dataset, k, plan, args.k, args.max_new_tokens)
                with open(conf, "a", encoding="utf-8") as f:
                    for key, value in extra.items():
                        if key == "decode_batch_size":
                            continue
                        f.write(f"{key} = {value}\n")
                cmd = [os.path.join(ROOT, binary), "bench", "--config", conf,
                       "--model", args.model,
                       "--data-dir", os.path.join(ROOT, "benchmarks", "data", dataset),
                       "--out", cell, "--bench-skip-dense"]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True)
                wall_ms = (time.time() - t0) * 1000.0
                if proc.returncode != 0:
                    print(f"[fail] {tag}: rc={proc.returncode} "
                          f"{(proc.stderr or '').strip().splitlines()[-1:] or ''}", flush=True)
                    continue
                stats = summarize(load_rows(cell), wall_ms)
                if stats is None:
                    print(f"[fail] {tag}: no rows", flush=True)
                    continue
                with open(os.path.join(cell, "_stats.json"), "w", encoding="utf-8") as f:
                    json.dump({k2: v for k2, v in stats.items() if k2 != "per_case_q"}, f)
                exists = os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not exists:
                        w.writerow(header.split(","))
                    w.writerow([dataset, arm, rnd] + [stats[key] for key in header.split(",")[3:]])
                print(f"[cell] {tag:<40} wall={wall_ms/1000:6.1f}s ttft={stats['ttft_min']:6.0f} "
                      f"e2e={stats['e2e_median']:7.0f} q={stats['q_mean']:.3f} "
                      f"overlap={stats['overlap_ratio']:.2f} handoff={stats['handoff_ms']:.1f}ms",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
