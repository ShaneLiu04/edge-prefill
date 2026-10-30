#!/usr/bin/env python3
"""AR015 matrix: cold (recompute) vs warm (restore from a KV snapshot).

Two arms per dataset, each run as its own bench process so "warm" really means
"a different process found the snapshot on disk":

  cold  snapshot_dir=D, snapshot_use=0  -> always prefill, always save
  warm  snapshot_dir=D, snapshot_use=1  -> restore if the key matches

`deep/multiturn` additionally runs a resume arm: the conversation is replayed
from scratch in a new process, but every turn restores the KV an earlier
process produced for that exact log.

Both arms share one snapshot directory, which is wiped per dataset unless
--keep-snapshots is passed.
"""
import argparse
import csv
import glob
import os
import shutil
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "scripts"))

DATASETS = [
    "explen/len4k",
    "explen/len8k",
    "explen/len16k",
    "deep/len32k",
    "deep/multineedle",
    "deep/multifact8k",
    "deep/multifact8k-hard",
    "explen/len64k",
]

# The shipped admission (AR012/AR013 defaults) plus the AR014 session knobs for
# the multi-turn dataset; AR015 only adds the snapshot directory.
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

FLAT_DATASETS = {"deep/multifact8k", "deep/multifact8k-hard", "deep/multiturn"}


def write_conf(path, dataset, arm, snapshot_dir, use, threads, repeats):
    budget = 0.05 if dataset in FLAT_DATASETS else None
    with open(path, "w", encoding="utf-8") as f:
        for key, value in BASE_CONF.items():
            f.write(f"{key} = {value}\n")
        if budget is not None:
            f.write(f"budget = {budget}\n")
        f.write(f"threads = {threads}\n")
        f.write("warmup = 0\n")
        f.write(f"max_new_tokens = {48 if dataset in FLAT_DATASETS else 16}\n")
        f.write(f"repeats = {repeats}\n")
        if dataset == "deep/multiturn":
            # cold/warm keep AR014's in-process reuse; `resume` emulates a
            # process that restarts every turn and can only restore from disk.
            reuse = 0 if arm == "resume" else 1
            f.write(f"session_reuse = {reuse}\n")
            f.write("session_ctx_max = 4096\n")
        f.write(f"snapshot_dir = {snapshot_dir}\n")
        f.write(f"snapshot_use = {use}\n")
        # A conversation writes one snapshot per turn (up to ~60 MB each), so the
        # cell needs a budget large enough to keep them all; the library default
        # (512 MiB) is an LRU policy that only retains the most recent turns.
        f.write("snapshot_max_bytes = 4294967296\n")


def run_bench(bin_path, model, dataset, arm, out_dir, conf, extra=()):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [bin_path, "bench", "--config", conf, "--model", model,
           "--data-dir", os.path.join(ROOT, "benchmarks", "data", dataset),
           "--out", out_dir, "--bench-skip-dense", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-5:])
        return f"bench failed ({proc.returncode}): {tail}"
    return None


def load_rows(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*_raw.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f) if r.get("dense") == "0"]
    return rows


def summarize(rows):
    if not rows:
        return None
    ttfts = [float(r["ttft_ms"]) for r in rows]
    mins = [float(r["ttft_min_ms"]) for r in rows if float(r.get("ttft_min_ms", 0)) > 0]
    return {
        "cases": len(rows),
        "q_mean": statistics.mean(float(r["quality"]) for r in rows),
        "q_min": min(float(r["quality"]) for r in rows),
        "ttft_median": statistics.median(ttfts),
        "ttft_min": statistics.median(mins) if mins else statistics.median(ttfts),
        "snapshot_hits": sum(1 for r in rows if r.get("snapshot_hit") == "1"),
        "snapshot_bytes": int(statistics.mean(int(r.get("snapshot_bytes", 0) or 0) for r in rows)),
        "load_ms": statistics.median(
            [float(r.get("snapshot_load_ms", 0) or 0) for r in rows]) if rows else 0.0,
        "save_ms": statistics.median(
            [float(r.get("snapshot_save_ms", 0) or 0) for r in rows]) if rows else 0.0,
        "kv_used": int(statistics.mean(int(r.get("kv_used_bytes", 0) or 0) for r in rows)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="benchmarks/results/ar015-kv-snapshot")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rerun cells that already have raw CSVs")
    ap.add_argument("--keep-snapshots", action="store_true")
    args = ap.parse_args(argv)

    datasets = [d for d in args.datasets.split(",") if d]
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    snap_root = os.path.join(out, "snapshots")

    cells = []
    per_case = {}
    for dataset in datasets:
        dataset_snap = os.path.join(snap_root, dataset.replace("/", "_"))
        conf_dir = os.path.join(out, "_conf")
        os.makedirs(conf_dir, exist_ok=True)
        tag_base = dataset.replace("/", "-")
        arms = ["cold", "warm"]
        if dataset == "deep/multiturn":
            arms.append("resume")
        for arm in arms:
            tag = f"{tag_base}-{arm}"
            cell_dir = os.path.join(out, tag)
            if not args.force and os.path.isdir(cell_dir) and glob.glob(
                    os.path.join(cell_dir, "*_raw.csv")):
                print(f"[skip] {tag}", flush=True)
                rows = load_rows(cell_dir)
            else:
                if arm == "cold" and os.path.isdir(dataset_snap) and not args.keep_snapshots:
                    shutil.rmtree(dataset_snap)
                os.makedirs(dataset_snap, exist_ok=True)
                conf = os.path.join(conf_dir, f"{tag}.conf")
                use = 0 if arm == "cold" else 1
                write_conf(conf, dataset, arm, dataset_snap, use, args.threads,
                           args.repeats)
                extra = []
                if args.gpu_layers:
                    extra = ["--n-gpu-layers", str(args.gpu_layers)]
                err = run_bench(os.path.join(ROOT, args.bin), args.model, dataset,
                                arm, cell_dir, conf, extra)
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
                rows = load_rows(cell_dir)
            stats = summarize(rows)
            if stats is None:
                print(f"[fail] {tag}: no sparse samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, **stats})
            per_case[(dataset, arm)] = {r["case"]: r for r in rows}
            print(f"[cell] {tag:<24} cases={stats['cases']:3d} "
                  f"q={stats['q_mean']:.3f} ttft_min={stats['ttft_min']:9.1f}ms "
                  f"hits={stats['snapshot_hits']:3d}/{stats['cases']:3d} "
                  f"load={stats['load_ms']:6.1f}ms bytes={stats['snapshot_bytes']:9d}",
                  flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median", "ttft_min",
              "snapshot_hits", "snapshot_bytes", "load_ms", "save_ms", "kv_used"]
    matrix_path = os.path.join(out, "matrix.csv")
    fresh = {c["dataset"] for c in cells}
    if os.path.exists(matrix_path):
        with open(matrix_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["dataset"] not in fresh:
                    cells.append({k: r.get(k, "") for k in fields})
    cells.sort(key=lambda c: (c["dataset"], c["arm"]))
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k in fields})

    # per-case table for the gate: quality equality and hit accounting
    case_path = os.path.join(out, "cases.csv")
    rows_out = []
    for (dataset, arm), cases in per_case.items():
        for case, r in cases.items():
            rows_out.append({
                "dataset": dataset, "arm": arm, "case": case,
                "quality": round(float(r["quality"]), 6),
                "ttft_min_ms": round(float(r.get("ttft_min_ms", r["ttft_ms"])), 1),
                "snapshot_hit": r.get("snapshot_hit", "0"),
                "snapshot_state": r.get("snapshot_state", ""),
                "snapshot_bytes": r.get("snapshot_bytes", "0"),
                "snapshot_load_ms": r.get("snapshot_load_ms", "0"),
                "kv_used_bytes": r.get("kv_used_bytes", "0"),
                "session_log_hash": r.get("session_log_hash", ""),
                "reused_tokens": r.get("reused_tokens", "0"),
                "pipeline": r.get("pipeline", ""),
            })
    fresh_case = {(r["dataset"], r["arm"]) for r in rows_out}
    if os.path.exists(case_path):
        with open(case_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r["dataset"], r["arm"]) not in fresh_case:
                    rows_out.append({k: r.get(k, "") for k in rows_out[0]})
    rows_out.sort(key=lambda r: (r["dataset"], r["arm"], r["case"]))
    with open(case_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", matrix_path, "and", case_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
