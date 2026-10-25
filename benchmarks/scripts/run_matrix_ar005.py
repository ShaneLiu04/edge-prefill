#!/usr/bin/env python3
"""AR005 T007: run the FR-6 benchmark matrix on CPU/GPU, then assemble
benchmarks/results/<out>/matrix.csv (+ matrix.md).

Design (see specs/changes/AR005-.../design.md, FR-6):
- One shared dense reference per dataset (dense is config-independent),
  measured once with --bench-skip-dense; every sparse grid point of the
  same dataset pairs against it. This replaces 24 in-run dense baselines
  (~15h) with 5 reference runs.
- Every sparse run passes --embedding-model: the embedding scorer is
  active for all grid points (semantic selection); the qg dimension only
  toggles the attention-side gather (2D query-gated vs scalar top-k).
- Net speedup = dense_ttft / (sparse_ttft + semantic_score_ms): the
  selection-stage scoring is charged to the sparse path (FR-4/FR-5 net
  accounting).
- Idempotent: a run dir that already contains *_raw.csv is skipped, so
  an interrupted matrix resumes where it stopped.

Usage (WSL):
  python3 benchmarks/scripts/run_matrix_ar005.py \
      --bin build-wsl/edge-prefill \
      --model /tmp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
      --embedding /tmp/models/bge-small-en-v1.5-q8_0.gguf
  --smoke          : len4k only, 1 repeat, 3 confs (plumbing check)
  --datasets A,B   : restrict to a subset (default: all FR-6 datasets)
  --confs A,B      : restrict sparse confs (default: FR-6 full grid per
                     dataset; 4K/8K/mn use their acceptance confs)
  --dense-repeats N: repeats for the dense reference (default: --repeats)
  --n-gpu-layers N : pass through to bench for CUDA builds (default: omit)
  --assemble-only  : skip runs, just rebuild matrix.csv/.md
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONF_DIR = os.path.join(ROOT, "benchmarks", "matrix")

SCALAR_ACCEPT = "b025-a24-pool0-qgoff.conf"
QG_ACCEPT = "b025-a24-pool0-qgv2.conf"

# dataset key -> (data_dir under --data-root, sparse conf list).
# len16k/len32k: dense + all 24 FR-6 grid points.
# len4k/len8k: dense + the scalar no-pool acceptance config (S12: q>=0.9).
# mn8k: dense + the qg-v2 acceptance config (S12: mean q>=0.66).
FULL_GRID = sorted(
    f for f in os.listdir(CONF_DIR)
    if f.endswith(".conf") and f != "dense.conf"
) if os.path.isdir(CONF_DIR) else []

DATASETS = [
    ("len16k", "explen/len16k", ["dense.conf"] + FULL_GRID),
    ("len32k", "deep/len32k", ["dense.conf"] + FULL_GRID),
    ("len4k", "explen/len4k", ["dense.conf", SCALAR_ACCEPT]),
    ("len8k", "explen/len8k", ["dense.conf", SCALAR_ACCEPT]),
    ("mn8k", "deep/multineedle", ["dense.conf", QG_ACCEPT]),
]

SMOKE_DATASETS = [
    ("len4k", "explen/len4k", ["dense.conf", SCALAR_ACCEPT, QG_ACCEPT]),
]


def latest_raw(run_dir):
    csvs = sorted(glob.glob(os.path.join(run_dir, "*_raw.csv")))
    return csvs[-1] if csvs else None


def run_one(args, dataset, data_dir, conf_name, out_root, repeats):
    tag = conf_name.replace(".conf", "")
    run_dir = os.path.join(out_root, "%s-%s" % (dataset, tag))
    os.makedirs(run_dir, exist_ok=True)
    if latest_raw(run_dir):
        print("[skip] %s/%s (raw.csv exists)" % (dataset, tag), flush=True)
        return True
    cmd = [
        args.bin,
        "bench",
        "--config", os.path.join(CONF_DIR, conf_name),
        "--model", args.model,
        "--data-dir", data_dir,
        "--out", run_dir,
        "--bench-skip-dense",
        "--repeats", str(repeats),
    ]
    if tag != "dense":
        cmd += ["--embedding-model", args.embedding]
    if args.n_gpu_layers > 0:
        cmd += ["--n-gpu-layers", str(args.n_gpu_layers)]
    print("[run ] %s/%s (repeats=%d)" % (dataset, tag, repeats), flush=True)
    t0 = time.time()
    with open(os.path.join(run_dir, "runner.log"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if rc != 0 or not latest_raw(run_dir):
        print("[FAIL] %s/%s rc=%d after %.0fs - see %s"
              % (dataset, tag, rc, dt, os.path.join(run_dir, "runner.log")),
              flush=True)
        return False
    print("[ok  ] %s/%s %.0fs" % (dataset, tag, dt), flush=True)
    return True


def read_rows(run_dir):
    path = latest_raw(run_dir)
    if not path:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def assemble(out_root):
    matrix_csv = os.path.join(out_root, "matrix.csv")
    matrix_md = os.path.join(out_root, "matrix.md")
    rows_out = []
    for dataset, _, confs in DATASETS:
        dense_dir = os.path.join(out_root, "%s-dense" % dataset)
        dense_ttft = {}
        for r in read_rows(dense_dir):
            if r.get("dense") == "1" or r.get("mode") == "dense":
                dense_ttft[r["case"]] = fnum(r, "ttft_ms")
        if not dense_ttft:
            print("[warn] no dense reference for %s" % dataset, flush=True)
        for conf_name in confs:
            tag = conf_name.replace(".conf", "")
            run_dir = os.path.join(out_root, "%s-%s" % (dataset, tag))
            sparse = [r for r in read_rows(run_dir)
                      if r.get("dense") != "1" and r.get("mode") != "dense"]
            if not sparse:
                continue
            speeds, quals, sem_ms, sel_ratio = [], [], [], []
            for r in sparse:
                case = r["case"]
                ttft = fnum(r, "ttft_ms")
                sem = fnum(r, "semantic_score_ms")
                q = fnum(r, "quality")
                net = ttft + sem
                if case in dense_ttft and net > 0:
                    speeds.append(dense_ttft[case] / net)
                quals.append(q)
                sem_ms.append(sem)
                pt = fnum(r, "prompt_tokens")
                if pt > 0:
                    sel_ratio.append(fnum(r, "selected_tokens") / pt)
            row = {
                "dataset": dataset,
                "conf": tag,
                "cases": len(sparse),
                "budget": sparse[0].get("budget", ""),
                "attn_top_k": sparse[0].get("attn_top_k", ""),
                "query_gating": sparse[0].get("query_gating", ""),
                "qg_source": sparse[0].get("qg_source", ""),
                "score_source": sparse[0].get("score_source", ""),
                "sel_ratio_mean": round(sum(sel_ratio) / len(sel_ratio), 4) if sel_ratio else 0.0,
                "quality_mean": round(sum(quals) / len(quals), 4),
                "quality_min": round(min(quals), 4),
                "semantic_ms_mean": round(sum(sem_ms) / len(sem_ms), 1),
                "ttft_sparse_ms_mean": round(
                    sum(fnum(r, "ttft_ms") for r in sparse) / len(sparse), 1),
                "speedup_net_mean": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
                "speedup_net_min": round(min(speeds), 3) if speeds else 0.0,
            }
            rows_out.append(row)

    with open(matrix_csv, "w", newline="") as f:
        if rows_out:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)

    with open(matrix_md, "w") as f:
        f.write("# AR005 T007 FR-6 matrix\n\n")
        f.write("Net speedup = dense_ttft / (sparse_ttft + semantic_score_ms)"
                " - selection-stage scoring is charged to the sparse path.\n\n")
        f.write("| dataset | conf | q_mean | q_min | sel_ratio | sem_ms | "
                "ttft_sparse_ms | speedup_net_mean | speedup_net_min |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows_out:
            f.write("| %s | %s | %.3f | %.3f | %.3f | %.1f | %.1f | %.2f | %.2f |\n" % (
                r["dataset"], r["conf"], r["quality_mean"], r["quality_min"],
                r["sel_ratio_mean"], r["semantic_ms_mean"],
                r["ttft_sparse_ms_mean"], r["speedup_net_mean"],
                r["speedup_net_min"]))
    print("assembled %d rows -> %s" % (len(rows_out), matrix_csv), flush=True)


def select_datasets(want, table):
    if not want:
        return table
    keys = [k.strip() for k in want.split(",") if k.strip()]
    out = []
    for key in keys:
        match = [t for t in table if t[0] == key]
        if not match:
            sys.exit("unknown dataset: %s (known: %s)"
                     % (key, ",".join(t[0] for t in table)))
        out.extend(match)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "build-wsl", "edge-prefill"))
    ap.add_argument("--model",
                    default="/tmp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--embedding",
                    default="/tmp/models/bge-small-en-v1.5-q8_0.gguf")
    ap.add_argument("--data-root",
                    default=os.path.join(ROOT, "benchmarks", "data"))
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "benchmarks", "results",
                                         "ar005-t007-matrix"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dense-repeats", type=int, default=0,
                    help="dense reference repeats (default: --repeats)")
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--datasets", default="",
                    help="comma list restricting datasets (default: all)")
    ap.add_argument("--confs", default="",
                    help="comma list restricting sparse confs (default: FR-6)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    if not args.dense_repeats:
        args.dense_repeats = args.repeats

    if args.assemble_only:
        assemble(args.out)
        return

    if not os.path.isfile(args.bin):
        sys.exit("binary not found: %s" % args.bin)
    if not os.path.isfile(args.model):
        sys.exit("model not found: %s" % args.model)
    if not os.path.isfile(args.embedding):
        sys.exit("embedding model not found: %s" % args.embedding)

    base_table = SMOKE_DATASETS if args.smoke else DATASETS
    datasets = select_datasets(args.datasets, base_table)

    conf_filter = None
    if args.confs:
        conf_filter = set(
            c if c.endswith(".conf") else c + ".conf"
            for c in args.confs.split(",") if c.strip())

    plan = []
    os.makedirs(args.out, exist_ok=True)
    failed = []
    for dataset, sub, confs in datasets:
        data_dir = os.path.join(args.data_root, sub)
        if not os.path.isdir(data_dir):
            print("[FAIL] dataset dir missing: %s" % data_dir, flush=True)
            failed.append(dataset)
            continue
        run_confs = [c for c in confs
                     if c == "dense.conf" or not conf_filter or c in conf_filter]
        for conf_name in run_confs:
            reps = args.dense_repeats if conf_name == "dense.conf" else args.repeats
            plan.append((dataset, conf_name, reps))
            if not run_one(args, dataset, data_dir, conf_name, args.out, reps):
                failed.append("%s/%s" % (dataset, conf_name))

    with open(os.path.join(args.out, "_plan.json"), "w") as f:
        json.dump({"argv": sys.argv,
                   "runs": ["%s/%s" % (d, c) for d, c, _ in plan]}, f, indent=1)

    if failed:
        print("FAILED runs: %s" % failed, flush=True)
        sys.exit(1)
    assemble(args.out)


if __name__ == "__main__":
    main()
