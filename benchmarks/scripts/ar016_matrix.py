#!/usr/bin/env python3
"""AR016 matrix: execution plan + latency-targeted budget.

Arms (each its own bench process, sparse only, cold snapshots):

  base16  threads = hardware_concurrency (the old implicit behaviour), no SLA
  plan    threads from the probed plan, no SLA
  sla     probed plan + budget_target_ms solved with the probed cost model

The cost model is taken from a separate `probe` run (`--plan-dir`), never from
the data being optimised. `--gpu-layers` switches every arm to the CUDA build's
placement; the audit columns prove the offload really happened.
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
    path = os.path.join(plan_dir, "plan.conf")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            plan[key.strip()] = value.strip()
    return plan


def write_conf(path, dataset, threads, repeats, warmup, plan=None,
               budget_target_ms=None, gpu_layers=0, max_new_tokens=48):
    with open(path, "w", encoding="utf-8") as f:
        for key, value in BASE_CONF.items():
            f.write(f"{key} = {value}\n")
        if dataset in FLAT_DATASETS:
            f.write("budget = 0.05\n")
        f.write(f"threads = {threads}\n")
        f.write(f"warmup = {warmup}\n")
        f.write(f"repeats = {repeats}\n")
        # 48 matches the AR012-AR015 datasets (multi-fact answers list
        # several facts; 16 tokens truncates them and tanks f1).
        f.write(f"max_new_tokens = {max_new_tokens}\n")
        f.write("snapshot_use = 0\n")
        if gpu_layers:
            f.write(f"n_gpu_layers = {gpu_layers}\n")
        if plan is not None:
            for key in ("cost_a_ms", "cost_b_ms_per_token", "cost_r2", "cost_n"):
                if key in plan:
                    f.write(f"{key} = {plan[key]}\n")
            if budget_target_ms:
                f.write(f"budget_target_ms = {budget_target_ms}\n")


def run_bench(bin_path, model, dataset, out_dir, conf):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [bin_path, "bench", "--config", conf, "--model", model,
           "--data-dir", os.path.join(ROOT, "benchmarks", "data", dataset),
           "--out", out_dir, "--bench-skip-dense"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-4:])
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
    mins = [float(r["ttft_min_ms"]) for r in rows if float(r.get("ttft_min_ms", 0) or 0) > 0]
    return {
        "cases": len(rows),
        "q_mean": statistics.mean(float(r["quality"]) for r in rows),
        "q_min": min(float(r["quality"]) for r in rows),
        "ttft_median": statistics.median(float(r["ttft_ms"]) for r in rows),
        "ttft_min": statistics.median(mins) if mins else 0.0,
        "sel_tokens": statistics.median(float(r["selected_tokens"]) for r in rows),
        "n_threads": int(statistics.median(int(r["n_threads"]) for r in rows)),
        "n_ubatch": int(statistics.median(int(r["n_ubatch"]) for r in rows)),
        "device": rows[0].get("device", ""),
        "device_state": rows[0].get("device_state", ""),
        "offloaded": int(rows[0].get("gpu_layers_offloaded", -1) or -1),
        "requested": int(rows[0].get("gpu_layers_requested", 0) or 0),
        "plan_source": rows[0].get("plan_source", ""),
        "budget_source": rows[0].get("budget_source", ""),
        "predicted": statistics.median(
            float(r.get("predicted_ttft_ms", 0) or 0) for r in rows),
        "plan_select_ms": statistics.median(
            float(r.get("plan_select_ms", 0) or 0) for r in rows),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="benchmarks/results/ar016-exec-plan")
    ap.add_argument("--plan-dir", required=True,
                    help="directory produced by `edge-prefill probe`")
    ap.add_argument("--base-threads", type=int, default=16)
    ap.add_argument("--plan-threads", type=int, default=0,
                    help="0 = take threads from plan.conf")
    ap.add_argument("--sla-target-ms", type=float, default=2500.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--arms", default="base16,plan,sla")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="keep rows for cells not measured in this run")
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args(argv)

    datasets = [d for d in args.datasets.split(",") if d]
    arms = [a for a in args.arms.split(",") if a]
    out = os.path.join(ROOT, args.out)
    conf_dir = os.path.join(out, "_conf")
    os.makedirs(conf_dir, exist_ok=True)
    plan = read_plan(os.path.join(ROOT, args.plan_dir))
    plan_threads = args.plan_threads or int(plan.get("threads", 0) or 0)
    if plan_threads <= 0:
        sys.stderr.write("plan has no threads; pass --plan-threads\n")
        return 2

    cells = []
    per_case = {}
    for dataset in datasets:
        tag_base = dataset.replace("/", "-")
        for arm in arms:
            tag = f"{tag_base}-{arm}{args.tag_suffix}"
            cell_dir = os.path.join(out, tag)
            if not args.force and glob.glob(os.path.join(cell_dir, "*_raw.csv")):
                print(f"[skip] {tag}", flush=True)
                rows = load_rows(cell_dir)
            else:
                conf = os.path.join(conf_dir, f"{tag}.conf")
                if arm == "base16":
                    write_conf(conf, dataset, args.base_threads, args.repeats,
                               args.warmup, gpu_layers=args.gpu_layers,
                               max_new_tokens=args.max_new_tokens)
                elif arm == "plan":
                    write_conf(conf, dataset, plan_threads, args.repeats,
                               args.warmup, gpu_layers=args.gpu_layers)
                elif arm == "sla":
                    write_conf(conf, dataset, plan_threads, args.repeats,
                               args.warmup, plan=plan,
                               budget_target_ms=args.sla_target_ms,
                               gpu_layers=args.gpu_layers)
                else:
                    sys.stderr.write(f"unknown arm {arm}\n")
                    return 2
                err = run_bench(os.path.join(ROOT, args.bin), args.model, dataset,
                                cell_dir, conf)
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
            print(f"[cell] {tag:<30} cases={stats['cases']:3d} q={stats['q_mean']:.3f} "
                  f"ttft_min={stats['ttft_min']:8.1f}ms sel={stats['sel_tokens']:6.0f} "
                  f"thr={stats['n_threads']} src={stats['budget_source']} "
                  f"dev={stats['device_state']}/{stats['offloaded']}", flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    matrix_path = os.path.join(out, "matrix.csv")
    if args.merge and os.path.exists(matrix_path):
        with open(matrix_path, newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        fresh = {(c["dataset"], c["arm"]) for c in cells}
        cells = [r for r in old_rows if (r["dataset"], r["arm"]) not in fresh] + cells
        cells.sort(key=lambda r: (r["dataset"], r["arm"]))
    fields = list(cells[0].keys())
    with open(matrix_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cells)

    # per-case table for the gate (quality equality + SLA prediction accuracy)
    cases_path = os.path.join(out, "cases.csv")
    if args.merge and os.path.exists(cases_path):
        with open(cases_path, newline="", encoding="utf-8") as f:
            keep = [r for r in csv.DictReader(f)
                    if (r["dataset"], r["arm"]) not in fresh]
    else:
        keep = []
    with open(cases_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "case", "arm", "quality", "ttft_ms", "ttft_min_ms",
                         "selected_tokens", "predicted_ttft_ms", "budget_source",
                         "n_threads", "device_state"])
        for r in keep:
            writer.writerow([r[k] for k in ("dataset", "case", "arm", "quality", "ttft_ms",
                                            "ttft_min_ms", "selected_tokens",
                                            "predicted_ttft_ms", "budget_source",
                                            "n_threads", "device_state")])
        for (dataset, arm), rows in sorted(per_case.items()):
            for case, r in sorted(rows.items()):
                writer.writerow([dataset, case, arm, r["quality"], r["ttft_ms"],
                                 r["ttft_min_ms"], r["selected_tokens"],
                                 r.get("predicted_ttft_ms", 0), r.get("budget_source", ""),
                                 r.get("n_threads", ""), r.get("device_state", "")])
    print(f"wrote {os.path.join(out, 'matrix.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
