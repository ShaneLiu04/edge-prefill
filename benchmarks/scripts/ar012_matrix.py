#!/usr/bin/env python3
"""AR012: sentence-complete admission matrix.

Arms:
  ar011        AR011 baseline (structure budget + span, whole blocks)
  ar012        AR012: sentence completion, union mode (add the missing
               continuations, remove nothing)
  ar012-trim   AR012 + drop tokens outside the admitted sentences
  gate-p16     AR006 in-graph page gating (comparison arm)

Datasets and per-dataset budget / answer length follow ar011_matrix.py (the
multi-fact families need a flat 0.05 budget and 48 answer tokens).

Outputs mirror ar011: <out>/<dataset>-<arm>/{bench.conf,*_raw.csv} + matrix.csv.
"""
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
    "ar011": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
              "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24},
    "ar012": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
              "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24,
              "sentence_completion": 1, "sentence_trim": 0},
    "ar012-trim": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                   "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24,
                   "sentence_completion": 1, "sentence_trim": 1},
    "gate-p16": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                 "qg_bound": "true", "qg_sel_pages": 16, "qg_local_rows": 0,
                 "budget": 0.25},
}

FLAT_BUDGET = {"deep/multifact8k": 0.05, "deep/multifact8k-hard": 0.05}
NEW_TOKENS = {"deep/multifact8k": 48, "deep/multifact8k-hard": 48}


def write_conf(path, arm, curve, threads, max_new_tokens, phrase_weight,
               coverage_weight, flat_budget, dump_dir=""):
    spec = ARMS[arm]
    lines = [f"mode = {spec['mode']}"]
    if flat_budget is not None:
        lines.append(f"budget = {flat_budget}")
    elif spec.get("budget") is not None:
        lines.append(f"budget = {spec['budget']}")
    elif curve:
        lines.append(f"budget_curve = {curve}")
    lines += [
        "attn_top_k = 24",
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
        f"threads = {threads}",
        "warmup = 0",
        f"max_new_tokens = {max_new_tokens}",
        "repeats = 1",
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
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def summarize(rows):
    want = [r for r in rows if int(r["dense"]) == 0]
    if not want:
        return None
    q = sorted(float(r["quality"]) for r in want)
    t = [float(r["ttft_ms"]) for r in want]
    aud = lambda f: statistics.mean(float(r[f]) for r in want) if f in want[0] else 0.0
    return {
        "cases": len(want),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "ttft_median": statistics.median(t),
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in want)),
        "span_trimmed": int(statistics.mean(aud("span_tokens") for _ in [0])),
        "sentences": aud("sentences_completed"),
        "extended": aud("sentences_extended"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build-clean/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k,explen/len16k,"
                                          "deep/len32k,deep/multineedle,"
                                          "deep/multifact8k,deep/multifact8k-hard")
    ap.add_argument("--arms", default="ar011,ar012,ar012-trim,gate-p16")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar012-sentence-complete")
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--coverage-weight", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
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
                           args.coverage_weight, flat, dump_dir)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir, True)
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
            stats = summarize(rows)
            if stats is None:
                print(f"[fail] {tag}: no sparse samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, "tag": tag, **stats})
            print(f"[cell] {tag:<30} q={stats['q_mean']:.3f} min={stats['q_min']:.3f} "
                  f"sel={stats['selected']:5d} sent={stats['sentences']:5.1f} "
                  f"ext={stats['extended']:4.1f} ttft={stats['ttft_median']:7.0f}ms",
                  flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median",
                  "selected", "span_trimmed", "sentences", "extended"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items() if k in fields})
    print("wrote", os.path.join(out, "matrix.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
