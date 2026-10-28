#!/usr/bin/env python3
"""AR011 FR-4: measure the combination matrix after structure-bounded admission.

Arms (all real GGUF, no mocks):

  ar010          AR010 semantics: curve budget, evidence-gated admission,
                 content-sized context (fill_priors=0), whole blocks
  ar011          AR011 all on: structure_budget=64, anchor radius 1,
                 coverage reserve, span admission (24-token spans)
  ar011-curve    AR011 with the anchor radius but no span/reserve (isolates FR-1)
  ar011-span     span admission only (isolates FR-3)
  gate-p16       AR006 in-graph page gating (comparison arm for the table)
  dense          full prefill reference (measured once per dataset)

Datasets use the AR007 budget curve; the multi-fact families need a flat 0.05
budget (the curve at 8K starves them), exactly as in the AR009/AR010 campaigns.

Outputs:
  <out>/<dataset>-<arm>/{bench.conf,*_raw.csv}
  <out>/matrix.csv  dataset,arm,cases,q_mean,q_min,ttft_median,speedup,selected,...audit
  <out>/matrix.md
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
    "ar010": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1},
    "ar011": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
              "structure_budget": 64, "span_admission": 1, "span_min_tokens": 24},
    "ar011-all": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                  "structure_budget": 64, "anchor_span_radius": 1,
                  "coverage_reserve": 1, "span_admission": 1,
                  "span_min_tokens": 24},
    "ar011-struct": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                     "structure_budget": 64},
    "ar011-agg": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                  "structure_budget": 64, "anchor_span_radius": 1,
                  "span_admission": 1, "span_min_tokens": 24},
    "ar011-span": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                   "span_admission": 1, "span_min_tokens": 24},
    "sb64r2": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
               "structure_budget": 64, "anchor_span_radius": 2},
    "sb96r1": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
               "structure_budget": 96, "anchor_span_radius": 1},
    "sb96r2": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
               "structure_budget": 96},
    "sb48r2": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
               "structure_budget": 48},
    "sb64r2res": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                  "structure_budget": 64, "coverage_reserve": 1},
    "sb64r2span": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                   "structure_budget": 64, "span_admission": 1,
                   "span_min_tokens": 24},
    "gate-p16": {"mode": "balanced", "fill_priors": 0, "prior_gate": 1,
                 "qg_bound": "true", "qg_sel_pages": 16, "qg_local_rows": 0,
                 "budget": 0.25},
    "dense": {"mode": "dense", "fill_priors": 1, "prior_gate": 1, "budget": 1.0},
}

# Datasets that need a flat budget instead of the curve (multi-fact families).
FLAT_BUDGET = {
    "deep/multifact8k": 0.05,
    "deep/multifact8k-hard": 0.05,
}

# multi-fact answers list several facts, so the AR009/AR010 campaigns used 48
# new tokens for these families and 16 elsewhere. Using 16 here silently
# under-scored them (0.576 instead of 0.836 on multifact8k).
NEW_TOKENS = {
    "deep/multifact8k": 48,
    "deep/multifact8k-hard": 48,
}


def write_conf(path, arm, curve, threads, max_new_tokens, phrase_weight,
               coverage_weight, flat_budget=None):
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
        f"threads = {threads}",
        "warmup = 0",
        f"max_new_tokens = {max_new_tokens}",
        "repeats = 1",
    ]
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
        return None, f"rc={proc.returncode} wall={wall:.0f}s {proc.stderr[-200:]}"
    rows = []
    for f in sorted(glob.glob(os.path.join(out_dir, "*_raw.csv"))):
        with open(f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        return None, f"no samples wall={wall:.0f}s"
    return rows, None


def audit_mean(rows, field):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[field]))
        except (KeyError, TypeError, ValueError):
            pass
    return statistics.mean(vals) if vals else 0.0


def summarize(rows, dense=False):
    want = [r for r in rows if int(r["dense"]) == (1 if dense else 0)]
    if not want:
        return None
    q = sorted(float(r["quality"]) for r in want)
    t = [float(r["ttft_ms"]) for r in want]
    return {
        "cases": len(want),
        "q_mean": statistics.mean(q),
        "q_min": min(q),
        "ttft_median": statistics.median(t),
        "selected": int(statistics.mean(int(r["selected_tokens"]) for r in want)),
        "structure": int(statistics.mean(audit_mean([r], "structure_tokens") for r in want)),
        "evidence": int(statistics.mean(audit_mean([r], "evidence_tokens") for r in want)),
        "span_trimmed": int(statistics.mean(audit_mean([r], "span_tokens") for r in want)),
        "uncovered": statistics.mean(audit_mean([r], "uncovered_facts") for r in want),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build-clean/edge-prefill")
    ap.add_argument("--model", default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k,explen/len16k,"
                                          "deep/len32k,deep/multineedle,"
                                          "deep/multifact8k,deep/multifact8k-hard")
    ap.add_argument("--arms", default="ar010,ar011,ar011-struct,gate-p16,dense")
    ap.add_argument("--curve", default="benchmarks/results/ar007-sweep/budget_curve.txt")
    ap.add_argument("--out", default="benchmarks/results/ar011-structure-bounded")
    ap.add_argument("--phrase-weight", type=float, default=0.5)
    ap.add_argument("--coverage-weight", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
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
            existing = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")))
            if existing and not args.force:
                rows = []
                for f in existing:
                    with open(f, newline="", encoding="utf-8") as fh:
                        rows.extend(csv.DictReader(fh))
            else:
                if args.force:
                    # A rerun appends a new timestamped CSV; without clearing
                    # the old ones the summary silently averaged two different
                    # configurations (found while comparing ar011 vs ar011-all).
                    for stale in glob.glob(os.path.join(cell_dir, "*_raw.csv")):
                        os.remove(stale)
                    for stale in glob.glob(os.path.join(cell_dir, "*_env.json")):
                        os.remove(stale)
                    for stale in glob.glob(os.path.join(cell_dir, "*_report.md")):
                        os.remove(stale)
                write_conf(conf, arm, os.path.join(ROOT, curve) if curve else "",
                           args.threads, new_tokens, args.phrase_weight,
                           args.coverage_weight, flat)
                rows, err = run_cell(bin_path, conf, model, data_dir, cell_dir,
                                     skip_dense=(arm != "dense"))
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
            stats = summarize(rows, dense=(arm == "dense"))
            if stats is None:
                print(f"[fail] {tag}: no matching samples", flush=True)
                continue
            cells.append({"dataset": dataset, "arm": arm, "tag": tag, **stats})
            print(f"[cell] {tag:<34} q={stats['q_mean']:.3f} min={stats['q_min']:.3f} "
                  f"sel={stats['selected']:5d} struct={stats['structure']:4d} "
                  f"span={stats['span_trimmed']:4d} unc={stats['uncovered']:.1f} "
                  f"ttft={stats['ttft_median']:8.0f}ms", flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    dense = {c["dataset"]: c for c in cells if c["arm"] == "dense"}
    fields = ["dataset", "arm", "cases", "q_mean", "q_min", "ttft_median", "speedup",
              "selected", "structure", "evidence", "span_trimmed", "uncovered"]
    with open(os.path.join(out, "matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c["dataset"], c["arm"])):
            d = dense.get(c["dataset"])
            w.writerow({
                "dataset": c["dataset"], "arm": c["arm"], "cases": c["cases"],
                "q_mean": round(c["q_mean"], 4), "q_min": round(c["q_min"], 4),
                "ttft_median": round(c["ttft_median"], 1),
                "speedup": round(d["ttft_median"] / c["ttft_median"], 2)
                if d and c["arm"] != "dense" else "",
                "selected": c["selected"], "structure": c["structure"],
                "evidence": c["evidence"], "span_trimmed": c["span_trimmed"],
                "uncovered": round(c["uncovered"], 2),
            })
    print("wrote", os.path.join(out, "matrix.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
