#!/usr/bin/env python3
"""AR006: assemble the campaign matrix into matrix.csv / matrix.md and gate
quality. Each result dir is one (dataset, mode) pair; the dense reference is
the sibling ``<dataset>-dense`` dir measured in the same session.

Usage: assemble_ar006.py [RESULTS_DIR]

Exit code 0 when every (dataset, bound) point has quality >= MIN_Q, else 1
(the quality gate is what CI should read).
"""
import csv
import glob
import os
import re
import sys

MIN_Q = 0.9
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmarks", "results", "ar006-campaign")
RESULTS = os.path.abspath(RESULTS)

# dir suffix -> display mode
MODES = [("dense", "dense"), ("scalar", "scalar"), ("qgv2", "qg-2d(AR005)"),
         ("boundp16", "bound-p16"), ("boundp24", "bound-p24"),
         ("boundp16-r2", "bound-p16"), ("boundp24-r2", "bound-p24"),
         ("boundp24-l32", "bound-p24+local32")]


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return 0.0
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def load(path):
    """Aggregate one result dir: per-case rows, medians, quality."""
    csvs = sorted(glob.glob(os.path.join(path, "*_raw.csv")))
    if not csvs:
        return None
    rows = []
    with open(csvs[-1]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None
    ttft = [float(r["ttft_ms"]) for r in rows]
    e2e = [float(r["e2e_ms"]) for r in rows]
    quality = [float(r["quality"]) for r in rows]
    return {
        "n": len(rows),
        "ttft_ms": median(ttft),
        "e2e_ms": median(e2e),
        "quality": sum(quality) / len(quality),
        "min_quality": min(quality),
        "qg_source": rows[0].get("qg_source", ""),
        "sem_ms": median([float(r.get("semantic_score_ms") or 0) for r in rows]),
        "repeats": rows[0].get("repeats", ""),
        "selected": median([float(r["selected_tokens"]) for r in rows]),
        "prompt": median([float(r["prompt_tokens"]) for r in rows]),
        "raw": csvs[-1],
    }


def split_suffix(name):
    for suffix, label in MODES:
        if name.endswith("-" + suffix):
            return name[: -len(suffix) - 1], suffix, label
    return None


dirs = {}
for path in sorted(glob.glob(os.path.join(RESULTS, "*"))):
    if not os.path.isdir(path):
        continue
    parsed = split_suffix(os.path.basename(path))
    if not parsed:
        continue
    dataset, suffix, label = parsed
    agg = load(path)
    if agg is None:
        print(f"note: no raw csv in {os.path.basename(path)}")
        continue
    dirs.setdefault(dataset, {})[suffix] = dict(agg, label=label)

rows = []
gate_failures = []
for dataset in sorted(dirs):
    entry = dirs[dataset]
    dense = entry.get("dense")
    dense_ttft = dense["ttft_ms"] if dense else 0.0
    for suffix, agg in sorted(entry.items(), key=lambda kv: kv[1]["label"]):
        if suffix == "dense":
            continue
        speedup = dense_ttft / agg["ttft_ms"] if agg["ttft_ms"] else 0.0
        rows.append({
            "dataset": dataset,
            "mode": agg["label"],
            "prompt_tokens": int(agg["prompt"]),
            "selected_tokens": int(agg["selected"]),
            "ttft_ms": round(agg["ttft_ms"], 1),
            "e2e_ms": round(agg["e2e_ms"], 1),
            "dense_ttft_ms": round(dense_ttft, 1),
            "speedup": round(speedup, 2),
            "quality": round(agg["quality"], 3),
            "min_quality": round(agg["min_quality"], 3),
            "qg_source": agg["qg_source"],
            "semantic_score_ms": round(agg["sem_ms"], 1),
            "repeats": agg["repeats"],
        })
        if agg["label"].startswith("bound") and agg["min_quality"] < MIN_Q:
            gate_failures.append(
                f"{dataset}/{agg['label']}: min quality {agg['min_quality']:.2f} < {MIN_Q}")

out_csv = os.path.join(RESULTS, "matrix.csv")
os.makedirs(RESULTS, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
    writer.writeheader()
    writer.writerows(rows)

out_md = os.path.join(RESULTS, "matrix.md")
with open(out_md, "w") as f:
    f.write("# AR006 campaign matrix\n\n")
    f.write("Net speedup = dense / mode TTFT, both measured in this session.\n")
    f.write("quality is the mean over cases; q < 0.9 on a bound row fails the gate.\n\n")
    f.write("| dataset | mode | tokens | sel | ttft ms | dense ms | speedup | q | src |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| {dataset} | {mode} | {prompt_tokens} | {selected_tokens} | "
                "{ttft_ms} | {dense_ttft_ms} | {speedup}x | {quality} | {qg_source} |\n"
                .format(**r))

print(f"rows={len(rows)} -> {out_csv}, {out_md}")
for r in rows:
    print(f"  {r['dataset']:<22} {r['mode']:<14} ttft={r['ttft_ms']:>9}ms "
          f"speedup={r['speedup']:>5}x q={r['quality']} src={r['qg_source']}")
if gate_failures:
    print("\nQUALITY GATE FAILED:")
    for failure in gate_failures:
        print("  " + failure)
    sys.exit(1)
print("\nquality gate: PASS")
