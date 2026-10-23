#!/usr/bin/env python3
"""AR004 T015: assemble the main experiment matrix from all AR004 result
dirs into benchmarks/results/ar004-t015-matrix/matrix.csv (+ .md summary).

Device is inferred from the source directory (ar004-t012-gpu and
*-gpu result dirs are GPU runs; everything else is CPU).
Re-runnable: skips missing dirs with a note.
"""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
OUT = os.path.join(RESULTS, "ar004-t015-matrix")
os.makedirs(OUT, exist_ok=True)

# source_dir -> (device, config_label)
SOURCES = [
    ("ar004-t002-final", "CPU", None),      # config from subdir name
    ("ar004-t006-t009", "CPU", None),
    ("ar004-t012-gpu", "GPU", None),
    ("ar004-t015-matrix", None, None),      # device from subdir suffix
]

def classify(sub):
    """lenXX-config -> (length, config)."""
    cfg = sub
    length = ""
    if sub.startswith("len"):
        parts = sub.split("-", 1)
        length = parts[0].replace("len", "")
        cfg = parts[1] if len(parts) > 1 else ""
    label = cfg.replace("attn24", "global-attn24")
    return length, label

def device_of(base, sub):
    if base == "ar004-t012-gpu" or sub.endswith("-gpu"):
        return "GPU"
    if base == "ar004-t015-matrix" and sub.endswith("gpu"):
        return "GPU"
    return "CPU"

rows = []
for base, dev, _ in SOURCES:
    for sub in sorted(glob.glob(os.path.join(RESULTS, base, "len*"))) + \
                 sorted(glob.glob(os.path.join(RESULTS, base, "mn-*"))):
        if not os.path.isdir(sub):
            continue
        csvs = sorted(glob.glob(os.path.join(sub, "*_raw.csv")))
        if not csvs:
            continue
        with open(csvs[-1]) as f:
            for r in csv.DictReader(f):
                if r.get("dense") == "1":
                    continue
                length, label = classify(os.path.basename(sub))
                device = dev or device_of(base, os.path.basename(sub))
                rows.append({
                    "length": length or r["case"].split("@")[-1],
                    "case": r["case"],
                    "config": label,
                    "device": device,
                    "ttft_sparse_ms": float(r["ttft_ms"]),
                    "quality": float(r["quality"]),
                    "prompt_tokens": int(r["prompt_tokens"]),
                    "selected_tokens": int(r["selected_tokens"]),
                    "attn_top_k": int(r["attn_top_k"]),
                    "query_gating": r.get("query_gating", "0"),
                    "qg_groups": r.get("qg_groups", "0"),
                    "repeats": r["repeats"],
                    "source": os.path.relpath(sub, RESULTS),
                })

# dense references per (source-run, case) come from the dense rows of the
# same raw.csv; keep them for speedup computation. Keyed by the full source
# path: identical subdir basenames across CPU/GPU result dirs must not
# collide.
dense_ref = {}
for base, dev, _ in SOURCES:
    for sub in sorted(glob.glob(os.path.join(RESULTS, base, "*"))):
        if not os.path.isdir(sub):
            continue
        csvs = sorted(glob.glob(os.path.join(sub, "*_raw.csv")))
        if not csvs:
            continue
        with open(csvs[-1]) as f:
            for r in csv.DictReader(f):
                if r.get("dense") == "1":
                    dense_ref[(os.path.relpath(sub, RESULTS), r["case"])] = float(r["ttft_ms"])

for row in rows:
    d = dense_ref.get((row["source"], row["case"]))
    row["ttft_dense_ms"] = d if d is not None else ""
    row["speedup"] = round(d / row["ttft_sparse_ms"], 2) if d else ""

fields = ["length", "case", "config", "device", "ttft_dense_ms",
          "ttft_sparse_ms", "speedup", "quality", "prompt_tokens",
          "selected_tokens", "attn_top_k", "query_gating", "qg_groups",
          "repeats", "source"]
out_csv = os.path.join(OUT, "matrix.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"matrix rows: {len(rows)} -> {out_csv}")

# markdown summary (medians per length/config/device over cases)
import statistics
groups = {}
for row in rows:
    key = (row["length"], row["config"], row["device"])
    groups.setdefault(key, []).append(row)
with open(os.path.join(OUT, "matrix.md"), "w") as f:
    f.write("# AR004 T015 main experiment matrix\n\n")
    f.write("| length | config | device | speedup | quality(med) | source |\n")
    f.write("|---|---|---|---:|---:|---|\n")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        g = groups[key]
        sps = [r["speedup"] for r in g if r["speedup"] != ""]
        qs = [r["quality"] for r in g]
        sp = statistics.median(sps) if sps else float("nan")
        f.write(f"| {key[0]} | {key[1]} | {key[2]} | {sp:.2f}x | "
                f"{statistics.median(qs):.3f} | {g[0]['source']} |\n")
print("summary -> matrix.md")
