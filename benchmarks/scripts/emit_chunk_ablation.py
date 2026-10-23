#!/usr/bin/env python3
"""Emit chunk_ablation.csv for fig5 from the chunk-512 / chunk-0 runs."""
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "benchmarks", "results", "ar004-t015-matrix")
rows = []
for tag, label in [("chunk-128", "chunk_size=128 (4x more chunks)"),
                   ("chunk-512", "chunk_size=512 (old default)"),
                   ("chunk-0", "chunk_size=0 (auto n_ubatch)")]:
    csvs = sorted(glob.glob(os.path.join(BASE, tag, "*_raw.csv")))
    if not csvs:
        continue
    with open(csvs[-1]) as f:
        for r in csv.DictReader(f):
            if r.get("dense") == "1":
                continue
            rows.append({"config": label, "ttft_sparse_ms": r["ttft_ms"],
                         "quality": r["quality"], "case": r["case"]})
out = os.path.join(BASE, "chunk_ablation.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["config", "case", "ttft_sparse_ms", "quality"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r["config"], r["ttft_sparse_ms"], "q=" + r["quality"])
print("->", out)
