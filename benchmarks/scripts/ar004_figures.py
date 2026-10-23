#!/usr/bin/env python3
"""AR004 T015 figures: renders the seven acceptance/explainability plots
from the assembled matrix (assemble_matrix.py) into
benchmarks/results/figures/. Safe to re-run; missing inputs are skipped.

Figures:
  ar004_fig1_ttft_curves    TTFT vs length per config (CPU + GPU)
  ar004_fig2_pareto         speedup vs quality, marker per config
  ar004_fig3_cliff_rescue   quality cliff without branches, pool rescue
  ar004_fig4_gating_ablation multineedle qg on/off bars
  ar004_fig5_chunk_harvest  large-chunk (auto n_ubatch) vs 512 chunking
  ar004_fig6_head_heatmap   head-classification profile matrix
  ar004_fig7_cpu_gpu        CPU vs GPU speedup bars
"""
import csv
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

def load_matrix():
    path = os.path.join(RESULTS, "ar004-t015-matrix", "matrix.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def med_group(rows, key_fn, val_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(val_fn(r))
    return {k: statistics.median(v) for k, v in groups.items() if v}

LEN_ORDER = {"4k": 4, "8k": 8, "16k": 16, "32k": 32, "8021": 8}

def fig1_ttft_curves(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, device in zip(axes, ["CPU", "GPU"]):
        for cfg, color in [("dense", "#d62728"), ("attn", "#2ca02c"),
                           ("head", "#9467bd"), ("global-attn24-pool", "#1f77b4")]:
            pts = {}
            for r in rows:
                if r["device"] != device or r["config"] != cfg:
                    continue
                L = LEN_ORDER.get(r["length"])
                if L is None:
                    continue
                pts.setdefault(L, []).append(float(r["ttft_sparse_ms"]) / 1000.0)
            if pts:
                xs = sorted(pts)
                ax.plot(xs, [statistics.median(pts[x]) for x in xs],
                        marker="o", label=cfg, color=color)
        ax.set_title(f"{device}: sparse prefill TTFT")
        ax.set_xlabel("prompt length (K tokens)")
        ax.set_ylabel("TTFT (s)")
        ax.legend(fontsize=8)
    fig.suptitle("AR004 fig1: TTFT scaling by config and device")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig1_ttft_curves.png"))
    plt.close(fig)

def fig2_pareto(rows):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    seen = set()
    for r in rows:
        if not r["speedup"]:
            continue
        key = (r["config"], r["device"])
        label = f"{r['config']} ({r['device']})" if key not in seen else None
        seen.add(key)
        ax.scatter(float(r["quality"]), float(r["speedup"]), s=42, label=label)
    ax.axhline(10, ls="--", c="gray", lw=1)
    ax.axvline(0.9, ls="--", c="gray", lw=1)
    ax.annotate("acceptance:\n>=10x, q>=0.9", xy=(0.91, 10.5), fontsize=8, color="gray")
    ax.set_xlabel("quality q")
    ax.set_ylabel("TTFT speedup vs dense (x)")
    ax.set_title("AR004 fig2: speedup-quality Pareto")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig2_pareto.png"))
    plt.close(fig)

def fig3_cliff_rescue(rows):
    """Aggressive attention gathers drop the needle (quality cliff); the
    pooled-branch config rescues it."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels, base_q, pool_q = [], [], []
    for L in ["4k", "8k", "16k", "32k"]:
        b = [float(r["quality"]) for r in rows
             if r["length"] == L and r["config"] == "attn" and r["device"] == "CPU"]
        p = [float(r["quality"]) for r in rows
             if r["length"] == L and "pool" in r["config"] and r["device"] == "CPU"]
        if not b:
            continue
        labels.append(L)
        base_q.append(statistics.median(b) if b else np.nan)
        pool_q.append(statistics.median(p) if p else np.nan)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, base_q, 0.34, label="scalar attn-24 (no branches)", color="#d62728")
    ax.bar(x + 0.18, pool_q, 0.34, label="+ pooled branches", color="#2ca02c")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("prompt length")
    ax.set_ylabel("needle quality q")
    ax.set_title("AR004 fig3: quality cliff and compressed-branch rescue")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig3_cliff_rescue.png"))
    plt.close(fig)

def fig4_gating_ablation(rows):
    fig, ax = plt.subplots(figsize=(6, 4))
    cases, off_q, on_q = [], [], []
    for case in ["mn-v1", "mn-v2", "mn-v3"]:
        off = [float(r["quality"]) for r in rows
               if r["case"].startswith(case) and r["config"] == "mn-global-attn24"]
        on = [float(r["quality"]) for r in rows
              if r["case"].startswith(case) and r["config"] == "mn-qg"]
        if not off and not on:
            continue
        cases.append(case)
        off_q.append(off[0] if off else np.nan)
        on_q.append(on[0] if on else np.nan)
    x = np.arange(len(cases))
    ax.bar(x - 0.18, off_q, 0.34, label="scalar attn-24", color="#1f77b4")
    ax.bar(x + 0.18, on_q, 0.34, label="query-chunk gating", color="#ff7f0e")
    ax.set_xticks(x, cases)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("needle quality q")
    ax.set_title("AR004 fig4: query-gating ablation (multineedle @8K)\n"
                 "1-D lexical per-position scores: no gain (see T009 analysis)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig4_gating_ablation.png"))
    plt.close(fig)

def fig5_chunk_harvest():
    """Reads the chunk ablation CSV (chunk_a, chunk_b rows) if present."""
    path = os.path.join(RESULTS, "ar004-t015-matrix", "chunk_ablation.csv")
    if not os.path.exists(path):
        print("fig5: chunk_ablation.csv missing, skipped")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = [r["config"] for r in rows]
    vals = [float(r["ttft_sparse_ms"]) / 1000.0 for r in rows]
    ax.bar(labels, vals, color=["#d62728", "#2ca02c"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("sparse prefill TTFT (s)")
    ax.set_title("AR004 fig5: large-chunk (auto n_ubatch) harvest")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig5_chunk_harvest.png"))
    plt.close(fig)

def fig6_head_heatmap():
    path = os.path.join(RESULTS, "exp-heads", "profile.json")
    if not os.path.exists(path):
        print("fig6: profile.json missing, skipped")
        return
    with open(path) as f:
        prof = json.load(f)
    heads = prof["heads"]  # [{layer, head, echo_score, induction_score, class}]
    mat = np.array([[h["echo_score"], h["induction_score"],
                     1.0 if h["class"] == "retrieval" else 0.0]
                    for h in heads], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   extent=[0, 3, len(heads), 0])
    ax.set_xticks([0.5, 1.5, 2.5], ["echo", "induction", "retrieval"])
    ax.set_ylabel("head (layer-major index)")
    ax.set_title("AR004 fig6: head classification profile")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig6_head_heatmap.png"))
    plt.close(fig)

def fig7_cpu_gpu(rows):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels, cpu, gpu = [], [], []
    for L in ["8k", "16k"]:
        for cfg in ["attn", "head"]:
            c = [float(r["speedup"]) for r in rows
                 if r["length"] == L and r["config"] == cfg and r["device"] == "CPU" and r["speedup"]]
            g = [float(r["speedup"]) for r in rows
                 if r["length"] == L and r["config"] == cfg and r["device"] == "GPU" and r["speedup"]]
            if not c and not g:
                continue
            labels.append(f"{L}-{cfg}")
            cpu.append(statistics.median(c) if c else np.nan)
            gpu.append(statistics.median(g) if g else np.nan)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, cpu, 0.34, label="CPU", color="#1f77b4")
    ax.bar(x + 0.18, gpu, 0.34, label="GPU (RTX 5000)", color="#ff7f0e")
    ax.axhline(10, ls="--", c="gray", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("TTFT speedup vs same-device dense (x)")
    ax.set_title("AR004 fig7: CPU vs GPU sparse speedup")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "ar004_fig7_cpu_gpu.png"))
    plt.close(fig)

def main():
    rows = load_matrix()
    if not rows:
        print("matrix.csv missing/empty; run assemble_matrix.py first")
        return
    fig1_ttft_curves(rows)
    fig2_pareto(rows)
    fig3_cliff_rescue(rows)
    fig4_gating_ablation(rows)
    fig5_chunk_harvest()
    fig6_head_heatmap()
    fig7_cpu_gpu(rows)
    print("AR004 figures written to", FIGDIR)

if __name__ == "__main__":
    main()
