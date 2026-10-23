#!/usr/bin/env python3
"""EdgePrefill result visualization.

Reads the exp-* benchmark CSVs, the head-classification profile, and the
selection dumps, then renders publication-quality figures into
benchmarks/results/figures/. Safe to re-run at any time: figures whose
inputs are missing are skipped with a note.

Run (Windows): C:\\Python311\\python.exe benchmarks\\scripts\\plot_results.py
"""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

C_DENSE = "#d62728"   # red
C_SEL = "#1f77b4"     # blue
C_ATTN = "#2ca02c"    # green
C_LAT = "#ff7f0e"     # orange
C_HEAD = "#9467bd"    # purple

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_csv(result_dir):
    """Return list of row dicts from the newest *_raw.csv in result_dir."""
    path = os.path.join(RESULTS, result_dir)
    if not os.path.isdir(path):
        return []
    csvs = sorted(f for f in os.listdir(path) if f.endswith("_raw.csv"))
    if not csvs:
        return []
    rows = []
    with open(os.path.join(path, csvs[-1]), newline="") as fh:
        for row in csv.DictReader(fh):
            for key in ("ttft_ms", "e2e_ms", "prefill_tps", "quality", "budget"):
                row[key] = float(row[key])
            for key in ("dense", "prompt_tokens", "selected_tokens", "attn_top_k", "repeats"):
                row[key] = int(row[key])
            rows.append(row)
    return rows


def med(rows, mode):
    """Median ttft/quality for a mode across cases."""
    sel = [r for r in rows if r["mode"] == mode]
    if not sel:
        return None
    return {
        "ttft": float(np.median([r["ttft_ms"] for r in sel])),
        "quality": float(np.mean([r["quality"] for r in sel])),
        "tps": float(np.median([r["prefill_tps"] for r in sel])),
        "tokens": sel[0]["prompt_tokens"],
        "sel_tokens": sel[0]["selected_tokens"],
        "k": sel[0]["attn_top_k"],
        "n": len(sel),
    }


def fig_length_scaling():
    lengths = [("len2k", "2K"), ("len4k", "4K"), ("len8k", "8K"), ("len16k", "16K")]
    series = {"dense": [], "sel": [], "attn": []}
    toks = {"dense": [], "sel": [], "attn": []}
    for d, _ in lengths:
        rows_sel = load_csv(f"exp-len-{d}-sel")
        rows_attn = load_csv(f"exp-len-{d}-attn")
        m = med(rows_sel, "dense") or med(rows_attn, "dense")
        s = med(rows_sel, "balanced")
        a = med(rows_attn, "balanced")
        if m:
            series["dense"].append(m["ttft"]); toks["dense"].append(m["tokens"])
        if s:
            series["sel"].append(s["ttft"]); toks["sel"].append(s["tokens"])
        if a:
            series["attn"].append(a["ttft"]); toks["attn"].append(a["tokens"])
    # 32K headline point: one run covers dense + selection+attn.
    rows_32k = load_csv("exp-len32k")
    m = med(rows_32k, "dense")
    a = med(rows_32k, "balanced")
    if m:
        series["dense"].append(m["ttft"]); toks["dense"].append(m["tokens"])
    if a:
        series["attn"].append(a["ttft"]); toks["attn"].append(a["tokens"])
    if not series["dense"]:
        print("skip fig_length_scaling (no data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    if series["dense"]:
        ax.plot(toks["dense"], series["dense"], "o-", color=C_DENSE, label="dense baseline")
    if series["sel"]:
        ax.plot(toks["sel"], series["sel"], "s-", color=C_SEL, label="EdgePrefill selection (b=0.5)")
    if series["attn"]:
        ax.plot(toks["attn"], series["attn"], "^-", color=C_ATTN, label="selection + attn top-k")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("TTFT (ms, median)")
    ax.set_title("(a) Time-to-first-token vs context length")
    ax.legend(fontsize=8)
    ax = axes[1]
    if series["sel"] and series["dense"]:
        n = min(len(series["sel"]), len(series["dense"]))
        ax.plot(toks["sel"][:n], [series["dense"][i] / series["sel"][i] for i in range(n)],
                "s-", color=C_SEL, label="selection")
    if series["attn"] and series["dense"]:
        n = min(len(series["attn"]), len(series["dense"]))
        ax.plot(toks["attn"][:n], [series["dense"][i] / series["attn"][i] for i in range(n)],
                "^-", color=C_ATTN, label="selection + attn")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("speedup vs dense (x)")
    ax.set_title("(b) Speedup over dense prefill")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_length_scaling.png"))
    plt.close(fig)
    print("wrote fig_length_scaling.png")


def fig_budget_pareto():
    pts = []  # (ttft, quality, label, color)
    base = load_csv("exp-len-len8k-sel")
    m = med(base, "dense")
    if m:
        pts.append((m["ttft"], m["quality"], "dense", C_DENSE))
    for b, color in [("0.25", C_SEL), ("0.4", C_SEL), ("0.7", C_SEL)]:
        rows = load_csv(f"exp-budget-{b}")
        s = med(rows, "balanced")
        if s:
            pts.append((s["ttft"], s["quality"], f"b={b}", color))
    s = med(base, "balanced")
    if s:
        pts.append((s["ttft"], s["quality"], "b=0.5", C_SEL))
    rows = load_csv("exp-latency")
    s = med(rows, "latency")
    if s:
        pts.append((s["ttft"], s["quality"], "latency+ffn-skip", C_LAT))
    if len(pts) < 2:
        print("skip fig_budget_pareto (no data)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ttft, q, label, color in pts:
        ax.scatter(ttft, q, s=70, color=color, zorder=3)
        ax.annotate(label, (ttft, q), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("TTFT (ms, median, 8K context)")
    ax.set_ylabel("answer quality (F1)")
    ax.set_title("Latency-quality Pareto frontier (needle-in-haystack, 8K)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_budget_pareto.png"))
    plt.close(fig)
    print("wrote fig_budget_pareto.png")


def fig_attnk_sweep():
    base = load_csv("exp-attnk-24")
    m = med(base, "dense")
    pts = []
    s0 = med(load_csv("exp-len-len8k-sel"), "balanced")
    if s0:
        pts.append((0, s0["ttft"], s0["quality"]))
    for k in (24, 48, 96):
        s = med(load_csv(f"exp-attnk-{k}"), "balanced")
        if s:
            pts.append((k, s["ttft"], s["quality"]))
    if not pts:
        print("skip fig_attnk_sweep (no data)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ks = [p[0] for p in pts]
    ttfts = [p[1] for p in pts]
    qs = [p[2] for p in pts]
    bars = ax.bar([str(k) for k in ks], ttfts,
                  color=[C_SEL] + [C_ATTN] * (len(ks) - 1), width=0.55)
    if m:
        ax.axhline(m["ttft"], color=C_DENSE, ls="--", lw=1.2)
        ax.text(len(ks) - 0.45, m["ttft"], " dense", color=C_DENSE, va="bottom", fontsize=9)
    for bar, t in zip(bars, ttfts):
        ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t/1000:.1f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("attention top-k blocks (0 = gather off), 32-token blocks")
    ax.set_ylabel("TTFT (ms, median, 8K)")
    ax2 = ax.twinx()
    ax2.plot([str(k) for k in ks], qs, "ko--", lw=1.2, ms=6)
    ax2.set_ylabel("quality (F1)")
    ax2.set_ylim(0, 1.15)
    ax2.grid(False)
    ax.set_title("Attention gather depth: latency vs quality")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_attnk_sweep.png"))
    plt.close(fig)
    print("wrote fig_attnk_sweep.png")


def fig_head_profile():
    path = os.path.join(RESULTS, "exp-heads", "profile.json")
    if not os.path.isfile(path):
        print("skip fig_head_profile (no profile)")
        return
    with open(path) as fh:
        prof = json.load(fh)
    n_layer, n_head = prof["n_layer"], prof["n_head"]
    scores = np.zeros((n_layer, n_head))
    retriev = np.zeros((n_layer, n_head), dtype=bool)
    for h in prof["heads"]:
        scores[h["layer"], h["head"]] = max(h["echo_score"], h["induction_score"])
        retriev[h["layer"], h["head"]] = h["retrieval"]
    frac = retriev.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    im = ax.imshow(scores, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ys, xs = np.where(retriev)
    ax.scatter(xs, ys, s=42, facecolors="none", edgecolors="red", linewidths=1.6)
    ax.set_xlabel("attention head")
    ax.set_ylabel("layer")
    ax.set_title(f"(a) Retrieval sensitivity per head\n(red = retrieval head, {frac*100:.0f}% of heads)")
    fig.colorbar(im, ax=ax, label="max(echo, induction) flip rate")
    ax = axes[1]
    flat = scores.ravel()
    ax.hist(flat, bins=30, color=C_HEAD, alpha=0.8)
    ax.axvline(0.10, color="red", ls="--", lw=1.2, label="retrieval threshold 0.10")
    ax.set_xlabel("flip rate")
    ax.set_ylabel("#heads")
    ax.set_title("(b) Head sensitivity distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_head_profile.png"))
    plt.close(fig)
    print("wrote fig_head_profile.png")


def _find_needle_block(context_path, n_blocks):
    try:
        with open(context_path, errors="ignore") as fh:
            text = fh.read()
        idx = text.find("MAGENTA-FALCON")
        if idx < 0:
            return None
        return int(idx / max(len(text), 1) * n_blocks)
    except OSError:
        return None


def fig_selection_map():
    dump_dir = os.path.join(RESULTS, "exp-sel-dump")
    if not os.path.isdir(dump_dir):
        print("skip fig_selection_map (no dump)")
        return
    files = [f for f in os.listdir(dump_dir) if f.endswith(".json")]
    if not files:
        print("skip fig_selection_map (empty dump)")
        return
    with open(os.path.join(dump_dir, files[0])) as fh:
        sel = json.load(fh)
    signals = sel["signals"]
    n_blocks = len(signals)
    fused = np.array([s["fused"] for s in signals])
    lexical = np.array([s["lexical"] for s in signals])
    chosen = np.zeros(n_blocks, dtype=bool)
    chosen[sel["block_indices"]] = True
    mand = np.zeros(n_blocks, dtype=bool)
    mand[sel["mandatory_blocks"]] = True
    ctx = os.path.join(ROOT, "benchmarks", "data", "niah", "contexts", "niah-01.txt")
    needle_block = _find_needle_block(ctx, n_blocks)
    order = np.argsort(-fused)
    rank_of = np.empty(n_blocks, dtype=int)
    rank_of[order] = np.arange(n_blocks)
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = np.arange(n_blocks)
    colors = np.where(chosen, C_SEL, "#c7c7c7")
    colors = np.where(mand, C_ATTN, colors)
    ax.bar(xs, fused, color=colors, width=1.0)
    ax.bar(xs, lexical * fused.max(), color="none", edgecolor="#2222aa",
           linewidth=0.4, width=1.0, label="_nolegend_")
    if needle_block is not None:
        ax.axvline(needle_block, color="red", lw=1.6, ls="--")
        ax.text(needle_block + 2, fused.max() * 0.95, "needle", color="red", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_ATTN, label="mandatory (sink+recent)"),
        Patch(color=C_SEL, label="selected by budget"),
        Patch(color="#c7c7c7", label="dropped"),
    ], fontsize=8, loc="upper right")
    ax.set_xlabel("block index (32 tokens/block)")
    ax.set_ylabel("fused signal")
    kept = sel["selected_tokens"]
    ax.set_title(f"Block selection for a needle query: kept {kept}/{sel['total_tokens']} tokens "
                 f"({kept/sel['total_tokens']*100:.0f}%), reason={sel['reason']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_selection_map.png"))
    plt.close(fig)
    print("wrote fig_selection_map.png")


def fig_mode_bars():
    entries = []
    m = med(load_csv("exp-len-len8k-sel"), "dense")
    if m:
        entries.append(("dense\nbaseline", m["ttft"], m["quality"], C_DENSE))
    s = med(load_csv("exp-len-len8k-sel"), "balanced")
    if s:
        entries.append(("selection\nb=0.5", s["ttft"], s["quality"], C_SEL))
    best_k, best = None, None
    for k in (24, 48, 96):
        s = med(load_csv(f"exp-attnk-{k}"), "balanced")
        if s and (best is None or s["ttft"] < best["ttft"]):
            best_k, best = k, s
    if best:
        entries.append((f"selection+attn\nk={best_k}", best["ttft"], best["quality"], C_ATTN))
    lat = med(load_csv("exp-latency"), "latency")
    if lat:
        entries.append(("latency\n+FFN skip", lat["ttft"], lat["quality"], C_LAT))
    if not entries:
        print("skip fig_mode_bars (no data)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [e[0] for e in entries]
    ttfts = [e[1] for e in entries]
    colors = [e[3] for e in entries]
    bars = ax.bar(labels, ttfts, color=colors, width=0.6)
    for bar, e in zip(bars, entries):
        ax.text(bar.get_x() + bar.get_width() / 2, e[1],
                f"{e[1]/1000:.1f}s\nF1={e[2]:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("TTFT (ms, median, 8K context)")
    ax.set_ylim(0, max(ttfts) * 1.25)
    ax.set_title("EdgePrefill optimization stack (Qwen2.5-1.5B q4_k_m, 16 CPU cores)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_mode_bars.png"))
    plt.close(fig)
    print("wrote fig_mode_bars.png")


def fig_quality_cliff():
    """Budget sweep including extremes: TTFT and quality vs budget."""
    pts = []  # (budget, ttft, quality)
    for b in (0.1, 0.15, 0.25, 0.4, 0.7):
        s = med(load_csv(f"exp-ext-{b}" if b in (0.1, 0.15) else f"exp-budget-{b}"), "balanced")
        if s:
            pts.append((b, s["ttft"], s["quality"]))
    s = med(load_csv("exp-len-len8k-sel"), "balanced")
    if s:
        pts.append((0.5, s["ttft"], s["quality"]))
    m = med(load_csv("exp-len-len8k-sel"), "dense")
    if not pts:
        print("skip fig_quality_cliff (no data)")
        return
    pts.sort()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bs = [p[0] for p in pts]
    ttfts = [p[1] for p in pts]
    qs = [p[2] for p in pts]
    ax.plot(bs, ttfts, "o-", color=C_SEL, label="EdgePrefill TTFT")
    if m:
        ax.axhline(m["ttft"], color=C_DENSE, ls="--", lw=1.2, label="dense baseline")
    ax.set_xlabel("selection budget (fraction of blocks)")
    ax.set_ylabel("TTFT (ms, median, 8K)")
    ax2 = ax.twinx()
    ax2.plot(bs, qs, "ks--", lw=1.2, ms=7, label="quality (F1)")
    ax2.set_ylabel("quality (F1)")
    ax2.set_ylim(0, 1.15)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Quality cliff: how far the budget can shrink")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_quality_cliff.png"))
    plt.close(fig)
    print("wrote fig_quality_cliff.png")


def fig_grid_heatmap():
    """Budget x attention-top-k TTFT heatmap at 8K."""
    grid = {}  # (budget, k) -> ttft
    for b in (0.25, 0.7):
        for k in (24, 48):
            s = med(load_csv(f"exp-grid-b{b}-k{k}"), "balanced")
            if s:
                grid[(b, k)] = s["ttft"]
    s = med(load_csv("exp-len-len8k-sel"), "balanced")
    if s:
        grid[(0.5, 0)] = s["ttft"]
    for k in (24, 48, 96):
        s = med(load_csv(f"exp-attnk-{k}"), "balanced")
        if s:
            grid[(0.5, k)] = s["ttft"]
    if len(grid) < 4:
        print("skip fig_grid_heatmap (no data)")
        return
    budgets = sorted({b for b, _ in grid})
    ks = sorted({k for _, k in grid})
    mat = np.full((len(budgets), len(ks)), np.nan)
    for (b, k), t in grid.items():
        mat[budgets.index(b), ks.index(k)] = t
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(mat / 1000.0, cmap="viridis_r", aspect="auto")
    for i in range(len(budgets)):
        for j in range(len(ks)):
            if not np.isnan(mat[i, j]):
                v = mat[i, j] / 1000.0
                ax.text(j, i, f"{v:.1f}s", ha="center", va="center", fontsize=10,
                        color="white" if v > np.nanmax(mat) / 2000 else "black")
    ax.set_xticks(range(len(ks)), [f"k={k}" for k in ks])
    ax.set_yticks(range(len(budgets)), [f"b={b}" for b in budgets])
    fig.colorbar(im, ax=ax, label="TTFT (s, median, 8K)")
    ax.set_title("Sparsity design space: budget x attention depth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_grid_heatmap.png"))
    plt.close(fig)
    print("wrote fig_grid_heatmap.png")


def fig_decode_gain():
    """Longer-generation run: smaller KV cache also speeds up decode."""
    rows = load_csv("exp-decode")
    m = med(rows, "dense")
    s = med(rows, "balanced")
    if not (m and s):
        print("skip fig_decode_gain (no data)")
        return
    def decode_ms(r):
        return r["e2e_ms"] - r["ttft_ms"]
    dm = [decode_ms(r) for r in rows if r["mode"] == "dense"]
    ds = [decode_ms(r) for r in rows if r["mode"] == "balanced"]
    n_gen = 128
    labels = ["dense", "EdgePrefill b=0.5"]
    ttft = [m["ttft"] / 1000, s["ttft"] / 1000]
    dec = [float(np.median(dm)) / 1000, float(np.median(ds)) / 1000]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(2)
    ax.bar(x - 0.18, ttft, 0.34, color=[C_DENSE, C_SEL], label="prefill TTFT")
    ax.bar(x + 0.18, dec, 0.34, color=["#a03030", "#3070a0"], label=f"decode {n_gen} tokens")
    ax.set_xticks(x, labels)
    ax.set_ylabel("time (s)")
    ax.legend(fontsize=8)
    ax.set_title("(a) Prefill and decode time, 8K context")
    ax = axes[1]
    tps_m = [n_gen / (d / 1000) for d in dm]
    tps_s = [n_gen / (d / 1000) for d in ds]
    bp = ax.boxplot([tps_m, tps_s], widths=0.5,
                    patch_artist=True, boxprops=dict(facecolor="#cfe3f5"))
    ax.set_xticklabels(labels)
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_title("(b) Decode throughput: smaller KV = less memory traffic")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_decode_gain.png"))
    plt.close(fig)
    print("wrote fig_decode_gain.png")


def fig_multineedle():
    """Robustness: three needles at different depths, one 8K context."""
    rows = load_csv("exp-multineedle")
    if not rows:
        print("skip fig_multineedle (no data)")
        return
    cases = sorted({r["case"].split("@")[0] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(cases))
    w = 0.34
    qd, qa_ = [], []
    for c in cases:
        d = [r["quality"] for r in rows if r["case"].startswith(c) and r["dense"]]
        a = [r["quality"] for r in rows if r["case"].startswith(c) and not r["dense"]]
        qd.append(float(np.mean(d)) if d else 0)
        qa_.append(float(np.mean(a)) if a else 0)
    ax.bar(x - w / 2, qd, w, color=C_DENSE, label="dense")
    ax.bar(x + w / 2, qa_, w, color=C_ATTN, label="selection+attn k=48")
    ax.set_xticks(x, cases)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("answer quality (F1)")
    ax.legend(fontsize=9)
    ax.set_title("Multi-needle robustness: 3 needles at 1/4, 1/2, 3/4 depth (8K)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_multineedle.png"))
    plt.close(fig)
    print("wrote fig_multineedle.png")


def fig_head_adaptive():
    """Head-adaptive sweep vs scalar attention path vs dense."""
    m = med(load_csv("exp-headk-64"), "dense")
    pts = []  # (k, ttft, quality)
    for k in (64, 128, 256):
        s = med(load_csv(f"exp-headk-{k}"), "balanced")
        if s:
            pts.append((k, s["ttft"], s["quality"]))
    h16 = med(load_csv("exp-head-16k"), "balanced")
    if not (m or pts):
        print("skip fig_head_adaptive (no data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if pts:
        ks = [p[0] for p in pts]
        ttfts = [p[1] for p in pts]
        bars = ax.bar([str(k) for k in ks], ttfts, color=C_HEAD, width=0.55)
        for bar, t in zip(bars, ttfts):
            ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t/1000:.1f}s",
                    ha="center", va="bottom", fontsize=9)
    if m:
        ax.axhline(m["ttft"], color=C_DENSE, ls="--", lw=1.2)
        ax.text(0.02, m["ttft"], " dense 8K", color=C_DENSE, va="bottom", fontsize=9)
    ax.set_xlabel("per-head top-k rows (retrieval heads stay dense)")
    ax.set_ylabel("TTFT (ms, median, 8K)")
    ax.set_title("(a) Head-adaptive gather depth")
    ax = axes[1]
    if pts:
        qs = [p[2] for p in pts]
        ax.plot([str(k) for k in ks], qs, "o-", color=C_HEAD, ms=8)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("per-head top-k rows")
        ax.set_ylabel("quality (F1)")
    if h16:
        ax.set_title(f"(b) Quality; 16K point: {h16['ttft']/1000:.1f}s F1={h16['quality']:.2f}")
    else:
        ax.set_title("(b) Quality vs gather depth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_head_adaptive.png"))
    plt.close(fig)
    print("wrote fig_head_adaptive.png")


def _load_llama_bench_csv():
    """Parse the newest llama-bench CSV (GPU dense baseline)."""
    path = os.path.join(RESULTS, "exp-gpu-dense")
    if not os.path.isdir(path):
        return []
    csvs = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
    if not csvs:
        return []
    rows = []
    with open(os.path.join(path, csvs[-1]), newline="") as fh:
        for row in csv.DictReader(fh):
            row["n_prompt"] = int(row["n_prompt"])
            row["avg_ts"] = float(row["avg_ts"])
            rows.append(row)
    return rows


def fig_cpu_gpu():
    """CPU vs GPU: does sparsity matter once the GPU does the math?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # ---- (a) CPU: dense vs sparse from our binary ----
    ax = axes[0]
    for d, series, color, label in (
        ("exp-len-len8k-sel", "dense", C_DENSE, "dense"),
        ("exp-len-len8k-sel", "balanced", C_SEL, "EdgePrefill b=0.5"),
    ):
        rows = load_csv(d)
        m = med(rows, series)
        if not m:
            continue
        d16 = med(load_csv("exp-len-len16k-sel"), "dense" if series == "dense" else "balanced")
        pts = [(m["tokens"], m["ttft"])]
        if d16:
            pts.append((d16["tokens"], d16["ttft"]))
        r32 = med(load_csv("exp-len32k"), "dense" if series == "dense" else "balanced")
        if r32:
            pts.append((r32["tokens"], r32["ttft"]))
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] / 1000 for p in pts], "o-", color=color, label=label)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("TTFT (s)")
    ax.set_title("(a) CPU (16 cores): sparsity pays")
    ax.legend(fontsize=8)
    # ---- (b) GPU: llama-bench dense + our GPU sparse runs ----
    ax = axes[1]
    bench_rows = _load_llama_bench_csv()
    for fa, color, label in (("off", C_DENSE, "GPU dense (llama-bench, FA off)"),
                             ("on", "#7f0000", "GPU dense (FA on)")):
        pts = sorted((r["n_prompt"], r["n_prompt"] / r["avg_ts"])
                     for r in bench_rows if r["flash_attn"] == fa)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, label=label)
    for d, label in (("exp-gpu-8k-sel", "GPU EdgePrefill b=0.5"),
                     ("exp-gpu-8k-attn", "GPU sel+attn k=24"),
                     ("exp-gpu-16k", "GPU sel+attn 16K"),
                     ("exp-gpu-32k", "GPU sel+attn 32K")):
        rows = load_csv(d)
        m = med(rows, "balanced")
        if m:
            ax.scatter([m["tokens"]], [m["tps"]], marker="s", s=60, color=C_ATTN,
                       label=label, zorder=3)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prefill throughput (tok/s)")
    ax.set_title("(b) GPU (RTX 5000): dense vs sparse")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_cpu_gpu.png"))
    plt.close(fig)
    print("wrote fig_cpu_gpu.png")


def fig_ppl_curve():
    """Perplexity-style quality vs budget from the ppl dataset."""
    pts = []
    for b, d in ((0.1, "exp-ppl-b0.1"), (0.25, "exp-ppl-b0.25"), (0.5, "exp-ppl")):
        s = med(load_csv(d), "balanced")
        if s:
            pts.append((b, s["quality"], s["ttft"]))
    if not pts:
        print("skip fig_ppl_curve (no data)")
        return
    m = med(load_csv("exp-ppl"), "dense")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bs = [p[0] for p in pts]
    qs = [p[1] for p in pts]
    ax.plot(bs, qs, "o-", color=C_SEL, ms=8)
    if m:
        ax.axhline(m["quality"], color=C_DENSE, ls="--", lw=1.2, label="dense")
        ax.legend(fontsize=9)
    ax.set_xlabel("selection budget")
    ax.set_ylabel("quality score (ppl task)")
    ax.set_title("Language-modeling quality vs budget")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_ppl_curve.png"))
    plt.close(fig)
    print("wrote fig_ppl_curve.png")


def main():
    print(f"results root: {RESULTS}")
    fig_length_scaling()
    fig_budget_pareto()
    fig_attnk_sweep()
    fig_quality_cliff()
    fig_grid_heatmap()
    fig_decode_gain()
    fig_multineedle()
    fig_head_profile()
    fig_head_adaptive()
    fig_selection_map()
    fig_mode_bars()
    fig_cpu_gpu()
    fig_ppl_curve()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
