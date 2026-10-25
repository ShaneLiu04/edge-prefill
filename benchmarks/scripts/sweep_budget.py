#!/usr/bin/env python3
"""AR007 FR-1: measure the admission-budget vs quality curve on the real model.

Runs `edge-prefill bench` for every (dataset, budget, signal) cell and writes:

    <out>/<tag>/bench.conf     the exact conf that produced the cell
    <out>/<tag>/raw.csv        the per-case samples (audit columns included)
    <out>/summary.csv          one row per cell
    <out>/curve.json           the machine readable curve + the rule output
    <out>/budget_curve.txt     the planner input (model_sha256 + length = budget)
    <out>/summary.md           human table + the recommendation

Speed is the median TTFT over the cases of a cell; quality is the mean of the
per-case quality and `min_quality` is the worst case, because one lost needle
must not be averaged away.

Quality is not monotone in the budget here (anchors and MMR change which
blocks are admitted, not only how many), so the rule is "fastest cell that
meets the floor, times the margin" - picking the smallest passing budget
instead would select a slower cell purely because its budget is lower. When no
cell reaches the floor the dataset is reported as `floor-missed` and its
fastest cell is still written, so the curve always covers every measured
length. The planner consumes the curve with ceiling semantics (see
budget_curve.h): a prompt between two lengths borrows the next longer cell.

Nothing is interpolated: a cell either ran or it is reported as missing.

Usage:
  sweep_budget.py --bin build-wsl/edge-prefill --model M.gguf \
      --datasets explen/len4k,explen/len8k,explen/len16k,deep/len32k \
      --budgets 0.05,0.08,0.12,0.20,0.25 --out benchmarks/results/ar007-sweep
"""
import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sha256_of(path):
    out = subprocess.run(["sha256sum", path], capture_output=True, text=True)
    if out.returncode != 0:
        return "unavailable"
    return out.stdout.split()[0]


def parse_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def cell_conf(budget, signal, model, embedding, repeats):
    lines = [
        "mode = balanced",
        f"budget = {budget:.4f}",
        "attn_top_k = 24",
        "pool_branches = false",
        "query_gating = false",
        "qg_bound = false",
    ]
    # There is no `score_source` conf key (it is an output column): the
    # embedding arm is selected by pointing the conf at an embedding GGUF,
    # which turns on the AR005 semantic scorer.
    if signal == "embedding":
        lines.append(f"embedding_model = {embedding}")
    else:
        lines.append("embedding_model =")
    lines += [
        "threads = 16",
        "warmup = 0",
        "max_new_tokens = 16",
        f"repeats = {repeats}",
    ]
    return "\n".join(lines) + "\n"


def run_cell(bin_path, conf_path, model, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        bin_path, "bench",
        "--config", conf_path,
        "--model", model,
        "--data-dir", data_dir,
        "--out", out_dir,
        "--bench-skip-dense",
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - started
    if proc.returncode != 0:
        return None, f"rc={proc.returncode} wall={wall:.0f}s " + proc.stderr[-400:]
    files = sorted(glob.glob(os.path.join(out_dir, "*_raw.csv")) +
                   glob.glob(os.path.join(out_dir, "raw.csv")))
    if not files:
        return None, f"no raw.csv wall={wall:.0f}s"
    with open(files[-1], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "raw.csv has no samples"
    return rows, None


def summarize(rows):
    quality = [float(r["quality"]) for r in rows]
    ttft = [float(r["ttft_ms"]) for r in rows]
    return {
        "prompt_tokens": max(int(r["prompt_tokens"]) for r in rows),
        "selected_tokens": int(rows[0]["selected_tokens"]) if len(
            {r["selected_tokens"] for r in rows}) == 1 else int(statistics.mean(
                int(r["selected_tokens"]) for r in rows)),
        "quality": statistics.mean(quality),
        "min_quality": min(quality),
        "ttft_ms": statistics.median(ttft),
        "score_source": rows[0]["score_source"],
        "budget_source": rows[0]["budget_source"],
    }


def recommend(cells, floor, margin, tolerance):
    """Cheapest cell that is fast enough, per measured length.

    Two cells that differ by a few percent of TTFT are a tie, and a tie must be
    broken by budget, not by timer noise: taking the raw minimum would pick
    b=0.12 over b=0.08 for a 1.7% difference and spend 50% more tokens for
    nothing. So: keep the passing cells within `tolerance` of the fastest,
    take the smallest budget among them.

    When nothing passes the floor the budget cannot be optimised - the honest
    output is the best quality measured, so the curve degrades to that cell and
    the note says `floor-missed`. No margin is applied there: with a
    non-monotone curve a bigger budget can be strictly worse (32K: b=0.05
    q=0.89, b=0.08 q=0.67), so inflating it would launder a worse point.

    `margin` > 1 snaps up to the next *measured* budget; it never invents a
    grid point that was not measured.
    """
    by_length = {}
    for cell in cells:
        by_length.setdefault(cell["length"], []).append(cell)
    out = {}
    for length, group in sorted(by_length.items()):
        passing = [c for c in group if c["quality"] >= floor and c["min_quality"] >= floor]
        if passing:
            fastest = min(c["ttft_ms"] for c in passing)
            close = [c for c in passing if c["ttft_ms"] <= fastest * tolerance]
            best = min(close, key=lambda c: (c["budget"], c["ttft_ms"]))
        else:
            best = max(group, key=lambda c: (c["quality"], -c["ttft_ms"]))
        budget = best["budget"]
        if passing and margin > 1.0:
            target = budget * margin
            grid = sorted({c["budget"] for c in group})
            above = [b for b in grid if b >= target - 1e-9]
            budget = above[0] if above else grid[-1]
        out[str(length)] = {
            "budget": budget,
            "signal": best["signal"],
            "quality": round(best["quality"], 3),
            "min_quality": round(best["min_quality"], 3),
            "note": "floor-met" if passing else "floor-missed",
            "measured_budget": best["budget"],
            "measured_ttft_ms": round(best["ttft_ms"], 1),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="edge-prefill binary")
    ap.add_argument("--model", required=True, help="GGUF used for every cell")
    ap.add_argument("--embedding", default="", help="embedding GGUF for signal=embedding")
    ap.add_argument("--datasets", default="explen/len4k,explen/len8k",
                    help="comma separated <suite>/<name> under benchmarks/data")
    ap.add_argument("--budgets", default="0.05,0.08,0.12,0.20",
                    help="comma separated admission budgets")
    ap.add_argument("--signals", default="lexical", help="lexical,embedding")
    ap.add_argument("--quality-floor", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="snap the budget up to the next measured grid point "
                         "when above 1.0 (off by default: the curve is not monotone)")
    ap.add_argument("--speed-tolerance", type=float, default=1.15,
                    help="cells within this TTFT ratio of the fastest are a tie; "
                         "the cheapest one wins")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="benchmarks/results/ar007-sweep")
    ap.add_argument("--force", action="store_true", help="re-run cells that already have raw.csv")
    args = ap.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    bin_path = args.bin if os.path.isabs(args.bin) else os.path.join(ROOT, args.bin)
    if not os.path.exists(bin_path):
        sys.stderr.write(f"missing binary {bin_path}\n")
        return 1
    signals = parse_list(args.signals)
    if "embedding" in signals and not args.embedding:
        sys.stderr.write("--embedding is required for signal=embedding\n")
        return 1
    model_sha = sha256_of(args.model)

    cells = []
    for dataset in parse_list(args.datasets):
        name = dataset.rsplit("/", 1)[-1]
        data_dir = os.path.join(ROOT, "benchmarks/data", dataset)
        for budget in [float(b) for b in parse_list(args.budgets)]:
            for signal in signals:
                tag = f"{name}-b{budget:.2f}-{signal}"
                cell_dir = os.path.join(out, tag)
                conf_path = os.path.join(cell_dir, "bench.conf")
                raw = sorted(glob.glob(os.path.join(cell_dir, "*_raw.csv")) +
                             glob.glob(os.path.join(cell_dir, "raw.csv")))
                if raw and not args.force:
                    with open(raw[-1], newline="", encoding="utf-8") as f:
                        rows = list(csv.DictReader(f))
                    err = None if rows else "raw.csv has no samples"
                else:
                    os.makedirs(cell_dir, exist_ok=True)
                    with open(conf_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(cell_conf(budget, signal, args.model,
                                          args.embedding, args.repeats))
                    rows, err = run_cell(bin_path, conf_path, args.model, data_dir, cell_dir)
                if err:
                    print(f"[fail] {tag}: {err}", flush=True)
                    continue
                stats = summarize(rows)
                cell = {
                    "dataset": dataset,
                    "tag": tag,
                    "budget": budget,
                    "signal": signal,
                    "length": stats["prompt_tokens"],
                    "repeats": args.repeats,
                    **{k: v for k, v in stats.items() if k != "prompt_tokens"},
                }
                cells.append(cell)
                print(f"[cell] {tag} len={cell['length']} q={cell['quality']:.3f} "
                      f"min={cell['min_quality']:.3f} ttft={cell['ttft_ms']:.0f}ms",
                      flush=True)

    if not cells:
        sys.stderr.write("no cells measured\n")
        return 1

    cols = ["dataset", "tag", "budget", "signal", "prompt_tokens", "selected_tokens",
            "quality", "min_quality", "ttft_ms", "repeats", "score_source", "budget_source"]
    by_tag = {c["tag"]: c for c in cells}
    with open(os.path.join(out, "summary.csv"), "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for dataset in parse_list(args.datasets):
            name = dataset.rsplit("/", 1)[-1]
            for budget in [float(b) for b in parse_list(args.budgets)]:
                for signal in signals:
                    cell = by_tag.get(f"{name}-b{budget:.2f}-{signal}")
                    if cell:
                        row = {k: cell[k] for k in cols if k in cell}
                        row["prompt_tokens"] = cell["length"]
                        w.writerow(row)

    rec = recommend(cells, args.quality_floor, args.margin, args.speed_tolerance)

    with open(os.path.join(out, "curve.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "model_sha256": model_sha,
            "quality_floor": args.quality_floor,
            "margin": args.margin,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "points": [{
                "dataset": c["dataset"],
                "length": c["length"],
                "budget": c["budget"],
                "signal": c["signal"],
                "quality": round(c["quality"], 4),
                "min_quality": round(c["min_quality"], 4),
                "ttft_ms": round(c["ttft_ms"], 1),
            } for c in sorted(cells, key=lambda c: (c["length"], c["budget"], c["signal"]))],
            "recommendation": rec,
        }, f, indent=2)
        f.write("\n")

    with open(os.path.join(out, "budget_curve.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# AR007 budget curve generated by sweep_budget.py\n")
        f.write("# cheapest budget within the speed tolerance that reaches the floor;\n")
        f.write("# floor-missed lengths carry their best measured quality instead\n")
        f.write("# consumed with ceiling semantics (shorter prompts borrow the next length)\n")
        f.write(f"model_sha256 = {model_sha}\n")
        f.write(f"quality_floor = {args.quality_floor}\n")
        for length in sorted(rec, key=int):
            f.write(f"{length} = {rec[length]['budget']:.2f}\n")

    lines = ["# AR007 budget sweep", ""]
    lines.append(f"model sha256 `{model_sha}`, quality floor {args.quality_floor}, "
                 f"margin {args.margin}, repeats {args.repeats}.")
    lines.append("")
    lines.append("| dataset | length | budget | signal | selected | q | min q | ttft ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: (c["length"], c["budget"], c["signal"])):
        lines.append(
            f"| {c['dataset']} | {c['length']} | {c['budget']:.2f} | {c['signal']} | "
            f"{c['selected_tokens']} | {c['quality']:.3f} | {c['min_quality']:.3f} | "
            f"{c['ttft_ms']:.0f} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "Quality is not monotone in the budget on this task family, so the rule "
        "does not walk up from the smallest budget: among the cells that meet the "
        "floor it keeps those within `--speed-tolerance` of the fastest and takes "
        "the cheapest, which breaks timer-noise ties by budget. `floor-missed` "
        "marks lengths where no cell reached the floor; those keep their best "
        "measured quality and no margin, because a larger budget can be worse."
    )
    lines.append("")
    for length in sorted(rec, key=int):
        r = rec[length]
        lines.append(
            f"- {length} tokens -> budget {r['budget']:.2f} (signal {r['signal']}, "
            f"q={r['quality']}, {r['note']})"
        )
    lines.append("")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    print(f"rows={len(cells)} -> {os.path.join(out, 'summary.md')}, curve.json, budget_curve.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
