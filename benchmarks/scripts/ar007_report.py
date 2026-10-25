#!/usr/bin/env python3
"""AR007 FR-5: turn the acceptance campaign into a report.

Reads `<in>/<dataset>-<tag>/*_raw.csv` produced by ar007_accept.sh and writes
`<in>/REPORT.md` with one table per dataset:

    dense TTFT      reference, all tokens resident
    auto TTFT       planner picks the budget from the measured curve
    AR006 TTFT      the previous fixed 0.25 budget

Speedups are dense/配置 medians, quality is the per-dataset mean plus the
worst case, and the audit columns (budget_source, anchor_blocks,
mmr_penalized) are echoed so a reader can tell a curve hit from a conf
override without opening the CSV.

Nothing is interpolated: a missing directory or a missing case is reported as
`missing`, and a campaign with zero samples fails loudly.
"""
import csv
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Order matters: the first three rows are the comparison the campaign was
# built for; any extra suffix (e.g. qg-p16) is appended after them.
TAGS = [("dense", "dense"), ("auto", "auto (curve)"), ("ar006-budget", "AR006 b=0.25")]
KNOWN_DATASETS = ["multineedle", "len32k", "len16k", "len8k", "len4k"]


def split_dir(dirname):
    """('len32k-qg-p24-local32') -> ('len32k', 'qg-p24-local32')."""
    for ds in KNOWN_DATASETS:
        if dirname == ds:
            return ds, "dense"
        if dirname.startswith(ds + "-"):
            return ds, dirname[len(ds) + 1:]
    return dirname, "dense"


def tag_label(tag):
    for known, label in TAGS:
        if tag == known:
            return label
    return tag.replace("qg-", "qg ")


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: ar007_report.py <campaign-dir>\n")
        return 2
    camp = argv[1]
    if not os.path.isabs(camp):
        camp = os.path.join(ROOT, camp)
    groups = {}
    for path in sorted(glob.glob(os.path.join(camp, "*"))):
        if not os.path.isdir(path):
            continue
        dataset, tag = split_dir(os.path.basename(path))
        groups.setdefault(dataset, {})[tag] = path
    datasets = sorted(groups, key=lambda d: (d not in KNOWN_DATASETS, KNOWN_DATASETS.index(d)
                                             if d in KNOWN_DATASETS else 0, d))
    if not datasets:
        sys.stderr.write(f"no dataset directories under {camp}\n")
        return 1

    lines = ["# AR007 acceptance campaign", ""]
    lines.append(
        "Measured with `benchmarks/scripts/ar007_accept.sh` in one session per "
        "dataset. `auto` lets the planner read `budget_curve.txt`; the AR006 "
        "column is the fixed 0.25 budget for comparison."
    )
    lines.append("")
    total_rows = 0

    for ds in datasets:
        per_tag = {}
        extra = [t for t in groups[ds] if t not in {k for k, _ in TAGS}]
        order = [k for k, _ in TAGS] + sorted(extra)
        for tag in order:
            rows = []
            path = groups[ds].get(tag)
            if path:
                for f in sorted(glob.glob(os.path.join(path, "*_raw.csv")) +
                                glob.glob(os.path.join(path, "raw.csv"))):
                    rows.extend(load(f))
            per_tag[tag] = rows
            total_rows += len(rows)
        dense = per_tag["dense"]
        if not dense:
            lines.append(f"## {ds}\n\nmissing dense reference\n")
            continue
        dense_ttft = statistics.median(float(r["ttft_ms"]) for r in dense)
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"dense TTFT median {dense_ttft:.0f} ms over {len(dense)} cases.")
        lines.append("")
        lines.append("| config | cases | q mean | q min | TTFT median ms | speedup | budget | source | anchors | mmr |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for tag in order:
            label = tag_label(tag)
            rows = per_tag[tag]
            if not rows:
                lines.append(f"| {label} | 0 | missing | missing | missing | missing | | | | |")
                continue
            q = [float(r["quality"]) for r in rows]
            ttft = [float(r["ttft_ms"]) for r in rows]
            med = statistics.median(ttft)
            speed = dense_ttft / med if med > 0 else 0.0
            budgets = sorted({r["budget"] for r in rows})
            sources = sorted({r["budget_source"] for r in rows})
            anchors = [int(r["anchor_blocks"]) for r in rows]
            mmr = sum(int(r["mmr_penalized"]) for r in rows)
            lines.append(
                f"| {label} | {len(rows)} | {statistics.mean(q):.3f} | {min(q):.3f} | "
                f"{med:.0f} | {speed:.1f}x | {','.join(budgets)} | {','.join(sources)} | "
                f"{min(anchors)}-{max(anchors)} | {mmr} |"
            )
        lines.append("")
        auto = per_tag.get("auto", [])
        base = per_tag.get("ar006-budget", [])
        if auto and base:
            amed = statistics.median(float(r["ttft_ms"]) for r in auto)
            bmed = statistics.median(float(r["ttft_ms"]) for r in base)
            aq = statistics.mean(float(r["quality"]) for r in auto)
            bq = statistics.mean(float(r["quality"]) for r in base)
            lines.append(
                f"auto vs AR006: {bmed / amed:.2f}x faster, quality "
                f"{aq:.3f} vs {bq:.3f}."
            )
            lines.append("")

    notes = os.path.join(camp, "NOTES.md")
    if os.path.exists(notes):
        with open(notes, encoding="utf-8") as f:
            lines.append(f.read().rstrip())
        lines.append("")

    lines.append("## Audit columns")
    lines.append("")
    lines.append(
        "`source=model` means the budget came from the curve and the curve "
        "fingerprint matched the loaded model; `override` means the conf pinned "
        "it. `anchors` counts blocks pinned by the sink/entity/stride anchors, "
        "`mmr` counts blocks the diversity pass reordered."
    )
    out = os.path.join(camp, "REPORT.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"datasets={len(datasets)} samples={total_rows} -> {out}")
    return 0 if total_rows else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
