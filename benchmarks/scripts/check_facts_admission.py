#!/usr/bin/env python3
"""AR012: deterministic admission audit - no model scoring involved.

For each dumped case (`dump_selection_dir`), count how many of the asked fact
answers are *fully* present in the admitted context. This measures the layer we
control (admission) instead of the model's ability to copy strings, which is
noisy across prompt variants.

Usage:
  check_facts_admission.py <results-dir> [--dataset deep/multifact8k-hard]

Reads  <results-dir>/<dataset>-<arm>/selected/<case>_selected.txt
       benchmarks/data/<dataset>/index.tsv
Prints one line per arm and exits non-zero if an arm drops below --floor.
"""
import argparse
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--dataset", default="deep/multifact8k-hard")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="minimum mean fact completeness (0 = report only)")
    ap.add_argument("--csv", default="",
                    help="write arm,facts_in_context,all_facts,cases to this file")
    ap.add_argument("--gate-arm", default="",
                    help="only this arm is gated; other arms are reported for "
                         "comparison (baselines are expected to sit lower)")
    args = ap.parse_args(argv)

    index = os.path.join(ROOT, "benchmarks/data", args.dataset, "index.tsv")
    answers = collections.defaultdict(list)
    with open(index, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ans_col = header.index("answer")
        id_col = header.index("id")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            answers[cols[id_col]].append(cols[ans_col].split())

    name = args.dataset.rsplit("/", 1)[-1]
    # Longest-first so "multifact8k" does not swallow "multifact8k-hard-*".
    all_names = sorted({d for d in os.listdir(os.path.join(ROOT, "benchmarks/data/deep"))}
                       | {d for d in os.listdir(os.path.join(ROOT, "benchmarks/data/explen"))},
                       key=len, reverse=True)
    rows = []
    root = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    for arm_dir in sorted(os.listdir(root)):
        owner = next((n for n in all_names if arm_dir.startswith(n + "-")), None)
        if owner != name:
            continue
        sel_dir = os.path.join(root, arm_dir, "selected")
        if not os.path.isdir(sel_dir):
            continue
        arm = arm_dir[len(name) + 1:]
        fracs = []
        for fn in sorted(os.listdir(sel_dir)):
            if not fn.endswith("_selected.txt"):
                continue
            case = fn[: -len("_selected.txt")]
            facts = answers.get(case)
            if not facts:
                continue
            text = open(os.path.join(sel_dir, fn), encoding="utf-8", errors="replace").read()
            codes = facts[0]
            fracs.append(sum(1 for c in codes if c in text) / len(codes))
        if fracs:
            mean = sum(fracs) / len(fracs)
            rows.append((arm, mean, sum(1 for f in fracs if f > 0.999), len(fracs)))

    if not rows:
        sys.stderr.write(f"no dumped selections under {root} for {name}\n")
        return 2
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="\n") as f:
            f.write("dataset,arm,facts_in_context,all_facts,cases\n")
            for arm, mean, full, n in rows:
                f.write(f"{args.dataset},{arm},{mean:.4f},{full},{n}\n")

    failures = 0
    for arm, mean, full, n in rows:
        gated = (not args.gate_arm) or arm == args.gate_arm
        flag = ""
        if gated and mean < args.floor:
            flag = "  <-- below floor"
            failures += 1
        elif not gated and mean < args.floor:
            flag = "  (baseline, not gated)"
        print(f"{arm:<14} facts_in_context={mean:.3f} all_facts={full}/{n}{flag}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
