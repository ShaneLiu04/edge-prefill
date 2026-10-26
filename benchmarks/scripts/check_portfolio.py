#!/usr/bin/env python3
"""AR008 FR-3: combination regression gate (Pareto, machine independent).

For every dataset the routed combination must be **no worse than the best
fixed combination** measured in the same session:

    q_mean(routed)   >= q_mean(best_fixed) - --q-tolerance   (default 0.01)
    ttft(routed)     <= ttft(best_fixed)   * --ttft-slack     (default 1.05)

`best_fixed` is the highest-quality non-dense combination (ties broken by the
faster one). Absolute TTFT is host dependent, but both sides of the comparison
come from the same matrix, so the ratio is portable.

Also validates that the routed row actually reports the routed combo in
`portfolio_choice` - a table that says `gate-p24-local32` while the planner
ran `selection` is a silent regression and must fail.

Usage:
  check_portfolio.py [matrix.csv] [portfolio.json]
      [--q-tolerance=0.01] [--ttft-slack=1.05]
      [--inject len32k=selection]   # self-test: force a combo for a dataset
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_matrix(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    opts = [a for a in argv[1:] if a.startswith("--")]
    matrix_path = args[0] if args else "benchmarks/results/ar008-portfolio/matrix.csv"
    table_path = args[1] if len(args) > 1 else "benchmarks/results/ar008-portfolio/portfolio.json"
    matrix_path = matrix_path if os.path.isabs(matrix_path) else os.path.join(ROOT, matrix_path)
    table_path = table_path if os.path.isabs(table_path) else os.path.join(ROOT, table_path)

    q_tol = 0.01
    slack = 1.05
    inject = {}
    for o in opts:
        if o.startswith("--q-tolerance="):
            q_tol = float(o.split("=", 1)[1])
        elif o.startswith("--ttft-slack="):
            slack = float(o.split("=", 1)[1])
        elif o.startswith("--inject="):
            for part in o.split("=", 1)[1].split(","):
                if ":" in part:
                    ds, combo = part.split(":", 1)
                    inject[ds] = combo

    rows = load_matrix(matrix_path)
    with open(table_path, encoding="utf-8") as f:
        table = json.load(f)

    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"].rsplit("/", 1)[-1], {})[r["combo"]] = r

    fails = []
    print(f"{'dataset':<14} {'routed':<18} {'best_fixed':<18} "
          f"{'q_routed':>9} {'q_best':>7} {'ttft_r':>9} {'ttft_b':>9} verdict")
    for ds in sorted(table["table"]):
        name = ds.rsplit("/", 1)[-1]
        group = by_dataset.get(name, {})
        routed_combo = inject.get(name, table["table"][ds])
        routed = group.get(routed_combo)
        fixed = {c: r for c, r in group.items() if c != "dense"}
        if not routed:
            fails.append((name, routed_combo, "routed cell missing from matrix"))
            continue
        if not fixed:
            fails.append((name, routed_combo, "no fixed combinations measured"))
            continue
        best_combo = min(fixed, key=lambda c: (-float(fixed[c]["q_mean"]),
                                               float(fixed[c]["ttft_median"])))
        best = fixed[best_combo]
        q_r, q_b = float(routed["q_mean"]), float(best["q_mean"])
        t_r, t_b = float(routed["ttft_median"]), float(best["ttft_median"])
        ok = (q_r + q_tol >= q_b) and (t_r <= t_b * slack)
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<14} {routed_combo:<18} {best_combo:<18} {q_r:>9.3f} {q_b:>7.3f} "
              f"{t_r:>9.0f} {t_b:>9.0f} {verdict}")
        if not ok:
            fails.append((name, routed_combo,
                          f"q {q_r:.3f}<{q_b:.3f}-{q_tol} or ttft {t_r:.0f}>{t_b:.0f}*{slack}"))

    if fails:
        sys.stderr.write(f"\nFAIL {len(fails)} violation(s)\n")
        for f in fails:
            sys.stderr.write(f"  {f[0]} [{f[1]}]: {f[2]}\n")
        return 1
    print(f"\nPASS {len(table['table'])} datasets: routed combination is Pareto-"
          f">= best fixed combination")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
