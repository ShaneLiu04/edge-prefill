#!/usr/bin/env python3
"""AR015 gate: warm-start must be faster and bit-identical to recompute.

Checks, per dataset:
  hit      every warm case restored from a snapshot (no silent fallback)
  quality  warm f1 == cold f1 for every case
  latency  warm ttft_min <= LATENCY_BOUND x cold ttft_min
  size     snapshot <= SIZE_BOUND x the KV the run actually used
and for the multi-turn dataset the resume arm must replay the conversation
with the same quality as the cold arm.

  --inject <dataset>:no-hit         pretend nothing was restored -> latency FAIL
  --inject <dataset>:quality-drift  pretend a restored answer changed -> quality FAIL
"""
import argparse
import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LATENCY_BOUND = 0.5       # warm must be at most half the cold TTFT
RESUME_BOUND = 0.5        # resumed turns must be at most half the cold replay
SIZE_BOUND = 1.30         # snapshot (state + logits) vs the KV the run used
MULTITURN = "deep/multiturn"


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def evaluate(result_dir, injects):
    matrix = load_rows(os.path.join(result_dir, "matrix.csv"))
    cases = load_rows(os.path.join(result_dir, "cases.csv"))
    checks = []

    def add(check, scope, ok, detail):
        checks.append({"check": check, "scope": scope, "ok": bool(ok), "detail": detail})

    by_arm = {}
    for row in cases:
        by_arm.setdefault(row["dataset"], {}).setdefault(row["arm"], {})[row["case"]] = row

    for dataset, arms in sorted(by_arm.items()):
        cold = arms.get("cold", {})
        warm = arms.get("warm", {})
        if not cold or not warm:
            add("coverage", dataset, False, "missing cold/warm arm")
            continue

        # hit accounting: a warm case that did not restore is a silent fallback
        misses = [c for c, r in warm.items() if r.get("snapshot_hit") != "1"]
        add("hit", dataset, not misses,
            f"{len(warm) - len(misses)}/{len(warm)} cases restored"
            + (f" (misses: {', '.join(sorted(misses)[:3])})" if misses else ""))

        # quality: same answers, case by case
        drift = []
        for case, wrow in warm.items():
            crow = cold.get(case)
            if crow is None:
                continue
            if abs(fnum(wrow, "quality") - fnum(crow, "quality")) > 1e-9:
                drift.append((case, fnum(crow, "quality"), fnum(wrow, "quality")))
        add("quality", dataset, not drift,
            "identical f1"
            if not drift else f"{len(drift)} case(s) differ, e.g. {drift[0]}")

        # latency from the cell summary (median of per-case ttft_min)
        cell = {(r["dataset"], r["arm"]): r for r in matrix}
        cold_cell = cell.get((dataset, "cold"))
        warm_cell = cell.get((dataset, "warm"))
        if cold_cell and warm_cell:
            c_ttft = fnum(cold_cell, "ttft_min")
            w_ttft = fnum(warm_cell, "ttft_min")
            ratio = w_ttft / c_ttft if c_ttft > 0 else 9.9
            add("latency", dataset, ratio <= LATENCY_BOUND,
                f"cold {c_ttft:.0f}ms -> warm {w_ttft:.1f}ms ({ratio:.3f}x, "
                f"bound {LATENCY_BOUND})")
            w_bytes = max(fnum(r, "snapshot_bytes") for r in warm.values())
            kv_used = max(fnum(r, "kv_used_bytes") for r in warm.values())
            if kv_used > 0:
                add("size", dataset, w_bytes <= SIZE_BOUND * kv_used,
                    f"snapshot {w_bytes / 1e6:.1f} MB vs kv_used {kv_used / 1e6:.1f} MB")
            else:
                add("size", dataset, False, "no kv_used_bytes in the warm arm")

        # multi-turn: replay-after-restart must match the cold replay
        resume = arms.get("resume")
        if resume:
            drift = []
            for case, rrow in resume.items():
                crow = cold.get(case)
                if crow is None:
                    continue
                same_hash = (rrow.get("session_log_hash") == crow.get("session_log_hash"))
                same_q = abs(fnum(rrow, "quality") - fnum(crow, "quality")) <= 1e-9
                if not (same_hash and same_q):
                    drift.append(case)
            add("resume", dataset, not drift,
                "every resumed turn matches the cold replay"
                if not drift else f"{len(drift)} turn(s) differ, e.g. {drift[0]}")
            resume_cell = cell.get((dataset, "resume"))
            if cold_cell and resume_cell:
                c_ttft = fnum(cold_cell, "ttft_min")
                r_ttft = fnum(resume_cell, "ttft_min")
                ratio = r_ttft / c_ttft if c_ttft > 0 else 9.9
                add("resume-latency", dataset, ratio <= RESUME_BOUND,
                    f"replay {c_ttft:.0f}ms -> resume {r_ttft:.1f}ms ({ratio:.3f}x, "
                    f"bound {RESUME_BOUND})")

    for spec in injects:
        dataset, kind = spec.split(":", 1)
        if kind == "no-hit":
            for c in checks:
                if c["check"] == "hit" and c["scope"] == dataset:
                    c["ok"] = False
                    c["detail"] += " [injected: nothing restored]"
        elif kind == "quality-drift":
            for c in checks:
                if c["check"] == "quality" and c["scope"] == dataset:
                    c["ok"] = False
                    c["detail"] += " [injected: answer drift]"
    return checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="benchmarks/results/ar015-kv-snapshot")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--inject", action="append", default=[],
                    help="<dataset>:no-hit | <dataset>:quality-drift")
    args = ap.parse_args(argv)

    path = args.results
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(os.path.join(path, "cases.csv")):
        sys.stderr.write(f"missing cases.csv under {path}\n")
        return 2
    checks = evaluate(path, args.inject)
    width = max(len(c["scope"]) for c in checks)
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"{mark} {c['check']:<15} {c['scope']:<{width}}  {c['detail']}")
    failed = [c for c in checks if not c["ok"]]
    if failed:
        print(f"\n{len(failed)} gate check(s) FAILED")
        return 1 if args.gate else 0
    print("\nall gate checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
