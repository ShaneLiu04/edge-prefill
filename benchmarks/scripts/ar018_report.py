#!/usr/bin/env python3
"""AR018 gate: is the prefill/decode pipeline real, lossless and faster?

Reads benchmarks/results/ar018-disaggregated (matrix.csv + the raw cell CSVs)
and checks the claims AR018 makes. Every check is meant to fail on a
degradation; --inject runs the corresponding negative control.

  python3 benchmarks/scripts/ar018_report.py --gate
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPEEDUP_VS_SERIAL = 1.5
OVERLAP_MIN = 1.02
SPEEDUP_VS_BATCH = 1.1
HANDOFF_MAX_MS = 100.0
LATENCY_LIMIT = 4.0
HASH_FLIP_LIMIT = 0.05


def read_matrix(d):
    path = os.path.join(d, "matrix.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def best_per_arm(rows):
    best = {}
    for r in rows:
        key = (r["dataset"], r["arm"])
        wall = float(r["wall_ms"])
        if key not in best or wall < float(best[key]["wall_ms"]):
            best[key] = r
    return best


def cell_rows(d, dataset, arm):
    pat = os.path.join(d, dataset.replace("/", "-") + "-" + arm + "-r*", "*_raw.csv")
    files = sorted(glob.glob(pat))
    if not files:
        return {}
    # latest round wins: the directory name sorts r1..rN
    out = {}
    for r in csv.DictReader(open(files[-1], newline="", encoding="utf-8")):
        if r.get("dense") == "1":
            continue
        out[r["case"]] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="benchmarks/results/ar018-disaggregated")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--inject", default="")
    args = ap.parse_args(argv)

    d = os.path.join(ROOT, args.dir)
    rows = read_matrix(d)
    if args.inject == "no-overlap":
        for r in rows:
            if r["arm"] == "pipeline":
                r["overlap_ratio"] = "1.00"
    if args.inject == "drop-handoff":
        for r in rows:
            if r["arm"] == "pipeline":
                r["handoff_bytes"] = "0"
    if args.inject == "device-swap":
        for r in rows:
            if r["arm"] == "pipeline":
                r["prefill_device_state"] = "cpu"

    best = best_per_arm(rows)
    datasets = sorted({r["dataset"] for r in rows})
    results = []

    def add(name, ok, detail):
        results.append((name, ok, detail))

    # 1. Quality: the pipeline must produce the serial arm's answers.
    diffs = []
    for ds in datasets:
        serial = cell_rows(d, ds, "serial")
        pipe = cell_rows(d, ds, "pipeline")
        for case, srow in pipe.items():
            sref = serial.get(case)
            if not sref:
                continue
            if abs(float(srow["quality"]) - float(sref["quality"])) > 1e-9:
                diffs.append(f"{ds}:{case}")
    add("quality", not diffs,
        "pipeline f1 identical to serial" if not diffs
        else f"{len(diffs)} cases differ: {', '.join(diffs[:3])}")

    # 2. Answer stability (generation hash) -- AR014 float-tie policy.
    flips = []
    arms = 0
    for ds in datasets:
        serial = cell_rows(d, ds, "serial")
        for arm in ("batch", "gpuall", "pipeline"):
            other = cell_rows(d, ds, arm)
            for case, orow in other.items():
                sref = serial.get(case)
                if not sref:
                    continue
                arms += 1
                if orow["generated_hash"] != sref["generated_hash"]:
                    flips.append(f"{ds}:{arm}:{case}")
    ratio = len(flips) / arms if arms else 0.0
    add("answer-stability", ratio <= HASH_FLIP_LIMIT,
        f"{len(flips)}/{arms} arm-cases ({ratio*100:.1f}%) differ from serial "
        f"(limit {HASH_FLIP_LIMIT*100:.0f}%)")

    # 3. The KV really crossed as state.
    bad_handoff = []
    for ds in datasets:
        row = best.get((ds, "pipeline"))
        if row is None:
            bad_handoff.append(f"{ds}:missing")
            continue
        if int(row["handoff_bytes"]) <= 0:
            bad_handoff.append(f"{ds}:bytes=0")
        if float(row["handoff_ms"]) > HANDOFF_MAX_MS:
            bad_handoff.append(f"{ds}:{row['handoff_ms']}ms")
    add("handoff-integrity", not bad_handoff,
        "every pipelined case transferred KV state" if not bad_handoff
        else ", ".join(bad_handoff))

    # 4/5. Throughput against serial and against AR017 batching.
    speed, speed_batch = [], []
    for ds in datasets:
        s = best.get((ds, "serial"))
        b = best.get((ds, "batch"))
        p = best.get((ds, "pipeline"))
        if not (s and p):
            continue
        speed.append((ds, float(s["wall_ms"]) / float(p["wall_ms"])))
        if b:
            speed_batch.append((ds, float(b["wall_ms"]) / float(p["wall_ms"])))
    ok = bool(speed) and min(x[1] for x in speed) >= SPEEDUP_VS_SERIAL
    add("throughput-vs-serial", ok,
        "; ".join(f"{ds}=>{sp:.2f}x" for ds, sp in speed) +
        f" (need {SPEEDUP_VS_SERIAL:.1f}x)")
    ok = bool(speed_batch) and min(x[1] for x in speed_batch) >= SPEEDUP_VS_BATCH
    add("throughput-vs-batch", ok,
        "; ".join(f"{ds}=>{sp:.2f}x" for ds, sp in speed_batch) +
        f" (need {SPEEDUP_VS_BATCH:.2f}x)")

    # 6. Real overlap: device work must exceed wall clock.
    # Only datasets large enough to amortise two model loads can show overlap
    # (the stage sum includes start-up); small ones are informational.
    overlaps = [(ds, float(best[(ds, "pipeline")]["overlap_ratio"]),
                 int(best[(ds, "pipeline")]["cases"]))
                for ds in datasets if (ds, "pipeline") in best]
    measured = [(ds, ov) for ds, ov, n in overlaps if n >= 8]
    ok = bool(measured) and min(x[1] for x in measured) >= OVERLAP_MIN
    add("overlap", ok,
        "; ".join(f"{ds}=>{ov:.2f}" for ds, ov, n in overlaps) +
        f" (gated on >=8-case datasets, need {OVERLAP_MIN:.2f})")

    # 7. Placement is what was asked for.
    bad_place = []
    for ds in datasets:
        row = best.get((ds, "pipeline"))
        if row is None:
            continue
        if row["prefill_device_state"] != "gpu":
            bad_place.append(f"{ds}:prefill={row['prefill_device_state']}")
        if row["decode_device_state"] != "cpu":
            bad_place.append(f"{ds}:decode={row['decode_device_state']}")
    add("placement", not bad_place,
        "prefill on gpu, decode on cpu, both verified" if not bad_place
        else ", ".join(bad_place))

    # 8. Honest latency: pipelining trades per-case latency for throughput, and
    # the report must not hide it behind a throughput-only claim.
    lat = []
    for ds in datasets:
        b = best.get((ds, "batch"))
        p = best.get((ds, "pipeline"))
        if b and p and float(b["e2e_median"]) > 0:
            lat.append((ds, float(p["e2e_median"]) / float(b["e2e_median"])))
    ok = bool(lat) and max(x[1] for x in lat) <= LATENCY_LIMIT
    add("latency-honesty", ok,
        "; ".join(f"{ds}=>{r:.2f}x batch e2e" for ds, r in lat) +
        f" (limit {LATENCY_LIMIT:.1f}x)")

    # 9. The other arms still are what they claim.
    bad = []
    for ds in datasets:
        s, b = best.get((ds, "serial")), best.get((ds, "batch"))
        if s and int(s["batch_effective"]) != 1:
            bad.append(f"{ds}:serial batch={s['batch_effective']}")
        if b and int(b["batch_effective"]) < 2:
            bad.append(f"{ds}:batch batch={b['batch_effective']}")
    add("arms-as-labelled", not bad, "serial=1 sequence, batch>1 sequence"
        if not bad else ", ".join(bad))

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}: {detail}")
    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print("GATE FAILED")
        return 1
    print("all gate checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
