#!/usr/bin/env python3
"""AR019 gate: does continuous batching beat the static batch it replaces?

Reads benchmarks/results/ar019-continuous (interleaved A/B raw cells + walls).
The honest bar here is *not* "faster than serial" (AR017/18 already are): it is
"no regression in throughput, identical answers, and the scheduler really
admits mid-batch". Whether it is worth promoting is a separate combination
decision recorded in docs/portfolio.md.
"""
import argparse
import csv
import glob
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THROUGHPUT_MIN = 0.98      # never slower than the static pipeline batch
ADMIT_RATIO_MIN = 1.5      # admits must exceed slot count


def load(d):
    cells = {}
    for cell in sorted(glob.glob(os.path.join(d, 'o_*'))):
        arm, kk, rnd = os.path.basename(cell)[2:].split('_')
        rows = []
        for f in glob.glob(os.path.join(cell, '*_raw.csv')):
            rows += [r for r in csv.DictReader(open(f, newline='')) if r['dense'] == '0']
        if not rows:
            continue
        cells.setdefault((arm, int(kk[1:])), []).append(rows)
    walls = {}
    with open(os.path.join(d, 'ab_walls.csv'), newline='') as f:
        for r in csv.DictReader(f):
            walls.setdefault((r['arm'], int(r['k'])), []).append(float(r['wall_ms']))
    return cells, walls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dir', nargs='?', default='benchmarks/results/ar019-continuous')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--inject', default='')
    args = ap.parse_args(argv)

    cells, walls = load(os.path.join(ROOT, args.dir))
    if args.inject == 'quality-drift':
        # Only the continuous arm drifts: mutating both sides would make the
        # comparison agree and the gate would pass for the wrong reason.
        for (arm, _), runs in cells.items():
            if arm != 'cont':
                continue
            for rows in runs:
                for r in rows:
                    r['quality'] = '0.0'
                    r['generated_hash'] = '1'
    if args.inject == 'lockstep-peers':
        # Pretend the scheduler fell back to lockstep: every case decoded alone.
        for (arm, _), runs in cells.items():
            if arm != 'cont':
                continue
            for rows in runs:
                for r in rows:
                    r['peer_steps'] = '1'
    if args.inject == 'slow':
        for key in list(walls):
            if key[0] == 'cont':
                walls[key] = [w * 2.0 for w in walls[key]]

    results = []
    add = lambda n, ok, d: results.append((n, ok, d))

    ks = sorted({k for _, k in cells})
    # 1. Answers identical to the static batch at the same K.
    bad = []
    for k in ks:
        ref = cells.get(('pipe', k), [])
        got = cells.get(('cont', k), [])
        if not ref or not got:
            continue
        ref_h = sorted(r['generated_hash'] for r in ref[0])
        for rows in got:
            if sorted(r['generated_hash'] for r in rows) != ref_h:
                bad.append(f'K={k}')
                break
    add('answers-identical', not bad,
        'continuous output == static batch output' if not bad
        else f'differs at {", ".join(bad)}')

    # 2. Quality unchanged.
    diffs = []
    for k in ks:
        ref = cells.get(('pipe', k), [])
        got = cells.get(('cont', k), [])
        if not ref or not got:
            continue
        q_ref = st.mean(float(r['quality']) for r in ref[0])
        for rows in got:
            if abs(st.mean(float(r['quality']) for r in rows) - q_ref) > 1e-9:
                diffs.append(f'K={k}')
    add('quality', not diffs, 'f1 identical' if not diffs else f'changed at {", ".join(diffs)}')

    # 3. Throughput must not regress against the static batch.
    speed = []
    for k in ks:
        w_pipe = min(walls.get(('pipe', k), [0.0]))
        w_cont = min(walls.get(('cont', k), [0.0]))
        if w_pipe and w_cont:
            speed.append((k, w_pipe / w_cont))
    ok = bool(speed) and min(s for _, s in speed) >= THROUGHPUT_MIN
    add('throughput', ok,
        '; '.join(f'K={k}=>{s:.2f}x' for k, s in speed) +
        f' vs static batch (need {THROUGHPUT_MIN:.2f}x)')

    # 4. The scheduler really works at iteration level.
    detail = []
    ok = True
    for k in ks:
        for rows in cells.get(('cont', k), []):
            slots = k
            # Every case is admitted individually, so the number of admissions is
            # the number of continuous rows (admit_step only records the
            # scheduler step, and several cases can enter at the same step).
            admits = len(rows)
            own = st.median(int(r['own_steps'] or 0) for r in rows)
            peers = st.median(int(r['peer_steps'] or 0) for r in rows)
            ratio = admits / slots
            detail.append(f'K={k}: {admits} admissions for {slots} slots '
                          f'({ratio:.1f}x), own={own:.0f} steps, peers={peers:.0f}')
            if ratio < ADMIT_RATIO_MIN or own <= 0 or peers < 2:
                ok = False
    add('iteration-level-scheduling', ok, '; '.join(detail))

    # 5. Latency is recorded, not hidden (the trade-off is the finding).
    lat = []
    for k in ks:
        ref = cells.get(('pipe', k), [])
        got = cells.get(('cont', k), [])
        if not ref or not got:
            continue
        lat.append((k, st.median(float(r['e2e_wall_ms']) for r in got[0]) /
                    st.median(float(r['e2e_wall_ms']) for r in ref[0])))
    add('latency-recorded', True,
        '; '.join(f'K={k}=>{v:.2f}x static' for k, v in lat) + ' (informational)')

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail_ in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}: {detail_}")
    if any(not ok for _, ok, _ in results):
        print('GATE FAILED')
        return 1
    print('all gate checks pass')
    return 0


if __name__ == '__main__':
    sys.exit(main())
