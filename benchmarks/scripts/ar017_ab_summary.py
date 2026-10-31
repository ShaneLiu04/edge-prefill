import csv, glob, os, statistics as st, sys

out = sys.argv[1]
dataset = sys.argv[2] if len(sys.argv) > 2 else "deep/multifact8k"
dest = sys.argv[3] if len(sys.argv) > 3 else None
rows = {}
records = []
for f in glob.glob(out + '/o_*/*_raw.csv'):
    key = f.split('/')[-2]
    _, k, r = key.split('_')
    data = [x for x in csv.DictReader(open(f, newline='')) if x['dense'] == '0']
    if not data:
        continue
    ttft = st.median(float(x['ttft_min_ms']) for x in data)
    e2e = st.median(float(x['e2e_ms']) for x in data)
    eff = max(int(x['batch_size'] or 1) for x in data)
    bw = st.median(float(x['batch_wall_ms'] or 0) for x in data)
    steps = st.median(int(x['batch_steps'] or 0) for x in data)
    nb = (len(data) + eff - 1) // eff
    dec = bw * nb if eff > 1 and bw > 0 else sum(
        max(0.0, float(x['e2e_ms']) - float(x['ttft_min_ms'])) for x in data)
    q = st.mean(float(x['quality']) for x in data)
    rec = dict(dataset=dataset, k=int(k), round=int(r), eff=eff, steps=steps,
               ttft_ms=ttft, e2e_median_ms=e2e, decode_wall_ms=dec, quality=q,
               cases=len(data),
               generated_hash=','.join(sorted({x['generated_hash'] for x in data})))
    rows.setdefault(int(k), []).append(rec)
    records.append(rec)

print('%-3s %-5s %6s %7s %8s %8s %8s %6s' % (
    'K', 'round', 'eff', 'steps', 'ttft', 'e2e_med', 'dec_wall', 'q'))
for k in sorted(rows):
    for rec in sorted(rows[k], key=lambda x: x['round']):
        print('%-3d %-5d %6d %7d %8.0f %8.0f %8.0f %6.3f' % (
            k, rec['round'], rec['eff'], rec['steps'], rec['ttft_ms'],
            rec['e2e_median_ms'], rec['decode_wall_ms'], rec['quality']))

print()
best = {k: min(v, key=lambda x: x['decode_wall_ms']) for k, v in rows.items()}
base = best.get(1)
for k in sorted(best):
    b = best[k]
    if base and b['decode_wall_ms'] > 0:
        print('K=%-2d best: decode_wall=%.0f ms (%.2fx) e2e_med=%.0f (%.2fx) ttft=%.0f '
              'q=%.3f eff=%d' % (
                  k, b['decode_wall_ms'], base['decode_wall_ms'] / b['decode_wall_ms'],
                  b['e2e_median_ms'], base['e2e_median_ms'] / b['e2e_median_ms'],
                  b['ttft_ms'], b['quality'], b['eff']))

if dest:
    exists = os.path.exists(dest)
    with open(dest, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(records)
    print('appended', len(records), 'rows to', dest)
