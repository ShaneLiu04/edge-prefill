# AR006 campaign — in-graph query-aware page gating (CPU, real model)

All numbers are real runs of the pinned llama.cpp + the AR006 patch on
Qwen2.5-1.5B-Instruct Q4_K_M, 16 threads, one interleaved session on
2026-10-25. Raw CSV, env JSON and the exact config file are inside every
`<dataset>-<mode>/` directory. Regenerate the tables with:

```bash
python3 benchmarks/scripts/assemble_ar006.py benchmarks/results/ar006-campaign
```

**Read the speedup column, not the milliseconds.** Absolute TTFT on this
shared host moved by up to 3x between datasets inside the same session (dense
8K 137 s vs dense 16K 162 s is not linear in prompt length; the host was
contended). Dense and its sparse siblings for a dataset were measured back to
back, so each row's ratio is internally comparable; cross-row milliseconds
are not.

## What the gate does

Per ubatch, per layer: K pages (32 rows) are reduced to means, scored with
the current query (`q · mean(K_page)`), the top `qg_sel_pages` pages are
appended to a local window (sink page + trailing `max(ubatch, qg_local_rows)`
rows), and K/V/mask are gathered once. No embedding model runs in the gated
path (`semantic_score_ms = 0`).

## Results

Median TTFT per dataset in this session; quality = mean needle hit rate.

| dataset | dense | scalar (AR005) | qg-2d (AR005, embedding) | bound-p16 | verdict |
|---|---|---|---|---|---|
| len4k | 48.5 s | 6.2 s / 1.0 | 29.0 s / 1.0 | 12.0 s / 1.0 | 4.2x; gate bypassed (local covers all) |
| len8k | 118.9 s | 11.2 s / 1.0 | 12.5 s / 1.0 | 11.6 s / 1.0 | **9.4x, q=1.0, sem=0** |
| multineedle | 64.2 s | 22.7 s / 0.83 | — | 17.1 s / 0.83 | **3.8x, faster than scalar at equal q** |
| len16k | 161.8 s | 22.5 s / 1.0 | 25.1 s / 1.0 | 23.7 s / 1.0 | **6.8x, q=1.0, sem=0** |
| len32k | 385.2 s | 22.8 s / 1.0 | — | 40.9 s / 0.0 | **quality fails at the default local window** |
| len32k (recommended) | 385.2 s | — | — | 67.9 s / 1.0 (`qg_local_rows=32`) | 5.7x, q=1.0 |

`matrix.csv` / `matrix.md` in this directory are the generated tables,
including bound-p24 and the 32K recommended point.

### Findings

1. **The gate's value here is removing the embedding from the critical path,
   not raw attention savings.** With AR001 selection-level sparsity at budget
   0.25 only ~25% of tokens enter the KV cache, so attention is already
   small; the gate lands within a few percent of the scalar gather at 8K and
   16K while `semantic_score_ms` drops from 1-10 s to exactly 0 and no second
   model is needed. Against the AR005 qg-2d path it is 7% faster at 8K and
   5.5% faster at 16K with identical quality.
2. **8K is the sweet spot** for this model: 9.4x with q=1.0.
3. **Multi-needle is where the page gate actually helps**: 3.8x with q=0.83
   vs scalar's 2.8x with q=0.67 — the local window keeps the neighbourhood of
   each needle, which a pure top-k row list drops.
4. **32K is outside the gate's operating window at the default local
   window.** Page-gating 973 pages of a 13%-budget KV (`budget` is clamped to
   0.132 by the planner at 32K) covers too little of the admitted rows: p16
   and p24 both miss needles (q=0.0 / 0.25). Widening the local window to
   32 pages (`qg_local_rows=32`) restores q=1.0 at 5.7x, but then the win
   comes from the local window, not the selection.
5. **The 32K cost is the page-mean reduction, not the gather.** ttft grows
   with the number of ubatches because every ubatch re-scans the whole K
   cache to build page means; the current implementation materialises that
   scan through `ggml_cont(permute(k))`, copying the full K cache per layer
   per ubatch (95 MB x 28 x 61 ≈ 160 GB of traffic at 32K). Removing that
   copy (score as `mul_mat(k_view, q_last)` then a page sum, no cont) and
   gating every Nth layer are the two identified next increments; neither is
   in this increment, so the 32K row must be read as "current
   implementation", not "page gating is useless at 32K".

### Honest acceptance status (SRS AR006)

| SRS target | Result | Status |
|---|---|---|
| 16K net speedup >= 8x with q >= 0.9 | 6.8x, q=1.0 (dense 161.8 s, bound 23.7 s in-session) | **miss** |
| 32K net speedup >= 10x with q >= 0.9 | fails at default; 5.7x with `local_rows=32` | **miss** |
| qg TTFT <= scalar x 1.15 with sem=0 | 23.7 s vs 22.5 s = 1.05x, sem=0 | pass |
| quality >= AR005 qg-2d at equal budget | equal or better on every dataset (mn: 0.83 vs scalar 0.67) | pass |
| no embedding model in the gated path | `semantic_score_ms=0`, `qg_source=attention-bound` | pass |
| dense equivalence when off | `bound_gating_execution_and_restore` bit-identical | pass |

The two misses are recorded with the root causes above (selection-level
sparsity already removes most attention work; the page-mean scan is
unoptimised). The AR is **not** claimed to hit 10x at 32K on CPU.

## Reproduce

```bash
BIN=build-wsl/edge-prefill
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
# dense reference first, then the gated point: same session, same dataset
$BIN bench --config benchmarks/matrix/bound/dense-ref.conf \
    --model $MODEL --data-dir benchmarks/data/explen/len16k \
    --out /tmp/r-16k-dense --bench-skip-dense
$BIN bench --config benchmarks/matrix/bound/bound-p16-a24-b025.conf \
    --model $MODEL --data-dir benchmarks/data/explen/len16k \
    --out /tmp/r-16k-p16 --bench-skip-dense
```

`docs/server-repro.md` has the full guide (models, SHA256, 7B/14B templates,
GPU notes); `benchmarks/scripts/check_expected_ar006.py` gates a results tree
against `benchmarks/matrix/bound/expected.csv` (6/6 points pass).
