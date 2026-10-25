# AR007 acceptance campaign

Measured with `benchmarks/scripts/ar007_accept.sh` in one session per dataset. `auto` lets the planner read `budget_curve.txt`; the AR006 column is the fixed 0.25 budget for comparison.

## multineedle

dense TTFT median 53420 ms over 6 cases.

| config | cases | q mean | q min | TTFT median ms | speedup | budget | source | anchors | mmr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 6 | 1.000 | 1.000 | 53420 | 1.0x | 1 | override | 0-0 | 0 |
| auto (curve) | 6 | 0.333 | 0.000 | 1752 | 30.5x | 0.05 | model | 7-7 | 0 |
| AR006 b=0.25 | 6 | 0.394 | 0.000 | 9368 | 5.7x | 0.25 | override | 20-21 | 0 |
| qg p16 | 6 | 0.833 | 0.000 | 9674 | 5.5x | 0.25 | override | 20-21 | 0 |

auto vs AR006: 5.35x faster, quality 0.333 vs 0.394.

## len32k

dense TTFT median 384145 ms over 4 cases.

| config | cases | q mean | q min | TTFT median ms | speedup | budget | source | anchors | mmr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 4 | 1.000 | 1.000 | 384145 | 1.0x | 1 | override | 0-0 | 0 |
| auto (curve) | 4 | 0.893 | 0.571 | 6956 | 55.2x | 0.05 | model | 16-17 | 0 |
| AR006 b=0.25 | 4 | 0.750 | 0.000 | 21949 | 17.5x | 0.1315983936,0.1316660773,0.1319120157,0.1320225624 | clamp | 36-38 | 0 |
| qg p24-local32 | 4 | 1.000 | 1.000 | 31433 | 12.2x | 0.1315983936,0.1316660773,0.1319120157,0.1320225624 | clamp | 36-38 | 0 |

auto vs AR006: 3.16x faster, quality 0.893 vs 0.750.

## len16k

dense TTFT median 199494 ms over 4 cases.

| config | cases | q mean | q min | TTFT median ms | speedup | budget | source | anchors | mmr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 4 | 1.000 | 1.000 | 199494 | 1.0x | 1 | override | 0-0 | 0 |
| auto (curve) | 4 | 1.000 | 1.000 | 5176 | 38.5x | 0.05 | model | 8-9 | 0 |
| AR006 b=0.25 | 4 | 0.356 | 0.000 | 19909 | 10.0x | 0.25 | override | 35-36 | 0 |
| qg p16 | 4 | 1.000 | 1.000 | 26664 | 7.5x | 0.25 | override | 35-36 | 0 |

auto vs AR006: 3.85x faster, quality 1.000 vs 0.356.

## len8k

dense TTFT median 48803 ms over 4 cases.

| config | cases | q mean | q min | TTFT median ms | speedup | budget | source | anchors | mmr |
|---|---|---|---|---|---|---|---|---|---|
| dense | 4 | 1.000 | 1.000 | 48803 | 1.0x | 1 | override | 0-0 | 0 |
| auto (curve) | 4 | 1.000 | 1.000 | 1674 | 29.1x | 0.05 | model | 7-7 | 0 |
| AR006 b=0.25 | 4 | 0.875 | 0.500 | 9085 | 5.4x | 0.25 | override | 19-21 | 0 |

auto vs AR006: 5.43x faster, quality 1.000 vs 0.875.

## Findings

- **The curve wins on the retrieval family.** 8K 29.1x / 16K 38.5x / 32K 55.2x
  against dense, all in one session, with quality 1.00 / 1.00 / 0.89. The
  AR006 fixed 0.25 budget in the same session is 5.4x / 10.0x / 17.5x with
  quality 0.875 / 0.356 / 0.750: more budget was not more quality.
- **Non-monotone budget is the headline.** 16K at b=0.25 scores 0.356 while
  b=0.05 scores 1.000 on the same 4 cases. The extra budget goes to
  proportional anchors and stride blocks, which crowd out the lexical
  evidence. This is why the sweep rule picks the cheapest cell within the
  speed tolerance instead of walking up from the smallest budget.
- **32K spends its budget on a floor, not on the curve.** b=0.05 would be
  1556 tokens; the planner's 8192-token ceiling does not bind, the effective
  run is 1565 selected tokens (`budget_source=model`) and still scores 0.893.
  The b=0.25 row is clamped to 4096 tokens (`budget_source=clamp`) and only
  reaches 0.750, i.e. the clamp itself is not a quality guarantee.
- **Multi-needle is a known weakness of selection-level sparsity.** auto scores
  0.333, b=0.08/0.12/0.25 score 0.333/0.500/0.394 (measured separately in
  `ar007-mn-probe/`), while AR006's in-graph page gate with a 32-row local
  window scores 0.83-1.00 on the same family. The `budget_curve` path is
  therefore opt-in only; multi-needle runs should use `qg_bound=on`.
- **Page-score de-copy is worth 2.2-2.7x at 32K.** AR006's gate (cont+pool
  page means) took 67.9 s at 32K p24+local32; the view+`mul_mat`+page-sum form
  takes 25.4 s on an idle host and 31.4 s in this campaign (56-63% TTFT
  reduction, target was >=30%), q=1.00 both times. 16K p16 goes 23.7 s ->
  21.8 s idle / 26.7 s loaded, q=1.00 in all runs.
- **The K operand shape is a live regression class.** `ggml_mul_mat(a, b)`
  only requires `b->ne[2] % a->ne[2] == 0`; feeding the raw cache tensor
  `[d, h_kv, n_kv]` aborts at 32K. The fix uses the `[d, n_kv, h_kv]` permuted
  view, and `bound_gating_execution_and_restore` now runs a `qg_local_rows=32`
  case so the page-scoring branch (not the local-only bypass) is exercised
  without a model download.

Absolute TTFT varies up to ~3x between sessions on this host; only same-session
ratios are comparable. Raw samples: `<dataset>-<config>/*_raw.csv`, one env JSON
per run with the model/data SHA256.

## Audit columns

`source=model` means the budget came from the curve and the curve fingerprint matched the loaded model; `override` means the conf pinned it. `anchors` counts blocks pinned by the sink/entity/stride anchors, `mmr` counts blocks the diversity pass reordered.
