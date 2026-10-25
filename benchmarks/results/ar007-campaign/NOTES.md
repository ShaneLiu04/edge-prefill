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
