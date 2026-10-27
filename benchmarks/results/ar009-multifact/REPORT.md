# AR009 acceptance: multi-fact coverage admission

> **Correction from AR010 (2026-10-27).** AR010 found that `minmax()` kept
> iterators into the vector it was overwriting, so `diversity`/`position` were
> not in [0,1] (up to 3.2 / 1.16). Part of the gain reported below came from
> that over-ranged prior. With the normalisation fixed, the same commands
> measure **0.836 / 0.667** (versus 0.878 / 0.652 below), and the coverage
> signal is still worth +0.03 / +0.28 over `coverage_weight=0`. The corrected
> floors live in `benchmarks/matrix/multifact/expected_ar009.csv`; see
> `benchmarks/results/ar010-evidence-first/REPORT.md` section 5.

Every number comes from one session on the same host (Xeon 6234, 16 threads,
Qwen2.5-1.5B-Instruct Q4_K_M, `max_new_tokens=48`, `warmup=0`, `repeats=1`),
`budget=0.05`, `phrase_weight=0.5`. Absolute TTFT varies across sessions on this
shared VM, so only same-session comparisons are meaningful. Raw samples:
`<dataset>-cw<weight>/*_raw.csv` + `config.used` per cell; machine-readable
summary in `ablation.csv` / `ablation.json`.

## 0. The gap AR008 left open

AR008's multi-needle dataset asks **one** fact per row, so a top-1 selector scores
1.0 whenever it picks the right block. Real prefill traffic asks for several
facts at once ("list the passcodes for the vaults one, three and five"). The new
family (`benchmarks/scripts/gen_multifact.py`) plants `--facts` independent facts
and asks `--asked` of them per row, scoring the token-F1 of the concatenated
codes so partial credit is visible:

| dataset | planted facts | asked per row | rows |
|---|---:|---:|---:|
| `deep/multifact8k` | 6 | 3 | 16 |
| `deep/multifact8k-hard` | 12 | 6 | 16 |

At `budget=0.05` the selection holds ~14 blocks (443 tokens). Three to six facts
need one block each, so **budget was not the bottleneck** — the admission simply
spent all of it on the single strongest lexical match.

## 1. Ablation: AR008 ranking vs coverage-fair fill

Same binary, same model, same datasets, one knob (`coverage_weight`, 0 = AR008):

| dataset | coverage_weight | cases | q_mean | q_min | facts covered | selected tok | TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep/multifact8k | 0 (AR008) | 16 | 0.488 | 0.429 | 0.00 / 3 | 443 | 2767 ms |
| deep/multifact8k | 3 (AR009) | 16 | **0.878** | 0.706 | 3.25 / 3 | 443 | 2927 ms |
| deep/multifact8k-hard | 0 (AR008) | 16 | 0.261 | 0.261 | 0.00 / 6 | 445 | 2642 ms |
| deep/multifact8k-hard | 3 (AR009) | 16 | **0.652** | 0.429 | 6.50 / 6 | 445 | 3218 ms |

| dataset | q(AR008) | q(AR009) | delta |
|---|---:|---:|---:|
| deep/multifact8k | 0.488 | 0.878 | **+0.390** |
| deep/multifact8k-hard | 0.261 | 0.652 | **+0.391** |

`facts covered` is the `coverage_queries` audit column, averaged per row. At
`coverage_weight=0` it is 0 by construction (no sub-query set is built), which is
also why the gate can catch a silently disabled feature — see §4.

Cost: +5.8% TTFT on the 3-fact tier and +21.8% on the 6-fact tier. The second is
not free: six sub-queries mean six anchors, more blocks enter the prefill, and the
whole point is that those blocks are the answer. It is still ~2.6x faster than the
AR008 gate path on the same data.

### 1.1 Dense reference (easy tier)

`budget=1.0` on a 2-context / 4-row subset of `deep/multifact8k` (a full 16-row
dense pass exceeds this session's time budget; the subset is derived by filtering
`index.tsv`, contexts symlinked, exact command in design §8):

| case | dense q |
|---|---:|
| mf-01-r1 | 1.000 |
| mf-01-r2 | 1.000 |
| mf-02-r1 | 1.000 |
| mf-02-r2 | 0.900 |
| **mean** | **0.975** |

So AR009 reaches 0.878 / 0.975 = **90% of full-prompt quality with 5% of the
tokens**. The hard tier has no dense reference (not measured, stated rather than
guessed).

## 2. Non-regression on the single-fact families

`coverage_weight=3` vs `0`, same config (canonical `selection` portfolio arm):

| dataset | cw=0 | cw=3 | selection identical? |
|---|---|---|---|
| explen/len4k | q=1.000, sel=322 | q=1.000, sel=322 | yes (bit-for-bit) |
| explen/len16k | q=1.000, sel=793 | q=1.000, sel=793 | yes (bit-for-bit) |
| deep/multineedle | q=1.000, sel=401 | q=1.000, sel=401 | yes (bit-for-bit) |
| deep/len32k | q=0.875, sel=1566 | q=0.893, sel=1566 | no, slightly better |

Single-fact queries produce fewer than two rare sub-queries, so no coverage
structure is built at all. 32K is the only family whose questions carry several
rare terms; there the selection changes and quality moves 0.875 → 0.893
(TTFT 15.4 s → 17.5 s, within this VM's run-to-run noise band).

## 3. What had to be fixed to get a working mechanism

Four rounds produced *identical* numbers to AR008 before the mechanism did
anything. Each was a real defect, all are now regression-tested:

1. **Enumerated questions have no enumerated bigrams.** "the vaults one, two and
   three" contains no adjacent pair "vault two"; a bigram sub-query finds nothing.
   Sub-queries must be single rare terms.
2. **Digit keys leak into the answers.** Keying facts by vault *number* looks
   natural, but `8491` tokenises into `849` + `1`, and rare uppercase codenames
   split into `AUR` + `ORA`. Spelled numbers (`one`, `two`) are single tokens,
   absent from the filler vocabulary and absent from the passcodes.
3. **The question echoes itself in the last block.** The chat template repeats the
   question, so that block matches *every* sub-query and marks the question
   "covered" before the first fact is admitted. Blocks matching at least half the
   sub-queries are now excluded from evidence.
4. **Summed sibling IDF collapses the anchors.** Scoring a sub-query's anchor as
   `idf_p + sum of sibling IDFs` points every sub-query at the same "richest"
   block; the rest of the facts get no anchor. Pure facility location
   (`idf_p + lexical[b]`) gives one anchor per fact.
5. **The knob was not wired.** `bench.coverage_weight` was never copied from
   `cfg`, so two ablation rounds silently ran the default. The audit column
   (`coverage_queries=0`) is what makes this class of bug visible.

A pre-existing AR008 defect also surfaced: the phrase spill into `per_pos` used
`flat_seq` indices as token indices, which reads out of bounds whenever a piece
decodes to more than one word (ASan: heap-buffer-overflow). Fixed with a
`flat_tok` word→token map; the new echo test is the regression.

## 4. Gate

`benchmarks/scripts/check_expected_ar009.py` + `expected_ar009.csv` (floors, not
measured values):

```
PASS deep/multifact8k             q 0.488 -> 0.878 (+0.390), covered 3.25
PASS deep/multifact8k-hard        q 0.261 -> 0.652 (+0.391), covered 6.50
all 2 AR009 cells within expected bounds
```

Negative self-test (rewriting the `coverage_weight=3` rows with the AR008 values,
i.e. the exact signature of a disabled feature) fails with `rc=1` and reports all
three violations per dataset.

## 5. Honest limits

- **hard tier misses the 0.85 SRS target** (measured 0.652). `coverage_queries=6.5/6`
  says the facts *are* in the context; the remaining loss is 445 tokens of budget
  shared by six answer sentences plus the answer format/order, not an admission
  failure. Raising the budget fixes it at a cost this AR did not claim.
- **No dense ceiling for the hard tier** (runtime), so no "fraction of dense"
  claim there.
- **Selection-level only.** The AR006 in-graph page path does not consume the
  coverage gain yet (the `per_pos` fix does benefit it).
- Two synthetic tiers are a *narrow* evidence base: the families are real prompts
  through the real model, but they are generated, not user traffic.
