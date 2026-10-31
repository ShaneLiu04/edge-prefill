# EdgePrefill

Real-GGUF, prefill-stage acceleration for edge-class CPUs/GPUs, built directly
on a pinned [llama.cpp](https://github.com/ggml-org/llama.cpp) commit. No
mocks, no synthetic timings: every number in `benchmarks/results/` comes from
a real model run with the exact command recorded alongside it.

## What it does

Long-prompt prefill dominates time-to-first-token (TTFT) for local LLM
serving. EdgePrefill plans and executes a sparse prefill:

1. **Block selection (AR001)** — lexical/signal block ranking picks which
   prompt blocks enter the KV cache at all (selection-level sparsity).
2. **FFN skip (AR002)** — transformer FFNs are skipped for non-selected
   tokens during prefill (masked, deterministic, reported with quality).
3. **Attention top-k (AR002/AR003)** — attention layers gather only the
   highest-ranked KV rows; per-head adaptive matrices come from an offline
   head-classification probe (`classify-heads`).
4. **Branch-gated sparse prefill (AR004)** — NSA/MoBA-style
   compressed branches (representative-token injection + pooled V-rows),
   per-head query gating, large-chunk scheduling, and head pattern routing
   (MInference-style) / pyramid layer prior (PyramidKV-style).
5. **In-graph query-aware page gating (AR006)** — the gate moves *inside*
   the prefill graph: per ubatch, per layer, K pages are reduced to means,
   scored against the current query (MoBA-style), and the top-k pages plus
   a local window are gathered once. No embedding model, no host-side
   selection, `semantic_score_ms = 0` on the gated path. Page scoring by
   Quest-style exact bounds was implemented, measured and rejected (biased
   toward large-norm keys); see `docs/research.md` §8.

6. **Length-adaptive admission budget (AR007)** — the admission budget is a
   measured function of prompt length, not a constant: `sweep_budget.py`
   produces a budget-vs-quality curve bound to the model's SHA256, and the
   planner reads it when the conf points at it (`budget_curve`), recording
   `budget_source = model|override|default|clamp` per sample. The same change
   adds structural anchors (sink / last block / query-word anchors / strided
   coverage), span growing (adjacent blocks get +0.5, so an answer that
   straddles a block boundary stays contiguous) and incremental MMR against
   near-duplicate text. Measured: quality is **not monotone** in the budget
   (16K: b=0.05 q=1.00 vs b=0.25 q=0.36), which is why the curve is measured
   per length instead of scaled.

7. **Phrase-aware admission + combination portfolio (AR008)** — the selector
   now admits *positional phrases* (adjacent query word pairs such as
   `vault 5`, `chapter 7`, `case 42`) as first-class terms, keeps
   single-character discriminators that the old filter dropped, and matches
   phrases across block boundaries. This fixed the multi-needle family
   (quality 0.33 → 1.00 at 5% budget) where a lone digit was the only thing
   separating sibling facts. On top of that, a measured `portfolio` table
   routes each prompt to the execution combination whose measurement reaches
   the quality floor (`selection` / `gate-p16` / `gate-p24-local32` / `dense`),
   and `check_portfolio.py` refuses any combination that is Pareto-worse than
   the best fixed one.

8. **Multi-fact coverage admission (AR009)** - a question can ask for several
   facts at once ("list the passcodes for the vaults one, three and five").
   AR008's admission is single-fact optimal: it spends the whole budget on the
   single strongest match, so on a new multi-fact family (6 or 12 planted
   facts, 3 or 6 asked per row) it recovered only 0.49 / 0.26 of the facts.
   The selector now extracts the *rare* query terms as sub-queries, gives each
   one an anchor, and adds a submodular facility-location gain
   (`Σ max(0, idf_p - covered_p)`) to the fill score, so every requested fact
   has to be paid for. Quality 0.488 -> **0.878** and 0.261 -> **0.652** at the
   same budget; single-fact families keep a bit-for-bit identical selection.
   `--coverage-weight 0` restores AR008 exactly.

9. **Evidence-first admission (AR010)** - a budget audit at 32K showed 62% of
   the selected tokens going to blocks with **zero** lexical/phrase overlap,
   because the position/diversity priors could create admission on their own.
   The root cause was a `minmax()` aliasing bug (`*lo` was an iterator into the
   vector being overwritten, so `diversity` was normalised to [0, 3.2] instead
   of [0, 1]). With the normalisation fixed, admission is split into
   `evidence` (lexical + phrase + draft) and `prior` (position + diversity),
   and priors only order blocks that already carry evidence
   (`fused = e + sat(e)*p`). The fill also gained an explicit stop reason, so
   the budget can be content-sized (`fill_priors=0`) instead of a quota:
   32K reaches **q=1.000 at 525 tokens / 2.26 s (198x vs dense)** where AR009
   needed the in-graph gate (5443 ms) for the same quality, and 4K/8K/16K cost
   1.4-3.0x fewer selected tokens at unchanged quality.

Sparse paths are lossy by construction. Every benchmark report pairs latency
with a quality metric (needle accuracy, QA overlap, or PPL retention), and
`budget=1.0` / all switches off remain numerically equivalent to dense.

## Quick start (Linux/WSL2)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

llama.cpp is fetched automatically at the pinned commit
(`3bb63cbcbfb62056c84fe9e521b0f3d756d29c32`) and patched with
`patches/*.patch`. To use a local tree:
`-DEDGE_PREFILL_LLAMA_DIR=/path/to/llama.cpp`.

Real-model tests are enabled by pointing at a GGUF file:

```bash
EDGE_PREFILL_TEST_MODEL=/path/to/model.gguf ./build/edge_prefill_tests
# without the variable, real-model cases SKIP; nothing is faked
```

## Running

```bash
# plan only (no model)
./build/edge-prefill plan --tokens 16384 --budget 0.25 --json

# single prompt, real model
./build/edge-prefill run --model M --prompt-file F --budget 0.25 \
    --attn-top-k 24 --head-adaptive on --head-profile P --head-top-k 96 \
    --max-new-tokens 32 --threads 16

# offline head classification (produces the profile for --head-profile)
./build/edge-prefill classify-heads --model M --output profile.json

# benchmark suite (see benchmarks/configs/)
./build/edge-prefill bench --config benchmarks/configs/exp-balanced.conf --model M
```

## Configuration

| Flag | Meaning | Default |
| --- | --- | --- |
| `--mode latency\|balanced\|quality\|reuse\|dense` | planning profile | balanced |
| `--budget <0..1>` | fraction of prompt tokens allowed into the KV cache | 0.5 |
| `--attn-top-k N` | KV blocks whose rows attention gathers (0 = off) | 0 |
| `--attn-top-k-layers 4,8,16` | per-layer top-k override | — |
| `--head-adaptive on\|off` | per-head gather matrix from a profile | off |
| `--head-profile P` | profile produced by `classify-heads` | — |
| `--head-top-k N` | rows each non-retrieval head gathers | 0 |
| `--head-retrieval-threshold F` | probe score above which a head stays dense | 0.10 |
| `--head-pattern-routing on\|off` | MInference-style per-class budgets (AR004) | off |
| `--pyramid-prior on\|off` | PyramidKV-style layer prior without a profile (AR004) | off |
| `--pyramid-k-min/--pyramid-k-max N` | pyramid budget range | 48/128 |
| `--kv-type f16\|q8_0\|q4_0` | KV cache precision | f16 |
| `--skip-ffn <spec>` | FFN layer skip mask for prefill | — |
| `--chunk-size N` | 0 = auto-align to n_ubatch (AR004) | 0 |
| `--budget-curve P` | length -> admission-budget table (AR007) | — |
| `--anchor-ratio F` | anchor budget as a fraction of the block target (AR007) | 0.35 |
| `--anchor-stride-scale F` | strided-coverage anchor multiplier (AR007) | 1.0 |
| `--mmr-lambda F` | MMR penalty weight on near-duplicate blocks (AR007) | 0.5 |
| `--mmr-threshold F` | Jaccard similarity above which MMR applies (AR007) | 0.6 |
| `--phrase-weight F` | positional phrase admission weight, 0 = AR007 behavior (AR008) | 0.5 |
| `--coverage-weight F` | multi-fact coverage weight, 0 = AR008 selection exactly (AR009) | 3.0 |
| `--prior-gate F` | 1 = priors cannot create admission, 0 = legacy additive fusion (AR010) | 1.0 |
| `--fill-priors F` | 0 = content-sized context (stop when evidence ends), 1 = fill the budget (AR010) | 1.0 |
| `--portfolio P` | measured execution-combination table (AR008) | — |
| `--structure-budget N` | sink+recent tokens reserved before evidence (AR011) | 0 (off) |
| `--anchor-span-radius N` | widen admitted anchors by N blocks on each side (AR011) | 0 |
| `--coverage-reserve F` | fraction of the target reserved for uncovered facts (AR011) | 0 |
| `--span-admission on\|off` | single-fact evidence trimmed to whole sentences within its block (AR011) | off |
| `--span-min-tokens N` | sentences shorter than this are not used to narrow a span (AR011) | 24 |
| `--sentence-completion on\|off` | evidence pulls in its whole sentence; multi-fact only (AR012) | off |
| `--sentence-trim on\|off` | trim remaining filler after completion (measured lossy, not routed) (AR012) | off |
| `--span-max-tokens N` | cap on one completed span (AR012) | 128 |
| `--dump-selected-text` | write the admitted text per prompt (AR012) | off |
| `--position-compaction on\|off` | collapse the gaps between admitted blocks so the KV span tracks the admitted tokens (AR013) | off |
| `--n-ctx N` | explicit context size; too small is an error, not a truncation (AR013) | grow to the span |
| `--session-reuse on\|off` | reuse the KV prefix across turns of the same conversation (AR014) | off |
| `--session-ctx-max N` | cap on the one-shot allocation for a conversation; 0 = no cap (AR014) | 4096 |
| `--snapshot-dir P` | content-addressed KV snapshot directory; empty disables (AR015) | — |
| `--snapshot-use 0\|1` | 1 = restore a matching snapshot instead of prefilling (AR015) | 1 |
| `--qg-bound on\|off` | in-graph query-aware page gating (AR006) | off |
| `--qg-sel-pages N` | pages kept per ubatch by the gate (quality knob) | 16 |
| `--qg-local-rows N` | local window rows around the ubatch (0 = ubatch span) | 0 |
| `--qg-pool-pages N` | pool pages folded into the gated branch | 0 |
| `--threads N` | CPU threads; now honoured by `bench` too (AR016) | physical cores |
| `--n-gpu-layers N` | GPU offload (99 = full); verified after load (AR016) | 0 |
| `--ngl N` | alias of `--n-gpu-layers` (AR016) | 0 |
| `budget_target_ms F` | shrink the budget until the predicted TTFT fits (AR016) | 0 (off) |
| `cost_a_ms` / `cost_b_ms_per_token` | fitted cost model, from `edge-prefill probe` (AR016) | unset |
| `allow_cpu_fallback 0\|1` | permit a requested-GPU run to continue on CPU (AR016) | 0 |
| `decode_batch_size K` | decode K sequences per forward pass (AR017); 1 = serial | 1 |
| `decode_batch_ctx N` | total KV cells of the batch context (AR017); 0 = K*1536 | 0 |
| `decode_batch_max_span N` | longest admitted span allowed to join a batch (AR017) | 1024 |

## Expected results

Real runs only: Qwen2.5-1.5B-Instruct Q4_K_M, 16 threads, Xeon Gold 6234,
one interleaved session per dataset (dense measured right before its sparse
siblings). Raw CSV + env JSON + the exact config live in
`benchmarks/results/ar006-campaign/`; the full write-up with the honest
acceptance status is `benchmarks/results/ar006-campaign/REPORT.md`.

| Setup | Prompt | TTFT | vs dense | Quality (needle) |
| --- | ---: | ---: | ---: | --- |
| dense | 8K | 118.9 s | 1.0x | 1.00 |
| in-graph gate p16 (AR006) | 8K | 11.6 s | **9.4x** | 1.00 |
| dense | 16K | 161.8 s | 1.0x | 1.00 |
| in-graph gate p16 (AR006) | 16K | 23.7 s | **6.8x** | 1.00 |
| AR005 qg-2d + embedding | 16K | 25.1 s | 6.4x | 1.00 |
| in-graph gate p24 + local32 | 32K | 67.9 s | **5.7x** | 1.00 |
| in-graph gate p16 | multi-needle | 17.1 s | 3.8x | 0.83 (scalar: 2.8x / 0.67) |

### AR007: budget curve vs fixed budget (same session, one dense reference per dataset)

| Setup | Prompt | TTFT | vs dense | Quality (needle) |
| --- | ---: | ---: | ---: | --- |
| dense | 8K | 48.8 s | 1.0x | 1.00 |
| curve, b=0.05 | 8K | **1.7 s** | **29.1x** | 1.00 |
| AR006 gate p16, b=0.25 | 8K | 9.1 s | 5.4x | 0.88 |
| dense | 16K | 199.5 s | 1.0x | 1.00 |
| curve, b=0.05 | 16K | **5.2 s** | **38.5x** | 1.00 |
| AR006 fixed b=0.25 | 16K | 19.9 s | 10.0x | 0.36 |
| dense | 32K | 384.1 s | 1.0x | 1.00 |
| curve, b=0.05 | 32K | **7.0 s** | **55.2x** | 0.89 |
| in-graph gate p24 + local32 (AR007 page score) | 32K | 31.4 s | 12.2x | 1.00 |
| curve, b=0.05 | multi-needle | 1.8 s | 30.5x | **0.33** |
| in-graph gate p16 + local32 | multi-needle | 9.7 s | 5.5x | 0.83 |

Raw CSV, env JSON and the conf of every combination cell are in
`benchmarks/results/ar008-portfolio/` (see its `REPORT.md` for the honest
write-up, including the discriminator bug the matrix exposed); AR007 assets are
still in
`benchmarks/results/ar007-campaign/`; the honest write-up (including the
non-monotone budget and the multi-needle weakness) is
`benchmarks/results/ar007-campaign/REPORT.md`, and the curve itself is
`benchmarks/results/ar007-sweep/{budget_curve.txt,curve.json,summary.md}`.
`budget_curve` is **opt-in** (no conf, no curve): on multi-needle documents the
selection-level path loses facts that the in-graph gate keeps, so use
`qg_bound=on` there. `benchmarks/scripts/check_expected_ar007.py` re-checks the
quality/audit columns on another machine without depending on absolute TTFT.

### AR008: phrase admission and combination routing

| Setup | Prompt | TTFT | Quality |
| --- | ---: | ---: | --- |
| selection, `phrase_weight=0` (AR007) | multi-needle, b=0.05 | 5.7 s | **0.33** |
| selection, `phrase_weight=0.5` | multi-needle, b=0.05 | 6.2 s | **1.00** |
| selection (routed) | 8K | 5.7 s (24.9x vs dense) | 1.00 |
| selection (routed) | 16K | 15.9 s (19.1x) | 1.00 |
| selection | 32K | 23.1 s (36.8x) | **0.88** |
| gate-p24-local32 (routed at 32K) | 32K | 66.8 s (12.8x) | **1.00** |
| selection (routed) | multi-needle | 8.2 s (11.2x) | 1.00 |

Routing rule: fastest combination whose mean *and* min quality reach 0.95,
else the highest quality - so everything stays on selection except 32K, where
the in-graph gate buys the floor back for 2.9x TTFT. `check_portfolio.py`
verifies that the routed combination is Pareto-no-worse than the best fixed
one (and fails on an injected slower arm).

The multi-needle row is the honest before/after: the same budget, the same
machine, only the admission signal changed. Full matrix (5 datasets × 4
combinations, dense reference per dataset) and the routed table live in
`benchmarks/results/ar008-portfolio/`; `./repro.sh ar008` regenerates them and
runs the Pareto gate.

### AR009: multi-fact coverage admission

AR008's multi-needle set asks one fact per row, so a top-1 selector looks
perfect. The new `gen_multifact.py` family plants 6 or 12 facts and asks 3 or 6
of them per row, scoring partial credit:

| Setup | Prompt | TTFT | Quality | facts covered |
| --- | ---: | ---: | --- | ---: |
| selection, `coverage_weight=0` (AR008) | 6 facts, ask 3, b=0.05 | 2.77 s | **0.488** | 0/3 |
| selection, `coverage_weight=3` | 6 facts, ask 3, b=0.05 | 2.93 s | **0.878** | 3.25/3 |
| selection, `coverage_weight=0` (AR008) | 12 facts, ask 6, b=0.05 | 2.64 s | **0.261** | 0/6 |
| selection, `coverage_weight=3` | 12 facts, ask 6, b=0.05 | 3.22 s | **0.652** | 6.50/6 |
| selection, `coverage_weight=3` | 4K / 16K / multi-needle | unchanged | 1.000 | n/a (single fact) |
| dense (2-context subset) | 6 facts, ask 3 | 8.5K full prompt | 0.975 | reference |

`facts covered` is the `coverage_queries` audit column. On single-fact families
fewer than two rare terms exist, so no coverage structure is built and the
selection is bit-for-bit the AR008 one (4K/16K/multi-needle verified). The hard
tier is honest about its ceiling: 0.652 is below the 0.85 target because six
answer sentences do not fit comfortably in 445 tokens, not because facts are
missing from the context.

`./repro.sh ar009` regenerates both datasets, runs the ablation and gates it
with `check_expected_ar009.py` (floors, not recorded values; a disabled feature
fails the gate). Full report: `benchmarks/results/ar009-multifact/REPORT.md`.

### AR010: evidence-first admission, and why 32K no longer needs the gate

A budget audit at 32K found 62% of the selected tokens sitting in blocks with
zero lexical/phrase overlap. Two defects caused it: `minmax()` normalised with
an iterator into the vector it was overwriting (`diversity` reached 3.2 instead
of 1.0, so priors could *create* admission), and "fill the budget" was a side
effect of those broken priors rather than a policy.

| Setup | Prompt | TTFT | Quality | selected tokens |
| --- | ---: | ---: | --- | ---: |
| selection, `fill_priors=1` (AR009 quota) | 32K | 7276 ms | **0.611** | 1565 |
| gate-p24-local32 (AR008 routing for 32K) | 32K | 5443 ms | 1.000 | 1165 |
| **selection, `fill_priors=0` (AR010)** | 32K | **2260 ms** | **1.000** | **525** |
| selection, quota vs content-sized | 16K | 3675 → **1251 ms** | 1.000 | 792 → **280** |
| selection, quota vs content-sized | 8K | 1698 → **1039 ms** | 1.000 | 388 → **228** |
| selection, quota vs content-sized | 4K | 1491 → **1050 ms** | 1.000 | 322 → **226** |
| selection | multi-needle | 2675 ms | 1.000 | 321 |

`./repro.sh ar010` re-measures the five-arm matrix, re-derives the table and
runs the Pareto gate (plus an injected degraded arm that must fail). GPU
consistency (CUDA, RTX 5000 16 GB) is documented in
`benchmarks/results/ar010-evidence-first/REPORT.md` section 4. The gated path
runs without an embedding model (`semantic_score_ms = 0`).

### AR011: structure-bounded admission

Two audits set up this AR. First, TTFT is now proportional to the admitted
token count (`selected / 155 token·s⁻¹` on this CPU), so the remaining lever is
the *count*. Second, the count was not evidence: a 4K prompt at b=0.08 admitted
310 tokens of which 9 of 10 blocks were mandatory anchors (sink, recent, the
best-lexical ±2 window and phrase pins), i.e. `anchor_ratio` capped a fraction
of a budget that the untrimmable anchors had already spent.

| Setup | Prompt | q_mean | q_min | selected | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR010 | 4K | 1.000 | 1.000 | 226 | 1046 ms |
| **AR011** | 4K | 1.000 | 1.000 | **146** | **755 ms** |
| AR010 | 8K | 1.000 | 1.000 | 228 | 1091 ms |
| **AR011** | 8K | 1.000 | 1.000 | **188** | **1028 ms** |
| AR010 | 16K / 32K | 1.000 | 1.000 | 280 / 525 | 1444 / 2371 ms |
| **AR011** | 16K / 32K | 1.000 | 1.000 | 280 / 525 | 1387 / 2329 ms |
| AR010 | multi-needle | 1.000 | 1.000 | 321 | 1790 ms |
| **AR011** | multi-needle | 1.000 | 1.000 | **247** | **1165 ms** |
| AR010 | multi-fact hard | 0.667 | **0.087** | 445 | 1997 ms |
| **AR011** | multi-fact hard | **0.709** | **0.571** | 445 | 1952 ms |

Structure (sink + recent) now gets an absolute token allowance and
`anchor_ratio` only governs the remaining evidence budget - and only when the
anchors actually dominate it (at 16K/32K they do not, so nothing changes).
Span admission trims evidence blocks to the sentence around the evidence peak,
decoupling the admission unit from the 32-token attention block; it is
automatically skipped when several facts are asked, because a multi-fact answer
needs one whole sentence per fact (measured: trimming there costs
0.836 → 0.800 and `q_min` 0.174 → 0.000).

`./repro.sh ar011` measures 7 datasets × 3 arms and gates the routed table
against AR010 on quality, tokens *and* TTFT, with an injected regression that
must fail. Two negative results are recorded in the AR011 report: the embedding
(AR005) signal buys exactly zero quality on these families while adding 1-2.2 s
of encoding scan, and the per-fact reservation fill has no reliable gain
(+0.019 at b=0.03, -0.007 at b=0.02), so it ships off by default.

### AR012: sentence-complete admission

The weakest cells after AR011 were the multi-fact tiers (0.836 / 0.709). A budget
sweep disproved the obvious explanation - more budget is *worse* (b=0.16: 0.771),
because the extra tokens are filler. Dumping the admitted text (new
`--dump-selected-text` and `dump_selection_dir`) showed the real cause: a fact
row is split by the 32-token block boundary and all-or-nothing block admission
ships half an answer (`... for vault six is TEAL-LYNX-8` | `853`).

| Setup | Prompt | q_mean | q_min | selected | facts fully in context |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR011 | multi-fact easy | 0.836 | 0.429 | 431 | 0.875 |
| **AR012** | multi-fact easy | **0.887** | 0.429 | 509 | **1.000** |
| AR011 | multi-fact hard | 0.709 | 0.571 | 445 | 0.760 |
| **AR012** | multi-fact hard | **0.739** | **0.621** | 513 | **0.990** |
| AR011 = AR012 | 4K / 8K / 16K / 32K / multi-needle | 1.000 | 1.000 | unchanged | n/a |

Evidence tokens pull in their whole sentence; a sentence that spills into an
unselected block pulls that block in, so the answer tail is admitted even when
its own block had no query hit. Completion engages only when several facts are
asked (on single-fact prompts it only added 15% tokens at q=1.000), and trimming
the remaining filler is available but *not routed*: it measured 1.000 -> 0.827 on
4K/32K because it removes sentences the model still needs.

`facts fully in context` is the deterministic audit added in this AR
(`check_facts_admission.py`): it counts how many asked answers appear verbatim in
the admitted context, so admission quality can be measured without model-score
noise. `./repro.sh ar012` runs the 4-arm matrix with per-case dumps, derives the
routing table, gates it against AR011 on a "no free cost" rule (quality must not
drop; tokens may only grow when quality does) and fails on an injected
degradation.

Which ARs are in the shipped combination, which stayed optional and which were
rejected as negative results is summarised in `docs/portfolio.md`.

### AR013: compact context (KV sized by what is admitted, not by the document)

Admission had driven the token count down 27x (4K: 146 tokens of 3962), but the
context still reserved the model's full training length: a 4K prompt allocated
**896 MiB** of KV cache for the 3.5 MB it actually wrote, and a document longer
than `n_ctx_train` could not run at all. AR013 makes the KV span follow the
admitted tokens.

| Dataset | AR012 q | AR013 q | KV cache AR012 | KV cache AR013 | TTFT (min of 5) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 1.000 | 1.000 | 112 MiB | **7 MiB** | 3311 -> **2163 ms** |
| 8K | 1.000 | 1.000 | 217 MiB | **7 MiB** | 1131 -> 1100 ms |
| 16K | 1.000 | 1.000 | 430 MiB | **14 MiB** | 1641 -> 1789 ms |
| 32K | 1.000 | 1.000 | 854 MiB | **21 MiB** | 2980 -> 3073 ms |
| multi-needle | 1.000 | 1.000 | 217 MiB | **14 MiB** | 1551 -> 1518 ms |
| multi-fact | 0.887 | **0.975** | 238 MiB | **21 MiB** | 3131 -> 3156 ms |
| multi-fact hard | 0.739 | **0.746** | 244 MiB | **21 MiB** | 2482 -> 2603 ms |
| **64K document** | cannot run | **1.000** | — | 35 MiB | 6436 ms |
| all, vs the pre-AR013 bench | | | **896 MiB** | **7-21 MiB** | noise-neutral (aggregate 0.96x) |

Two mechanisms, both opt-in so `budget=1.0` stays dense-equivalent:

- **Context sized by the span** (`LlamaBackend::ensure_context`): the bench
  stops reserving `n_ctx_train`; each case grows the context to the highest
  position it will write. An explicit `--n-ctx` that is too small is a hard
  error, never a silent truncation.
- **Block position compaction** (`--position-compaction`): after admission the
  gaps between kept blocks are collapsed while every *intra-block* offset is
  kept verbatim, so a fact and its value keep the exact distance the model was
  trained on and only the dropped filler stops occupying cells. The gather
  indices stay in original prompt coordinates and are mapped to decode rows.

The multi-fact gain is not an admission effect: the deterministic audit reports
the same `facts_in_context` (1.000 easy / 0.990 hard) for both arms, yet the
model answers better, because the evidence blocks now sit next to each other
instead of thousands of filler positions apart.

`./repro.sh ar013` runs the 8-dataset matrix, gates it (quality must not drop,
TTFT must not regress beyond measurement noise, KV must shrink under per-dataset
ceilings, and the 64K document must run where AR012 provably cannot), and fails
on two injected degradations. The 64K contexts themselves are not committed
(1.6 MB of synthetic text that the push host rejects): `repro.sh ar013`
regenerates them with the pinned seed, and `benchmarks/data/explen/len64k/`
keeps the index plus the per-file SHA256 in `manifest.json`.

Absolute TTFT depends on the host (this session ran up to 3x slower than the
AR004 session on the same CPU); compare ratios measured in one session, and
use `benchmarks/scripts/check_expected_ar006.py` for the machine-independent
gate. AR006 does **not** reach 10x at 32K on CPU with the current page-mean
implementation — the reason and the next increments are in the REPORT.

### AR014: session KV reuse (a conversation pays once for its context)

AR013 made one request cheap, but every request rebuilt the context from
position 0. In a real conversation (same document, follow-up questions) turns
2..N overlap turn 1 almost entirely and still paid a full prefill.

AR014 gives each conversation an **append-only log**: a turn appends the
evidence it newly admits plus its question, then decodes only that tail; the
prefix KV stays where it is. Both arms therefore feed the model the same tokens
at the same positions — `session_log_hash` proves it — and only the batch shape
differs.

| Turn | Recompute q | Reuse q | TTFT reuse/recompute | reused | appended |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.791 | 0.791 | **0.91x** (4379 -> 3968 ms) | 0 | 954 |
| 2 | 0.684 | 0.684 | **0.27x** (5660 -> 1500 ms) | 970 | 292 |
| 3 | 0.727 | 0.727 | **0.26x** (7916 -> 2057 ms) | 1239 | 388 |
| 4 | 0.561 | 0.636 | **0.20x** (9299 -> 1827 ms) | 1634 | 314 |
| 5 | 0.625 | 0.625 | **0.13x** (10554 -> 1358 ms) | 1948 | 228 |
| all 40 turns | 0.678 | **0.693** | 0.26x of the per-turn median | 1153 | 425 |

`deep/multiturn` is 8 sessions x 5 turns over the same 3.5 MB document; the
session log grows 1042 -> 2268 tokens while each turn only decodes 164-388.
Single-turn datasets are untouched (no `session` column): `len4k` 1.000 -> 1.000
and `multifact8k` 0.975 -> 0.975, both bit-identical to AR013.

Three mechanisms, all opt-in:

- **Append-only session log** (`SessionStore` + `--session-reuse`): the
  conversation owns its token log and the set of original document indices it
  already contains, so a turn's delta is the newly admitted evidence in
  document order. A turn that points at a different document is an error, never
  a silent cross-document reuse.
- **Incremental decode** (`Options.append`): `llama_decode` continues from
  `next_pos`; the sequence is not reset and the prefix is not recomputed.
- **One-shot context sizing** (`--session-ctx-max`): re-creating the context
  wipes the KV, so a conversation is sized once from its turn count and first
  delta, capped at 4096 cells by default. A turn that outgrows the cap falls
  back to rebuilding the log for that turn — correct, just slower.

One honest caveat: 39 of 40 turns are bit-identical between the arms, and one
(`mt-03-t4`) flips in the reuse arm's favour (0.200 -> 0.800) although its log
hash is identical. The cause is batch shape, not context drift: a whole-sequence
chunked prefill and an incremental decode reach the same K/V through different
ggml tilings, and greedy decoding turns the last-bit difference into one token.
The gate therefore checks the provable invariants (identical logs) plus a bound
(mean must not drop, at most 5% of turns may flip, no turn may lose a quarter of
its fields) instead of pretending fp is associative.

`./repro.sh ar014` regenerates the corpus, runs both arms plus the single-turn
spot checks, gates them and fails on two injected degradations
(`no-reuse`, `quality-drift`).

### AR015: persistent KV snapshots (a repeat request pays disk, not compute)

AR014 keeps KV inside one process. Edge processes die: the user re-asks, the app
is killed and relaunched, an agent retries. AR015 makes the *admitted* log a
content-addressed object on disk, so the next process restores the sequence
instead of recomputing it.

The key is everything that decides what the model sees: the model fingerprint,
the whole admission policy, the KV precision, the context size, and the admitted
tokens plus their positions. The snapshot is the sequence state *plus the logits
of the last prompt token*, which is what makes the warm path both compute-free
and bit-identical: the first generated token comes from stored bytes, and the
rest of the generation continues from the restored KV.

| Dataset | cold TTFT (min) | warm TTFT (min) | ratio | snapshot | quality |
| --- | ---: | ---: | ---: | ---: | --- |
| 4K | 3292 ms | **117 ms** | 0.036x | 19.0 MB | identical |
| 8K | 6141 ms | **190 ms** | 0.031x | 32.6 MB | identical |
| 16K | 5979 ms | **195 ms** | 0.033x | 33.0 MB | identical |
| 32K | 7422 ms | **203 ms** | 0.027x | 34.0 MB | identical |
| multi-needle | 6174 ms | **188 ms** | 0.030x | 32.6 MB | identical |
| multi-fact | 2356 ms | **93 ms** | 0.040x | 15.2 MB | identical |
| multi-fact hard | 2073 ms | **95 ms** | 0.046x | 15.3 MB | identical |
| 64K document | 5074 ms | **191 ms** | 0.038x | 34.0 MB | identical |

Note (AR016 erratum): the GPU numbers originally quoted here were measured on
CPU -- the bench ignored the CLI `--n-gpu-layers` flag, so llama.cpp offloaded
0/29 layers. Restore itself is disk + H2D, which is device independent, so the
warm numbers stand; the cold ones are corrected in
`benchmarks/results/ar015-kv-snapshot-gpu/ERRATUM.md`. AR016 adds placement
verification (`device`, `gpu_layers_offloaded`, `device_state` columns) so this
cannot happen silently again.

Snapshot size is the compact KV AR013 made possible (`kv_used + 0.6 MB` of
logits): before AR013 the same sequence would have needed 896 MiB.

Composed with AR014, a conversation replayed in a *new* process restores every
turn (40/40 turns, identical answers and identical `session_log_hash`) instead
of re-prefilling the accumulated log.

Safety comes from the key, not from trust: a stale or corrupt snapshot is a miss
(manifest mismatch, payload hash mismatch, token mismatch), never a silent
reuse, and the cold path records its own snapshot right after the prefill.

`./repro.sh ar015` runs the cold/warm matrix, the multi-turn resume trio and the
gate, and fails on injected `no-hit` and `quality-drift` degradations.

### AR016: the execution plan is verified, and the budget can target a latency

Everything above is a promise about *time on a specific machine*. AR016 makes
that promise checkable, because it turned out not to be: the bench ignored CLI
`--threads` and `--n-gpu-layers`, so every run silently used
`hardware_concurrency()` threads and zero GPU layers (see the AR015 erratum).
Two consequences, both measurable:

| | before | after |
| --- | ---: | ---: |
| threads used by `bench` | `hardware_concurrency()` (16 here) | from the plan (12 here) |
| GPU layers used by `bench` | 0, even with `--n-gpu-layers 99` | as requested, and verified |
| prefill slope `b` (4K doc) | 6.2 ms/token (16 threads) | **3.46 ms/token** (12 threads) |
| TTFT, 8K / 32K / 64K | 6117 / 6594 / 6407 ms | **4193 / 4352 / 4376 ms** |

`probe` measures the real device and writes a plan overlay:

```bash
./build/edge-prefill probe --model M --prompt-file CTX --threads 12 --out /tmp/plan
# probe: a=-3.12 ms b=3.46013 ms/token r2=0.999 n=5 device=CPU
cat /tmp/plan/plan.conf          # threads/cost model, appendable to any conf
cat /tmp/plan/probe_device.txt   # request/offload/state, for GPU runs
```

Placement is verified after load: if the plan asks for GPU layers and llama.cpp
reports `offloaded 0/N`, the run fails with `Unavailable` instead of quietly
measuring the CPU (opt into the old behaviour with `allow_cpu_fallback = 1`, which
stamps `device_state=cpu(fallback)`).

The cost model is then used for a latency target:

```bash
# budget_target_ms: shrink the admission budget until the *predicted* TTFT fits
./build-cpu/edge-prefill bench --config conf-with-cost-model --sla 2500
```

Same quality, half the wait, on every dataset:

| Dataset | plan TTFT | SLA (2500 ms target) | tokens | f1 |
| --- | ---: | ---: | ---: | --- |
| 4K | 2325 ms | 2204 ms | 642 -> 610 | unchanged |
| 8K | 4193 ms | 2277 ms | 1117 -> 631 | unchanged |
| 16K | 4222 ms | 2293 ms | 1128 -> 633 | unchanged |
| 32K | 4352 ms | 2243 ms | 1166 -> 622 | unchanged |
| 64K | 4376 ms | 2259 ms | 1163 -> 628 | unchanged |
| multi-fact | 1880 ms | 1869 ms | 521 | unchanged (~<2500, no shrink) |

`./repro.sh ar016` runs the probe, the three-arm matrix, the gate, and rejects a
cost-model skew injection.

### AR017: one decode step for many sequences

Every case above still paid its generation tokens one at a time, one case at a
time. AR017 batches the *decode* phase: N cases that are already prefilled each
contribute one token, and llama.cpp decodes them in a single forward pass
(`decode_batch`). Prefill is untouched — the sparse gather keeps a different
position map and mask per sequence, so it cannot share a graph.

| | serial (K=1) | K=2 | K=4 | K=8 |
| --- | ---: | ---: | ---: | ---: |
| decode wall (interleaved A/B) | 1.00x | **2.01x** | 2.74x | 2.51x |
| single-case end-to-end | 1.00x | 0.89x | 0.71x | 0.39x |
| f1 (36 cases) | identical | identical | identical | identical |

K=2 is the recommended point: decode throughput doubles while a single case
still finishes only 11% later. Larger K trades per-case latency for aggregate
throughput, and K=8 loses to K=4 because the per-sequence KV share (`decode_batch_ctx`
split K ways) starts pushing cases onto the serial fallback.

```bash
# batch the decode of up to 4 cases whose admitted span fits the KV share
printf 'decode_batch_size = 4\ndecode_batch_ctx = 12288\n' >> conf
./build/edge-prefill bench --config conf
```

```ini
decode_batch_size = 2       # 1 = serial; K = sequences decoded per step
decode_batch_ctx = 6144     # total KV cells of the batch context (0 = K*1536)
decode_batch_max_span = 1024 # longest admitted span that may join a batch
```

The audit columns say what actually happened rather than what was requested:
`batch_size`, `batch_steps`, `batch_wall_ms`, `seq_id`, `decode_ms_per_token`,
`generated_hash`. A case that does not fit the KV share, or a session case
(AR014), runs serially and reports `batch_size=1` — never silently.

Constraints are real and recorded in the report: the batch context must be sized
*before* the first sequence is prefilled (`ensure_context` growth wipes the KV of
every prefilled sequence), snapshots are disabled in batch mode (AR015 snapshots
assume seq 0), and an exact `generated_hash` match is not guaranteed because the
batch shape changes GEMM reduction order — quality (f1) is identical on all 36
cases, 2 of 72 batched arms differ by one token.

`./repro.sh ar017` runs the K sweep matrix, the gate, and rejects injected
`shared-seq` and `quality-drift` degradations.

## Repository layout

```
src/            planner, selector, backend, benchmark, CLI
include/        public headers (Status/ErrorCode, no exceptions across API)
patches/        llama.cpp patches (attn top-k, FFN skip)
tests/          unit + real-model integration tests (SKIP without model)
benchmarks/     configs, datasets, raw CSV reports, analysis scripts
scripts/        experiment orchestration; diagnostics/ = archived probes
specs/          SRS / design / task tracking per change request
docs/           architecture.md, experiments.md, portfolio.md (AR001-AR017 inventory)
```

## Reproducibility rules

- Pinned llama.cpp commit: `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32`
- Benchmark model: Qwen2.5-1.5B-Instruct Q4_K_M GGUF,
  SHA256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
  (`scripts/download_model.sh` verifies the hash and refuses mismatches).
- Draft model (optional, AR005): Qwen2.5-0.5B-Instruct Q4_K_M GGUF,
  SHA256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
  (`scripts/download_model.sh --draft`). Fallback: `--draft-model` may point
  at the main model file itself — the vocab matches, so `LlamaDraftScorer`
  accepts it; only the draft forward gets slower. The explicit draft path
  failing to load is a hard error, never a silent downgrade.
- Embedding model (AR005): bge-small-en-v1.5 Q8_0 GGUF,
  SHA256 `f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804`
  (`scripts/download_model.sh --embed`). Activates with `--embedding-model`;
  a failed load is a hard error, and every request records
  `score_source` / `semantic_score_ms` / `qg_source` audit columns so
  speedups stay net of the scoring overhead.
- No performance number without the raw CSV and the exact command.
- Prefill-only FFN skip and attention sparsity are lossy; every latency claim
  carries its quality metric in the same report.
- `budget=1.0` with all sparsity switches off is numerically equivalent to
  dense prefill (enforced by tests).
