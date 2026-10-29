# Architecture

EdgePrefill is a planning layer plus a patched llama.cpp execution layer.
All public APIs return `Status` (no exceptions across boundaries); errors
carry a stable `ErrorCode`.

## Layered view

```
CLI / benchmark (src/main.cpp, src/benchmark.cpp)
        |
Planner (src/config.cpp, src/chunk_scheduler.cpp, src/planner*)
        |  Plan: budget, chunk_size, skip_ffn_mask, dense_fallback, mode
        v
Selection (src/selector.cpp)
        |  Selection: token_indices, block_indices, mandatory_blocks,
        |             signals, total_tokens, dense
        v
Sparsity controllers (src/attention_sparsity.cpp)
        |  gather indices / ranked pool / head matrices
        v
SparsePrefiller (src/sparse_prefill.cpp, src/backend.cpp)
        |  Options -> llama.cpp patched graph ops
        v
llama.cpp @ 3bb63cb (patches/llama-prefill-attn-top-k.patch,
                     patches/llama-prefill-ffn-skip.patch)
```

## Two sparsity levels

1. **Selection level** — blocks whose tokens never enter the KV cache. The
   selector keeps mandatory anchors (sink / recent / best lexical match);
   planned injection (AR004) adds block representatives so dropped context
   stays addressable.
2. **Attention level** — rows that exist in the cache but are not gathered
   by attention. The ranked pool is `mandatory-first, then fused-signal
   descending`; per-head matrices (uniform, pattern-routed, or pyramid)
   assign each head a row budget; dense heads use -1.

The pool ordering is what makes small per-head budgets safe: a landmark head
with k=32 still gathers the sink + recent anchors because they head the pool.

## Patch interface (llama.cpp side)

`patches/llama-prefill-attn-top-k.patch` adds, among others:

- `llama_set_prefill_attn_top_k` — scalar top-k gather
- `llama_set_prefill_attn_ranked_rows` — pool + per-head matrix gather
  (matrix values: -1 dense, k>0 rows from the pool prefix)
- `llama_set_prefill_query_gated` — per-query-chunk row masks (AR005)
- `llama_set_prefill_qg_bound` / `llama_get_prefill_qg_bound` — in-graph
  query-aware page gating (AR006): per ubatch, per layer, the graph builds
  K page means, scores `q · mean(K_page)`, keeps local ∪ top-k pages and
  gathers once; no host-side selection and no embedding model
- `llama_set_prefill_skip_ffn_mask` — per-layer FFN skip during prefill

Design contracts for AR004 extensions (pool blocks / query gating) live in
`specs/changes/AR004-branch-gated-sparse-prefill-and-gitee-repro/design.md`;
the AR006 contract is in `specs/changes/AR006-in-graph-query-aware-sparse-prefill/design.md`.

## Two gate placement options (AR005 vs AR006)

- **Host-side selection** (AR004/AR005): the selector ranks rows/pages and the
  patch gathers them. Costs one selection pass; the AR005 semantic variant
  also pays an embedding call (~1.1 s at 4K) before TTFT can start.
- **In-graph gating** (AR006): selection happens inside the prefill graph per
  ubatch, so there is no extra host pass, no embedding, and no sync. The
  graph pays one page-mean reduction plus one top-k over `n_pages` per layer.
  In exchange the selection is per-ubatch (not per 128-token chunk), heads
  share the page set (ggml masks broadcast over heads), and the gate is only
  active when `n_kv` exceeds the local window (short prompts stay local-only).
  Since AR007 the page score is `mul_mat(k_view, q_heads)` plus a page sum
  instead of `cont(permute(k))` + pooling: same ranking, but the K cache is no
  longer copied per layer (32K gate 67.9 s -> 25-31 s).

## Admission budget: measured curve, not a constant (AR007)

The selector's budget comes from three places, in priority order:

1. an explicit conf/CLI `budget` (`budget_source=override`);
2. a **budget curve** (`budget_curve = path`): `model_sha256`-fingerprinted
   table of `prompt_length = budget`, produced by
   `benchmarks/scripts/sweep_budget.py`, read with ceiling semantics
   (`budget_source=model`);
3. the mode default 0.5, then the legacy 4096-token clamp
   (`budget_source=default|clamp`; the curve path is allowed 8192 tokens
   because its recommendation was measured to hold quality).

The curve exists because quality is not monotone in the budget: with
proportional anchors, a larger budget admits more arbitrary coverage blocks
and can push the evidence out (16K: b=0.05 q=1.00, b=0.25 q=0.36). A bench
grid can pass `budget=0` as "let the curve decide"; the runner then keeps
`budget_overridden=false` and every sample records where the number came from.
Fingerprint mismatch clears the curve before planning, so a curve is never
applied to a different model.

Selection itself gained structural anchors (sink, last block, query-word
anchors, strided coverage, priority order sink > recent > quote > stride),
span growing (adjacent candidates get +0.5 so a straddling answer stays
contiguous) and incremental MMR (near-duplicate blocks are penalised only when
their Jaccard similarity exceeds `mmr_threshold`). `Selection.anchor_blocks`
and `Selection.mmr_penalized` are printed into `raw.csv` as audit columns.

## Evidence-first admission (AR010)

The block score used to be one additive mixture of five signals, so a *prior*
(recency, information density) could admit a block on its own. A 32K audit found
62% of the budget going to blocks with zero lexical/phrase overlap. The root
cause was a normalisation bug (`minmax` kept iterators into the vector it was
overwriting, so diversity reached 3.2 instead of 1.0) plus a fill loop whose
exit condition was "the best score is exactly zero" - i.e. "fill the budget"
was a side effect rather than a policy.

The score is now split:

```text
evidence = w_lex*lexical + w_phrase*phrase + w_draft*draft
prior    = w_div*diversity + w_pos*position
fused    = evidence + sat(evidence)*prior          sat(e) = min(1, e/(0.25*e_max))
```

`prior_gate=0` restores the additive mixture. The fill has an explicit
`stop_reason` (`budget` / `evidence-exhausted` / `no-signal` / `dense`), and
`fill_priors=0` makes the budget a ceiling instead of a quota: the context is
sized by the evidence. That is what lets 32K reach q=1.000 with 525 tokens
(2.4x faster than the in-graph gate at the same quality) and cuts the 4K-16K
token counts by 1.4-2.8x at unchanged quality. New audit columns:
`evidence_blocks`, `stop_reason`.

## Multi-fact coverage admission (AR009)

AR008 made admission optimal for *one* fact per question. A question that asks
for several facts ("list the passcodes for the vaults one, three and five")
needs one block per fact, and a relevance-ranked fill spends the budget on the
single strongest match (measured: 0.488 of the facts at b=0.05, 0.261 with six
asked facts).

**Sub-queries.** The selector treats every rare query term (present in at most
two blocks) as a sub-query - rarity, not token length, because a length filter
drops short keys like `one`/`two` while keeping template words. Each sub-query
gets one anchor at `argmax(idf_p + lexical[b])`; this replaces the AR007 entity
heuristic whenever coverage is active (they are the same idea, and running both
starved the fill).

**Facility-location fill.** The greedy score gains
`coverage_weight * Σ_p max(0, idf_p - covered_p) / idf_max`, i.e. the best
*uncovered* fact a block would contribute. A block that matches most of the
question is treated as a question echo and never counts as evidence - the chat
template repeats the question in the final block, and without that exclusion the
whole question is "covered" before the first fact is admitted. With fewer than
two sub-queries no structure is built at all, which is why single-fact families
keep a bit-for-bit identical selection. `Selection.coverage_queries` (audit
column `coverage_queries`) reports how many requested facts were covered.

## Phrase-aware admission and combination routing (AR008)

Two findings drove AR008. First, the selector filtered query terms with
`len >= 2`, which silently discarded a lone digit - and a digit was the only
thing separating `vault 5` from `vault 6` in the multi-needle dataset, so
three sibling facts scored identically and the answer was whichever block came
first (quality 0.33). Second, the repository has several complete execution
paths whose cost/quality trade-off differs by prompt (32K: selection 7 s at
q=0.89 vs in-graph gate 31 s at q=1.00) and nothing chose between them.

**Phrase admission.** Query *and* document are turned into adjacent word pairs
(`vault 5`, `chapter 7`, `case 42`) whose block-level IDF makes a phrase hit
decisive; the pair is also matched across block boundaries, because a needle
sentence routinely starts in block b and ends in b+1. Blocks containing the
most discriminative phrase are pinned as anchors (protected from both anchor
trimming and stride eviction) and the hit is mirrored into `per_pos_scores`,
so the AR004 query-chunk and AR006 in-graph paths see the same discriminator.
`phrase_weight=0` disables the feature and restores the AR007 selection
exactly.

**Combination routing.** The selector pass doubles as the profiler: prompt
length, discriminative-term count, phrase-hit block count and lexical
concentration feed a measured table (`portfolio`) whose rows map
`<discriminability>:<length_bucket>` to a combo (`selection` / `gate-p16` /
`gate-p24-local32` / `dense`). The table rule is "fastest combo whose mean and
min quality reach the floor, else the best quality", and
`check_portfolio.py` enforces that the routed combo is Pareto-no-worse than
the best fixed combo, so a new arm that does not improve the combination is
rejected rather than shipped.

## Structure-bounded admission (AR011)

TTFT is proportional to the admitted token count, so the remaining lever after
AR010 is the count itself. A 4K audit showed 9 of 10 selected blocks were
mandatory: nchor_ratio capped a fraction of a budget the untrimmable anchors
(best-lexical +-2, phrase pins) had already spent.

Structure (sink + recent) now gets an absolute allowance and nchor_ratio
governs only the remaining evidence budget:

`	ext
structure       = sink_blocks + last block          (never trimmed)
evidence_budget = target - structure_tokens
anchor_cap      = structure_budget + ceil(evidence_budget * anchor_ratio)
`

The accounting only engages when the anchors actually dominate the budget
(nchor_tokens > target/2): at 16K/32K they are 20-37% of it, and forcing the
cap there admits *more* evidence (280 -> 320 tokens) for no quality gain.

Span admission (span_admission) decouples the admission unit from the
execution unit: the attention pool still addresses 32-token blocks, but the
admitted tokens are trimmed to the sentence around the evidence peak. It is
skipped whenever several facts are asked - a multi-fact answer needs one whole
sentence per fact (trimming measured 0.836 -> 0.800 and q_min 0.174 -> 0.000).
Audit columns: structure_tokens, evidence_tokens, span_tokens,
uncovered_facts.

## Sentence-complete admission (AR012)

The admission atom was the 32-token block, but fact rows do not respect block
boundaries: `... the passcode for vault six is TEAL-LYNX-8` continues in the next
block (`853`), and all-or-nothing block admission then ships half an answer
(deterministic audit: 76% of asked answers complete on the hard multi-fact tier).
Raising the budget does not help - b=0.16 measures 0.771 vs 0.836 at b=0.05,
because the extra tokens are filler.

`sentence_completion` makes the sentence the atom: every evidence token pulls in
its whole sentence, a sentence that spills into an unselected block pulls that
block into the selection (closure), and only tokens outside the admitted
sentences are trimmed. It engages only when several facts are asked; on
single-fact prompts it measured no quality gain and +15% tokens.

`sentence_trim=1` additionally drops the filler inside evidence blocks. It is
measured but never routed: it wins tokens (445 -> 372) yet costs quality
(4K/32K 1.000 -> 0.827, hard 0.739 -> 0.689) because the removed sentences carry
context the model still uses.

Audit: `sentences_completed`, `sentences_extended`, and the admitted text itself
(`--dump-selected-text`, `dump_selection_dir` in a bench conf) which
`benchmarks/scripts/check_facts_admission.py` turns into a model-free quality
signal.

## Compact context: KV sized by the admitted span (AR013)

Sparse admission cut the work 27x but left the *memory* untouched: the bench
created the context with `n_ctx = 0`, so llama.cpp reserved the model's training
length — 896 MiB of KV cache for a 4K prompt that writes 3.5 MB.

Two independent pieces fix that:

- **Lazy context sizing.** The bench starts at 256 cells and each case calls
  `LlamaBackend::ensure_context(max_position + max_new + 8)`, which re-creates
  the context only when it must and refuses anything above `n_ctx_train` with a
  clear error (an explicit `--n-ctx` that is too small also errors instead of
  truncating silently). `kv_alloc_bytes` is derived from the model geometry
  (`n_ctx x (K+V) x n_layer x head_dim x dtype`) and is documented against
  llama.cpp's own "CPU KV buffer size" line.
- **Block position compaction** (`position_compaction`). After admission the
  selector emits a second, parallel position vector: for each kept block the
  original offsets are kept verbatim and the gaps between blocks are collapsed,
  so a fact and its value keep the exact RoPE distance they had while the dropped
  filler stops occupying cells. `token_indices`, `block_indices` and the gather
  indices stay in *original* prompt coordinates — `sparse_prefill` maps them to
  decode rows — so nothing downstream (attention pool, chunk scheduler, dumps)
  had to change.

Measured (same session, real GGUF): KV 112 -> 7 MiB at 4K, 854 -> 21 MiB at 32K,
and multi-fact quality *rises* (0.887 -> 0.975) even though the deterministic
`facts_in_context` audit is identical for both arms: with the evidence blocks
adjacent instead of thousands of filler positions apart, the model can combine
them. A 64K document becomes runnable on a 32K-training-context model at all
(1163 positions, 35 MiB, q=1.000).

## Edge accounting and the dense invariant

`budget=1.0`, `qg_bound=false`, `query_gating=false`, `attn_top_k=0` and
`skip_ffn_mask=0` must remain bit-equivalent to dense. The AR006 arming call
clears every other sparse path in the context so exactly one mechanism owns
the attention graph; disabling it restores the dense graph, verified by
`bound_gating_execution_and_restore`.

## Chunked prefill

`ChunkScheduler` splits a long prompt into physical ubatches.
AR004 semantics: `chunk_size = 0` auto-aligns to `n_ubatch`;
`chunk_size >= n_tokens` collapses to a single range (dense-safe). The
attention path clamps the effective chunk to `n_ubatch` because gather
indices are chunk-scoped.

## Quality measurement

`benchmark.cpp` scores every sparse sample against a dense reference
computed in the same run: needle exact-match, QA token overlap, and PPL
retention `min(1, ppl_dense / ppl_sparse)` (per-case dense reference cached).
Samples with unusable answers are skipped with a warning, never silently.

## Testing

Unit tests run everywhere; real-model integration tests activate only when
`EDGE_PREFILL_TEST_MODEL` points at a GGUF (otherwise explicit SKIP).
Equivalence tests pin `budget=1.0` / switches-off to dense outputs.
