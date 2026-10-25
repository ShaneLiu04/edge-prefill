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
