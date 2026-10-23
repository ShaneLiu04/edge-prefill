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
- `llama_set_prefill_skip_ffn_mask` — per-layer FFN skip during prefill

Design contracts for AR004 extensions (pool blocks / query gating) live in
`specs/changes/AR004-branch-gated-sparse-prefill-and-gitee-repro/design.md`.

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
