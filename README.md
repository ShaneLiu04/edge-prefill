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
| `--qg-bound on\|off` | in-graph query-aware page gating (AR006) | off |
| `--qg-sel-pages N` | pages kept per ubatch by the gate (quality knob) | 16 |
| `--qg-local-rows N` | local window rows around the ubatch (0 = ubatch span) | 0 |
| `--qg-pool-pages N` | pool pages folded into the gated branch | 0 |
| `--threads N` | CPU threads | physical cores |
| `--n-gpu-layers N` | GPU offload (99 = full) | 0 |

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

Raw CSV, env JSON and the conf of every cell are in
`benchmarks/results/ar007-campaign/`; the honest write-up (including the
non-monotone budget and the multi-needle weakness) is
`benchmarks/results/ar007-campaign/REPORT.md`, and the curve itself is
`benchmarks/results/ar007-sweep/{budget_curve.txt,curve.json,summary.md}`.
`budget_curve` is **opt-in** (no conf, no curve): on multi-needle documents the
selection-level path loses facts that the in-graph gate keeps, so use
`qg_bound=on` there. `benchmarks/scripts/check_expected_ar007.py` re-checks the
quality/audit columns on another machine without depending on absolute TTFT.

The gated path runs without an embedding model (`semantic_score_ms = 0`).
Absolute TTFT depends on the host (this session ran up to 3x slower than the
AR004 session on the same CPU); compare ratios measured in one session, and
use `benchmarks/scripts/check_expected_ar006.py` for the machine-independent
gate. AR006 does **not** reach 10x at 32K on CPU with the current page-mean
implementation — the reason and the next increments are in the REPORT.

## Repository layout

```
src/            planner, selector, backend, benchmark, CLI
include/        public headers (Status/ErrorCode, no exceptions across API)
patches/        llama.cpp patches (attn top-k, FFN skip)
tests/          unit + real-model integration tests (SKIP without model)
benchmarks/     configs, datasets, raw CSV reports, analysis scripts
scripts/        experiment orchestration; diagnostics/ = archived probes
specs/          SRS / design / task tracking per change request
docs/           architecture.md, experiments.md
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
