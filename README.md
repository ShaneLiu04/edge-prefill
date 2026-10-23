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
4. **Branch-gated sparse prefill (AR004, in progress)** — NSA/MoBA-style
   compressed branches (representative-token injection + pooled V-rows),
   per-head query gating, large-chunk scheduling, and head pattern routing
   (MInference-style) / pyramid layer prior (PyramidKV-style).

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
| `--threads N` | CPU threads | physical cores |
| `--n-gpu-layers N` | GPU offload (99 = full) | 0 |

## Expected results

Numbers below are from this repository's raw runs (16 threads, Xeon Gold
6234, Qwen2.5-1.5B-Instruct Q4_K_M; full commands + CSV in
`benchmarks/results/exp*/`). The AR004 refresh (branch + gating) is in
progress; the table reflects committed, reproducible runs.

| Setup | Prompt | TTFT | vs dense | Quality |
| --- | ---: | ---: | ---: | --- |
| dense | 16K | 165.6 s | 1.0x | 1.0 |
| selection b=0.25 | 16K | 27.2 s | 6.1x | needle hit (old metric) |
| head-adaptive k=96 | 16K | see `exp-head-16k` | ~6x | see report |

AR004 acceptance target (tracked in `specs/changes/AR004-*`): 16K TTFT
>= 10x dense with quality >= 0.9, reported with raw CSV.

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

- llama.cpp is pinned; model and data SHA256 are recorded in every report.
- No performance number without the raw CSV and the exact command.
- Prefill-only FFN skip and attention sparsity are lossy; every latency claim
  carries its quality metric in the same report.
- `budget=1.0` with all sparsity switches off is numerically equivalent to
  dense prefill (enforced by tests).
