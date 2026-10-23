# Experiments

All results live in `benchmarks/results/exp*/`, one directory per run, each
containing: `_raw.csv` (per-sample rows), `_report.md` (human table),
`_env.json` (model/data SHA256, llama commit, threads, exact command,
timestamp). Orchestration scripts are in `scripts/`
(`orchestrator.sh` chains phases 2-5).

## Run index (committed)

| Directory | What it measures |
| --- | --- |
| `exp-budget-*` | budget sweep (0.25 / 0.4 / 0.7) — selection-level sparsity |
| `exp-attnk-24/-48/-96`, `exp-attnk-24-fixed` | attention top-k sweep + repro fix |
| `exp-heads`, `exp-headk-64/128/256`, `exp-head-16k` | head-adaptive gather |
| `exp-grid-*` | budget x attn-top-k grid |
| `ext-ext-0.1/0.15` | extreme-budget quality cliffs |
| `exp-decode` | decode impact of sparse prefill (KV shape changes) |
| `exp-multineedle` | multi-needle robustness |
| `exp-ppl-b0.25/0.1` | PPL retention (pre-AR004 metric; superseded by retention fix) |
| `exp-gpu-dense`, `smoke-*` | GPU dense baseline, smoke runs |
| `exp-len-*` | prompt-length scaling (2K-16K) |

Headline (Xeon Gold 6234, 16 threads, Qwen2.5-1.5B-Instruct Q4_K_M,
llama.cpp `3bb63cb`):

- 16K niah, dense: 165.6 s TTFT; selection b=0.25: 27.2 s (6.1x).
- 8K niah, b=0.5 + attn-24: 2.2x with needle hit (`exp-attnk-24-fixed`).

## Environment capture

Every run records `model_sha256`, `data_sha256`, `llama_commit`, thread
count, KV type and the exact command in `_env.json`. Reproduce with:

```bash
scripts/download_model.sh   # AR004: ModelScope/hf-mirror + SHA256 verify
./build/edge-prefill bench --config benchmarks/configs/<cfg>.conf --model M
```

## Notes and caveats

- `exp-ppl-*` predates the PPL-retention metric fix (AR004 T003): quality
  values there use the old `exp(mean logprob)` scale and must not be
  compared to retention values.
- Runs before AR004 T001 used `chunk_size=512`; current default is
  `0` (auto-align to n_ubatch). Reports record the command actually used.
- The AR004 acceptance experiment matrix (16K >= 10x at q >= 0.9, CPU+GPU)
  lands with the branch/gating tasks and will refresh this index.
