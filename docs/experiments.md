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

## AR006 campaign (in-graph query-aware page gating)

Everything below is CPU, Qwen2.5-1.5B-Instruct Q4_K_M, 16 threads, and was
measured in one interleaved session (dense first, then scalar / qg-2d /
bound) so the speedups are same-session ratios, never cross-session.

```bash
BIN=build-wsl/edge-prefill
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
for ds in explen/len4k explen/len8k deep/multineedle explen/len16k deep/len32k; do
  for conf in benchmarks/matrix/bound/dense-ref.conf \
              benchmarks/matrix/bound/bound-p16-a24-b025.conf \
              benchmarks/matrix/bound/bound-p24-a24-b025.conf; do
    tag=$(basename "$conf" .conf)
    $BIN bench --config "$conf" --model "$MODEL" \
        --data-dir "benchmarks/data/$ds" --bench-skip-dense \
        --out "benchmarks/results/ar006-campaign/$(basename $ds)-$tag"
  done
done
python3 benchmarks/scripts/assemble_ar006.py benchmarks/results/ar006-campaign
python3 benchmarks/scripts/check_expected_ar006.py \
    benchmarks/results/ar006-campaign benchmarks/matrix/bound/expected.csv
```

Notes:

- `assemble_ar006.py` derives the dense reference from `*-dense/` in the same
  directory and prints net speedup per (dataset, mode).
- `check_expected_ar006.py` gates on quality and on a speedup floor; it is
  machine-independent by construction (ratios only).
- `warmup=1, repeats=3` in the checked-in confs. The campaign used
  `repeats=1` for the sparse rows to fit the compute budget; TTFT medians are
  stable at 1 repeat on this model (A/B: 49.0s at 1 vs 48.9s at 5).
- `--bench-skip-dense` means "this run has one mode": without it the harness
  measures dense + the conf's mode, doubling the run.
- `--repeats` on the CLI used to silently default to 5 and override the conf
  file; fixed 2026-10-25 (sentinel -1). AR005 numbers were all 5-repeat
  medians because of it — statistically fine, but 5x the runtime.
