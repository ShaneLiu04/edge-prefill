# AR016 — execution plan + latency-targeted budget

Host: Xeon Gold 6234 (16 logical), WSL2, shared machine. Model:
`qwen2.5-1.5b-instruct-q4_k_m.gguf` (sha256 `6a1a2eb6…9407e`), llama.cpp
`3bb63cbc…`. All numbers are `ttft_min` (fastest of the repeats) medians per
cell, sparse arm only (`--bench-skip-dense`), snapshots off.

> **Read the noise caveat first.** This host is co-tenanted and stalls for
> seconds at a time. The 16-thread arm is far more sensitive to that than the
> 12-thread arm (it fights for SMT siblings), so absolute base16 numbers drifted
> by up to 3.6x across rounds *within this session* (e.g. 4K: 3.45 s -> 4.40 s ->
> 12.59 s) while the plan arm stayed within 2.33-2.58 s. The matrix below is the
> last full round; §3 records the observed ranges instead of hiding them.

## 1. What the audit says (this is the part that cannot be flaky)

| check | evidence |
|---|---|
| plan reaches the runner | every row's `n_threads` equals the plan (12 CPU / 4 GPU), `plan_source=cli|conf` |
| placement is verified | CPU rows `device_state=cpu`; GPU rows `device_state=gpu`, `gpu_layers_offloaded=29`, `gpu_layers_total=29` |
| no silent CPU fallback | `plan_gpu_request_never_silently_degrades` (real model, CPU build) returns `Unsupported`; with a GPU it returns `device_state=gpu` |
| quality is untouched | 58 cases x 3 arms: f1 identical in every case (CPU and GPU matrices) |
| SLA is a real contract | every `predicted_ttft_ms <= 2500 ms`; `sla-no-grow` shows the planner never *adds* tokens |

## 2. Cost model (probe, separate process, real GGUF)

| device | a (ms) | b (ms/token) | r² | samples |
|---|---:|---:|---:|---:|
| CPU, 12 threads | -3.12 | 3.460 | 0.9994 | 5 |
| CUDA0 Quadro RTX 5000, 29/29 layers | 79.65 | 2.614 | 0.9791 | 5 |

Prefill is sublinear in the token count (the fixed graph setup is amortised), so
an affine fit legitimately lands on a slightly negative intercept on CPU; the
fitter rejects it only if a *sampled* point goes non-positive.

Controlled thread scan (same session, `run`, 270 admitted tokens, single case):

| threads | 2 | 4 | 8 | **12** | 16 |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 3172 | 1853 | 1156 | **1006** | 1375 |

16 threads (what `bench` used to force) costs **37%** over the machine's optimum.

## 3. CPU matrix — 8 datasets × 3 arms

`base16` = the old implicit behaviour (hardware_concurrency, no plan);
`plan` = probe's threads (12), no SLA; `sla` = probe's cost model, 2500 ms target.

| dataset | base16 | plan | sla | base16/plan | sel (plan -> sla) | q (all arms) |
|---|---:|---:|---:|---:|---:|---:|
| explen/len4k | 12588 ms | 2578 ms | **2125 ms** | 4.88x | 642 -> 578 | 1.000 |
| explen/len8k | 9054 ms | 4512 ms | **2210 ms** | 2.01x | 1117 -> 609 | 1.000 |
| explen/len16k | 9779 ms | 4498 ms | **2241 ms** | 2.17x | 1128 -> 616 | 1.000 |
| deep/len32k | 10085 ms | 4758 ms | **2248 ms** | 2.12x | 1166 -> 622 | 1.000 |
| deep/multineedle | 9476 ms | 4692 ms | **2195 ms** | 2.02x | 1117 -> 605 | 1.000 |
| deep/multifact8k | 2876 ms | 1877 ms | 1871 ms | 1.53x | 521 (fits) | 0.975 |
| deep/multifact8k-hard | 2571 ms | 1806 ms | 1805 ms | 1.42x | 511 (fits) | 0.746 |
| explen/len64k | 6465 ms | 4339 ms | **2226 ms** | 1.49x | 1163 -> 619 | 1.000 |

- median base16/plan = **2.01×** (range 1.42-4.88×); median plan/sla = **1.98×**.
- f1 is identical in all 58 cases across all three arms — the SLA planner changes
  the number of admitted tokens, not the answer, on these families.
- multi-fact is under the target without shrinking (521 tokens ≈ 1.9 s), so the
  planner correctly leaves it alone.

Observed ranges across the rounds in this session (same cells, same config):

| dataset | base16 | plan |
|---|---|---|
| len4k | 3.45 / 4.40 / 12.59 s | 2.33 / 2.58 s |
| len8k | 6.12 / 9.05 s | 4.19 / 4.51 s |
| len16k | 6.15 / 9.78 s | 4.22 / 4.50 s |
| len32k | 6.59 / 10.09 s | 4.35 / 4.76 s |
| multineedle | 6.36 / 9.48 s | 4.34 / 4.69 s |
| multifact8k | 2.61 / 2.88 s | 1.88 s |
| multifact8k-hard | 2.57 / 2.61 s | 1.81 s |
| len64k | 6.41 / 6.47 s | 4.34 / 4.38 s |

## 4. GPU matrix — same three arms, `--n-gpu-layers 99`

Placement is verified per row (`device_state=gpu`, `offloaded=29/29`). The GPU is
the same Quadro RTX 5000 the AR015 erratum is about, and it sits in P8
(645 MHz of a 2100 MHz maximum, 31 W of 230 W) because WSL cannot lock clocks.

| dataset | base16 | plan | sla | plan/sla | sel (plan -> sla) | q |
|---|---:|---:|---:|---:|---:|---:|
| explen/len4k | 2175 ms | 2248 ms | 2208 ms | 1.02x | 642 -> 642 | 1.000 |
| explen/len8k | 4196 ms | 4214 ms | **2788 ms** | 1.51x | 1117 -> 797 | 1.000 |
| explen/len16k | 4188 ms | 4309 ms | **2788 ms** | 1.55x | 1128 -> 800 | 1.000 |
| deep/len32k | 4351 ms | 4391 ms | **2967 ms** | 1.48x | 1166 -> 800 | 1.000 |
| deep/multineedle | 4425 ms | 4349 ms | **3095 ms** | 1.41x | 1117 -> 797 | 1.000 |
| deep/multifact8k | 1742 ms | 1749 ms | 1748 ms | 1.00x | 521 | 0.975 |
| deep/multifact8k-hard | 1698 ms | 1718 ms | 1670 ms | 1.03x | 511 | 0.746 |
| explen/len64k | 4349 ms | 4315 ms | **3076 ms** | 1.40x | 1163 -> 803 | 1.000 |

Two honest observations:

1. **Threads do not matter on the GPU** (median base16/plan = 0.99×). The gate
   says so explicitly instead of pretending the CPU win transfers.
2. **The device buys evidence, not wall time (here).** At the same 2500 ms
   target the GPU planner can afford **797-803 tokens vs 610-622 on CPU**
   (+29-31%), because its marginal slope is lower (2.61 vs 3.46 ms/token) even
   though its fixed cost is higher. On a GPU that is not P8-limited, that gap
   would widen with the slope, not with the thread count.

## 5. Gate

```
CPU  all gate checks pass   (9 checks, incl. thread-speedup 2.01x, sla 1.98x)
GPU  all gate checks pass   (9 checks, incl. placement-offload 29/29, sla 1.40x)
```

Injections must fail (and do):

| injection | result |
|---|---|
| `fake-placement` (claim a verified GPU plan on a CPU matrix) | FAIL — `placement`, `placement-offload` |
| `cost-skew` (halve the slope, i.e. an over-optimistic model) | FAIL — `sla-predicted` |

## 6. Why this is not just a bug fix

Three things were added that did not exist:

- an `ExecPlan` as an auditable object with **field-level** provenance
  (CLI > conf > probe > default) and a placement **contract** enforced after load;
- a measured, persisted **cost model** with fit-quality gates (r², sample count,
  positivity over the sampled range) — a bad model refuses to plan;
- a **latency-targeted budget solver** that re-runs the real selector (never a
  proxy for it) until the conservative prediction fits, and records
  `budget_source=cost-model` plus `predicted_ttft_ms` / `plan_select_ms`.

The bug (CLI flags dropped, model SHA256 recomputed per cell inside the timed
loop, `warmup=0`) is what made the old numbers untrustworthy; the plan is what
makes the new ones checkable.

## 7. Reproduce

```bash
./build/edge-prefill probe --model M --prompt-file CTX --threads 12 --out /tmp/plan
python3 benchmarks/scripts/ar016_matrix.py --plan-dir /tmp/plan \
    --base-threads 16 --sla-target-ms 2500 --repeats 4 --warmup 1 --force
python3 benchmarks/scripts/ar016_report.py benchmarks/results/ar016-exec-plan --gate
# GPU (build with GGML_CUDA=ON)
./build-cuda/edge-prefill probe --model M --prompt-file CTX --threads 4 --ngl 99 --out /tmp/plan-gpu
python3 benchmarks/scripts/ar016_matrix.py --bin build-cuda/edge-prefill \
    --plan-dir /tmp/plan-gpu --plan-threads 4 --gpu-layers 99 \
    --out benchmarks/results/ar016-exec-plan-gpu --force
python3 benchmarks/scripts/ar016_report.py benchmarks/results/ar016-exec-plan-gpu --gate
```
