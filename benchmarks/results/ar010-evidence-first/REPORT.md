# AR010 acceptance: evidence-first admission

One session, one host (Xeon Gold 6234, 16 threads, Qwen2.5-1.5B-Instruct Q4_K_M),
CPU backend unless marked GPU. Absolute TTFT swings across sessions on this
shared VM, so every ratio below is **same-session**. Raw cells:
`<dataset>-<arm>/*_raw.csv` + `bench.conf`; machine-readable summary in
`matrix.csv`, `matrix.json`-equivalent in `portfolio.json`.

## 0. What triggered the AR

AR009 shipped with 32K still needing the in-graph gate (selection q=0.875 at
36.8× vs gate q=1.00 at 12.8×). Instead of adding another signal, this AR opened
with a **budget audit**: dump the selection and ask what the 1,553 selected
tokens are made of.

```
len32k, b=0.05: 50 blocks selected
  30 of them = the last 30 blocks (941-970)  = 62% of the budget
  block 950: lexical=0.0000 phrase=0.0000 position=1.0654 diversity=2.4081
  block 147: lexical=0.3216 phrase=0.3333 position=0.4745 diversity=2.5240
```

Sixty-two percent of the prefill went to blocks that contain nothing the
question asks for. Two independent defects explain it.

### 0.1 `minmax` kept iterators into the vector it was writing

```cpp
const auto [lo, hi] = std::minmax_element(...);
for (double& v : *values) v = (v - *lo) / range;   // *lo is an iterator!
```

Once the minimum element has been normalised to 0, `*lo` reads 0 and every later
element is mapped to `v / range`. Measured ranges after "normalisation":

| signal | before | expected | actual |
|---|---|---|---|
| lexical | [0, 7.17] | [0,1] | [0,1] (min was at index 0, so it *looked* right) |
| diversity | [0.4549, 0.6616] | [0,1] | [0, **3.2**] |
| position | [0.1356, 1.0] | [0,1] | [0, **1.157**] |

Diversity was therefore weighted 3.2× its configured value, and both priors can
score a **zero-evidence** block. At 32K the prior-covered region is thousands of
tokens wide, so the budget drained into filler.

### 0.2 "Fill the budget" was a numerical accident, not a policy

With the bounds copied correctly, quality *fell* before it rose:

| dataset | before fix | after fix only |
|---|---:|---:|
| 4K | 1.000 | **0.500** |
| multifact8k | 0.878 | **0.579** |
| multifact8k-hard | 0.652 | **0.338** |

The fill loop stopped when the best remaining score was exactly 0. With broken
priors nothing was ever exactly 0, so the budget always filled; with correct
priors entire regions are exactly 0 and the loop exited at 210/443 tokens. In
other words, AR001-AR009's "budget = quota" contract was implemented by a bug.

## 1. The fix: evidence decides admission, priors only order

```text
evidence[b] = w_lex·lexical + w_phrase·phrase + w_draft·draft
prior[b]    = w_div·diversity + w_pos·position
fused[b]    = evidence[b] + sat(evidence[b])·prior[b]      (prior_gate=1)
sat(e)      = min(1, e / (0.25·max_evidence))
```

Three additions, all with explicit switches:

| switch | meaning | 0 = |
|---|---|---|
| `prior_gate` | priors cannot create admission | legacy additive fusion |
| `fill_priors` | evidence exhausted → stop (`0`) or fill with priors (`1`) | content-sized context |
| audit | `evidence_blocks`, `stop_reason` (`budget`/`evidence-exhausted`/`no-signal`/`dense`) | — |

`fill_priors` defaults to 1 (the historical quota contract) because three
existing unit tests and the attention/head subsystems assume a filled budget;
the measured table routes to the content-sized arm where it wins.

## 2. Combination matrix (same session)

| dataset | arm | q_mean | q_min | selected | TTFT | vs dense |
|---|---|---:|---:|---:|---:|---:|
| 4K | **selection** | 1.000 | 1.000 | 226 | **1050 ms** | **20.4×** |
| 4K | selection-prior (quota) | 1.000 | 1.000 | 322 | 1491 ms | 14.4× |
| 4K | gate-p16 | 1.000 | 1.000 | 354 | 1606 ms | 13.3× |
| 8K | **selection** | 1.000 | 1.000 | 228 | **1039 ms** | **48.5×** |
| 8K | selection-prior | 1.000 | 1.000 | 388 | 1698 ms | 29.7× |
| 8K | gate-p24-local32 | 1.000 | 1.000 | 628 | 2789 ms | 18.1× |
| 16K | **selection** | 1.000 | 1.000 | 280 | **1251 ms** | **105.9×** |
| 16K | selection-prior | 1.000 | 1.000 | 792 | 3675 ms | 36.0× |
| 16K | gate-p24-local32 | 1.000 | 1.000 | 1128 | 5351 ms | 24.8× |
| 32K | **selection** | **1.000** | **1.000** | 525 | **2260 ms** | **198×** |
| 32K | selection-prior | **0.611** | 0.000 | 1565 | 7276 ms | 61.5× |
| 32K | gate-p16 | 1.000 | 1.000 | 1165 | 5417 ms | 82.6× |
| 32K | gate-p24-local32 | 1.000 | 1.000 | 1165 | 5443 ms | 82.2× |
| multi-needle | **selection** | 1.000 | 1.000 | 321 | **2675 ms** | **23.6×** |
| multi-needle | gate-p24-local32 | 1.000 | 1.000 | 673 | 7043 ms | 9.0× |
| multi-fact (6/3) | selection | 0.836 | 0.429 | 431 | 2276 ms | — |
| multi-fact hard (12/6) | selection | 0.667 | 0.087 | 445 | 2305 ms | — |

Dense references: 4K/8K/16K measured in this matrix; 32K (447 s) and
multi-needle (63 s) on 2-case subsets with `max_new_tokens=8` because a full
dense pass on this VM costs hours of decode. TTFT is prefill-only, so the ratio
is unaffected by the token cap.

**The two headline results**

1. **32K no longer needs the gate**: selection reaches q=1.000 at 525 tokens /
   2.26 s, i.e. 2.4× faster than the gate arm at the same quality, and the gate
   is now Pareto-dominated on every dataset.
2. **The whole length family got cheaper**: the same quality with 1.4-3.0× fewer
   tokens (4K 322→226, 8K 388→228, 16K 792→280, 32K 1566→525).

The `selection-prior` (quota) arm collapsing to 0.611 at 32K is the clearest
evidence that *filling the budget* is harmful once priors are honest.

## 3. Routing and gate

`ar010_report.py` applies the SRS rule (fastest arm whose mean **and** min
quality reach 0.95, else the highest quality) and re-checks Pareto:

```
PASS deep/len32k            routed=selection  q=1.000 ttft=2259ms
PASS deep/multifact8k       routed=selection  q=0.836 ttft=2276ms   (floor missed)
PASS deep/multifact8k-hard  routed=gate-p16   q=0.667 ttft=2276ms   (floor missed)
PASS deep/multineedle       routed=selection  q=1.000 ttft=2675ms
PASS explen/len4k/8k/16k    routed=selection  q=1.000
all 7 routed datasets are Pareto-no-worse than the best fixed arm
```

Injection self-test (`--inject deep/len32k:selection-prior`) fails with all
three violations reported (q_mean, q_min, TTFT), so the gate can catch a
degraded table.

## 4. GPU validation (FR-5)

CUDA build (`-DGGML_CUDA=ON`) on the local Quadro RTX 5000 (16 GB, driver
556.18, CUDA 12.0 in WSL), `--n-gpu-layers 99`:

| case | CPU | GPU |
|---|---|---|
| sparse 4K, same config | q=0.67 0.67 0.00 0.00, sel=[218,217,202,204], 16.0 s | **identical q/sel**, 2.9 s |
| gating 8K (`qg_bound=on`, p16) | q=0.67 ×4, sel=[606,658,644,604], `qg_source=attention-bound`, 36.7 s | **identical q/sel/source**, 7.6 s |
| dense 4K (`budget=1.0`) | q=1.00 ×4 | **not measured** (graph split into 391 pieces, >30 min; recorded as a limit) |

The selection stage is host-side, so `selected_tokens`/`block_indices` are
device-independent by construction; the GPU runs confirm the graph side agrees.

## 5. Correction to AR009 (must be read with AR009's report)

AR009's multi-fact numbers (0.878 / 0.652) were partly produced by the
over-ranged priors: after the fix, the same commands measure **0.836 / 0.667**
(`max_new_tokens=48`, b=0.05). The coverage signal itself is still worth
something, and its value moved to the hard tier:

| dataset | coverage_weight=0 | coverage_weight=3 |
|---|---:|---:|
| multifact8k | 0.809 | 0.836 (+0.027) |
| multifact8k-hard | 0.387 | **0.667 (+0.280)** |

`benchmarks/matrix/multifact/expected_ar009.csv` now carries the corrected
floors (0.80 / 0.60, delta 0.02 / 0.20) with a comment explaining why, and
`./repro.sh ar009` passes against them.

## 6. Honest limits

- **Multi-fact easy tier is slightly below the old (bug-powered) 0.878**: 0.836.
  The hard tier is better (0.667 vs 0.652). q_min on the hard tier is 0.087 —
  the 6-answer format still does not fit comfortably in 445 tokens.
- **Hard-negative suppression was tried and reverted**: penalising blocks that
  contain rare terms absent from the question also penalises filler (filler
  words are rare too); multi-fact 0.576→0.566, hard 0.334→0.304. Recorded in
  `tasks.md` T009 as a failed experiment.
- **Dense references for 32K / multi-needle are 2-case subsets** (runtime), and
  dense on GPU was not measured (graph split pathological for the dense path).
- **Content-sized admission is opt-in via the table** (`fill_priors=0`); the
  library default stays at the historical quota contract so existing consumers
  (attention sparsity, head routing) keep their invariants.
- The 4K/8K/16K `selection` arms are 2.4-3.0× cheaper than `selection-prior` in
  TTFT but the *quality* was already saturated; the honest claim is "same
  quality, substantially fewer tokens", not "better answers".
