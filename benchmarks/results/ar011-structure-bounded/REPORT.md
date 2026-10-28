# AR011 验收报告：结构锚点预算与证据预算分离

| 项 | 值 |
|----|----|
| AR | AR011（structure-bounded admission） |
| 模型 | qwen2.5-1.5b-instruct-q4_k_m（SHA256 `6a1a2eb6…e9407e`） |
| 对照模型（仅前提实验） | bge-small-en-v1.5-q8_0（最终配置不使用） |
| CPU | Xeon 6234，16 线程（WSL2），llama.cpp `3bb63cb` |
| GPU | Quadro RTX 5000 16 GB，CUDA 12.0（`build-cuda`） |
| 命令 | `python3 benchmarks/scripts/ar011_matrix.py` + `ar011_report.py --gate` |
| 原始数据 | 本目录 `<dataset>-<arm>/*_raw.csv`、`matrix.csv`、`portfolio.json` |
| 结论 | **7/7 数据集不劣于 AR010（质量、token、TTFT），4K −35% token，8K −18%，多针 −23%，多事实 hard `q_min` 0.087→0.571** |

## 1. 起点（两次实测审计）

AR010 之后 TTFT ≈ `selected_tokens / 155 token·s⁻¹`，所以剩下的优化空间只在"准入 token 数"。
审计发现两件事：

1. **结构锚点吃掉了预算。** 4K NIAH（b=0.08，预算 314 token）实际准入 **310 token / 10 块，其中 9 块是
   mandatory**；多事实 hard（预算 429）准入 422 / 14 块，13 块 mandatory。
   `anchor_ratio=0.35` 对 `best_lex±2`（5 块 = 160 token）和 phrase 锚点没有约束力，
   于是"预算"在真实预算区间（b=0.05–0.08）里退化成了"结构配额"。
2. **语义（embedding）信号零收益、纯成本。** 5 个数据集 lexical vs embedding：
   mf-hard 0.334→0.337、mf 0.576→0.573、multineedle/len4k/len8k 1.000→1.000，
   而编码扫描 +0.99 s（4K）、+1.96 s（8K）、+2.19 s（8K-hard）。
   → 记录为**负结果**，默认保持 lexical，不再投入。

## 2. 做了什么

### FR-1 结构锚点与证据预算分离（`structure_budget`）

结构层（sink + recent）改为绝对预算，`anchor_ratio` 只作用于剩余证据预算；当锚点确实主导预算
（`anchor_tokens > target/2`）时，锚点上限才连 `best_lex±2` 窗口与 phrase 锚点一起生效。
长度更长、预算更宽时自动不介入（16K 实测 280 token 不变；强行介入会变 320）。

### FR-2 覆盖保底（`coverage_reserve`，默认关）

每个未覆盖子查询保留一个证据块，并审计 `uncovered_facts`。
实测在 b=0.02/0.03/0.04 上 q 变化 −0.007 / **+0.019** / 0.000，token 数不变
→ **没有可靠收益，默认关闭并写进负结果**，只保留 `uncovered_facts` 诊断能力。

### FR-3 子块（span）准入（`span_admission`）

证据块裁成"证据峰所在句子"的 token 段（先按句边界扩展，不足 24 token 再对称补齐），
跨块答案不会被切断；`block_indices` 不变（注意力池仍按块），只有 `token_indices` 变窄
—— **准入粒度与执行粒度解耦**。

**多事实场景自动禁用**：裁剪多事实选择实测 easy 0.836→0.800、hard 0.669→0.444（`q_min` 0.174→0.000），
因为每个事实需要完整句子。单事实场景保留（多针 283→247，−13%，q=1.000）。

### FR-4 组合择优（`ar011_matrix.py` + `ar011_report.py`）

规则：在**不劣于 AR010 质量**（per-dataset floor，单针 0.95、多事实以 AR010 实测为 floor）
的臂里选 **token 最少**者；闸门检查 q_mean/q_min 不降超 0.01、token 不超过最优 1.05×、
TTFT 不超过 1.05×。注入劣化自测：
`--inject explen/len8k:ar010` → `FAIL selected 228 > 1.05 x 188; ttft 1680 > 1.05 x 1506`。

## 3. 结果（同会话 CPU 矩阵）

| 数据集 | 臂 | q_mean | q_min | 准入 token | TTFT | 相对 AR010 |
|---|---|---:|---:|---:|---:|---|
| explen/len4k | AR010 | 1.000 | 1.000 | 226 | 1046 ms | — |
| | **AR011** | 1.000 | 1.000 | **146** | **755 ms** | token −35%，TTFT −28% |
| explen/len8k | AR010 | 1.000 | 1.000 | 228 | 1091 ms | — |
| | **AR011** | 1.000 | 1.000 | **188** | **1028 ms** | token −18% |
| explen/len16k | AR010 | 1.000 | 1.000 | 280 | 1444 ms | — |
| | **AR011** | 1.000 | 1.000 | 280 | 1387 ms | 持平（结构不主导，自动不介入） |
| deep/len32k | AR010 | 1.000 | 1.000 | 525 | 2371 ms | — |
| | **AR011** | 1.000 | 1.000 | 525 | 2329 ms | 持平 |
| deep/multineedle | AR010 | 1.000 | 1.000 | 321 | 1790 ms | — |
| | **AR011** | 1.000 | 1.000 | **247** | **1165 ms** | token −23%，TTFT −35% |
| deep/multifact8k | AR010 | 0.836 | 0.429 | 431 | 1924 ms | — |
| | **AR011** | 0.836 | 0.429 | 431 | 2002 ms | 持平 |
| deep/multifact8k-hard | AR010 | 0.667 | 0.087 | 445 | 1997 ms | — |
| | **AR011** | **0.709** | **0.571** | 445 | 1952 ms | q_mean +0.042，**q_min ×6.6** |

dense 参照：4K 30.6 s、8K 69.7 s、16K 173.1 s、多针 69.3 s（同一机器，dense 单元每个 case 写入 4 行）。
组合的 dense 加速：4K **40.4×**、8K **67.8×**、16K **124.5×**、多针 **59.4×**、32K **≈82×**（32K dense 复用
AR010 的 2 例子集口径 447 s，本会话未重测）。

## 4. GPU（CUDA）一致性

`build-cuda`（RTX 5000 16 GB，99 层 offload）与 CPU 跑同一配置（4K，AR011 全开）：

```
cpu  q=1.000 qmin=1.000 sel=[122,121,170,172] ttft=1091.5ms
gpu  q=1.000 qmin=1.000 sel=[122,121,170,172] ttft=  45.2ms   (24x)
```

准入在 host 侧，所以选择与质量必须逐例一致（事实如此），GPU 只加速被准入 token 的 prefill。
dense 路径在 CUDA 上会被切成数百个 graph split，未纳入结论。

## 5. 回退与负结果（诚实记录）

| 尝试 | 实测 | 处置 |
|---|---|---|
| embedding（语义）准入 | 质量 0 收益，扫描 +1.0–2.2 s | 默认保持 lexical，负结果入文档 |
| span 准入用于多事实 | easy 0.836→0.800，hard 0.669→0.444，q_min→0 | 选择器内按"无子查询"硬门控 |
| `anchor_span_radius=1` | mf 0.836→0.800、hard 0.669→0.709→(0.669 with r1) | 默认保持 ±2（r1 仅为可选项） |
| `coverage_reserve=1` | b=0.03 +0.019，b=0.02 −0.007，b=0.04 0 | 默认关，仅保留审计 |
| `structure_budget` 无条件生效 | 16K 280→320 token（质量不变） | 改为"锚点主导预算时才介入" |

## 6. 已知限制

- 多事实 easy 档仍是 0.836（6 事实 3 问、48 新 token 的部分分口径），hard 档 0.709；
- 4K/8K 的收益主要来自"锚点收窄"，16K/32K 已达证据驱动下限，AR011 不再改善；
- TTFT 跨会话可差 2–3×，本文档内所有对比均为同会话；
- 32K dense 参照复用 AR010 口径（2 例子集），未在本会话重测（单例约 7.5 min）。

## 7. 复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
EDGE_PREFILL_TEST_MODEL=... ./build/edge_prefill_tests        # 163 例
./repro.sh ar011                                              # 矩阵 + 表 + 闸门
```
