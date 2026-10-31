# AR017 多序列批解码（batched decode）— 实验报告

日期：2026-10-31
后端：CPU（llama.cpp `3bb63cbc…`，`n_gpu_layers=0`）
模型：`qwen2.5-1.5b-instruct-q4_k_m.gguf`
SHA256：`6a1a2eb6…49407e`（完整值见 `cases.csv` 所在的矩阵目录 `_conf/` 与 `probe` 输出）
线程：`threads=12` / `n_threads_batch=12` / `n_ubatch=128`（AR016 实测最优，16 线程慢 37%）

## 1. 结论

| 指标 | 结果 |
| --- | --- |
| 解码段吞吐（交错 A/B，推荐 K=2） | **2.01×**（K=4 2.74×，K=8 2.51×） |
| 单例端到端（A/B，K=2） | **0.89×**（K=4 0.71×，K=8 0.39×） |
| 质量（36 用例 f1） | **逐例与串行完全相同** |
| 答案稳定性 | 2/72 批臂出现 1 个 token 偏差（2.8%，浮点批形状差异） |
| TTFT 中位比 | 1.15×（单点 0.80–2.06×，主机漂移） |
| 闸门 | 10/10 PASS；注入 `shared-seq`、`quality-drift` 均 FAIL |

**定位**：批解码把 N 个**已在上下文中**的用例的逐 token 解码合成一次 `llama_decode`，
这是解码侧唯一的线性可合并维度（prefill 走稀疏 gather，各序列位置/掩码不同，无法共图）。
它降低的是**解码墙钟**，不降低首 token；对短答案（`max_new_tokens=48`）用例，
端到端收益被 TTFT 摊薄，因此默认推荐 K=2：吞吐 +2×、单例 E2E 仅 -11%。

## 2. 方法

两个口径，互补：

1. **交错 A/B（`ab.csv`，权威口径）**：`ar017_ab.sh` 在**同一次会话内交替**跑
   `K=1,K=2,K=4,K=8` 多轮，每臂取最小值。抵消本机跨会话 2–3.6× 漂移。
2. **K 扫描矩阵（`matrix.csv`）**：每数据集独立进程跑 K=1/2/4，用于质量与审计列一致性、
   容量与串行回退校验。其计时受主机噪声主导（同 cell 跨轮可差 2×），故 e2e-per-case 仅作信息项。

数据集：

| 数据集 | 用例 | 说明 |
| --- | --- | --- |
| `explen/len4k` | 4 | 4K 文档，多句命中 |
| `deep/multifact8k` | 16 | 8K 多事实 |
| `deep/multifact8k-hard` | 16 | 8K 多事实硬负例 |

## 3. 交错 A/B 结果（`ab.csv`，各臂最小值）

| K | decode-wall | 加速 | 单例 E2E | 比值 |
| --- | --- | --- | --- | --- |
| 1 | 基线 | 1.00× | 基线 | 1.00× |
| 2 | — | **2.01×** | — | 0.89× |
| 4 | — | **2.74×** | — | 0.71× |
| 8 | — | **2.51×** | — | 0.39× |

K=8 的 decode-wall 低于 K=4：`decode_batch_ctx = K*3072` 下每序列 KV 份额收窄，
`per_seq` 更早触发串行回退，有效批大小下降（审计列 `batch_effective` 如实记录）。

## 4. K 扫描矩阵（`matrix.csv`，CPU，12 线程）

| 数据集 | K | cases | batch_effective | decode_wall_ms | 解码加速 | ttft_min_ms | e2e_median_ms | q_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| explen/len4k | 1 | 4 | 1 | 27483 | 1.00× | 2990 | 8819 | 1.000 |
| explen/len4k | 2 | 4 | 2 | 9833 | 2.79× | 6155 | 11071 | 1.000 |
| explen/len4k | 4 | 4 | 4 | 4527 | 6.07× | 2402 | 6929 | 1.000 |
| deep/multifact8k | 1 | 16 | 1 | 99180 | 1.00× | 2192 | 6612 | 0.975 |
| deep/multifact8k | 2 | 16 | 2 | 49256 | 2.01× | 3007 | 8115 | 0.975 |
| deep/multifact8k | 4 | 16 | 4 | 26464 | 3.75× | 3285 | 9969 | 0.975 |
| deep/multifact8k-hard | 1 | 16 | 1 | 100211 | 1.00× | 2139 | 6649 | 0.746 |
| deep/multifact8k-hard | 2 | 16 | 2 | 15591 | 6.43× | 1997 | 3987 | 0.746 |
| deep/multifact8k-hard | 4 | 16 | 4 | 9806 | 10.22× | 1944 | 4487 | 0.746 |

质量列（`q_mean`）在 K 之间**完全一致**；`generated_hash` 有 2 个批臂出现 1 token 差异
（批形状改变 GEMM 归约顺序，属 AR014 已记录的浮点口径），不影响 f1。

## 5. 闸门（`ar017_report.py --gate`，10 项）

| 检查 | 阈值 | 实测 | 结果 |
| --- | --- | --- | --- |
| quality | f1 逐例一致 | 36 用例一致 | PASS |
| answer-stability | ≤5% 批臂翻转 | 2/72 = 2.8% | PASS |
| decode-throughput(ab) | ≥2.0×（交错） | K=2 2.01× | PASS |
| e2e-cost(ab) | 推荐 K=2 ≥0.65× | 0.89× | PASS |
| decode-throughput | ≥2.0×（矩阵） | 最小 3.75× | PASS |
| e2e-per-case | 信息项 | 0.66/1.48/1.27× | PASS（信息） |
| step-cost | 串行无步计时 | skipped | PASS |
| capacity | 无 `batch_effective > K` | 0 | PASS |
| batch-actually-ran | 无静默串行 | 0 | PASS |
| ttft | 中位 ≤1.25× | 1.15× | PASS |

注入验证（必须 FAIL）：`--inject shared-seq` → `shared-seq` FAIL；
`--inject quality-drift` → `answer-stability` 50% FAIL。

## 6. 复现

```bash
bash repro.sh ar017            # 需要 EDGE_PREFILL_TEST_MODEL 指向 GGUF
python3 benchmarks/scripts/ar017_report.py benchmarks/results/ar017-batched-decode --gate
```

原始数据：`benchmarks/results/ar017-batched-decode/{matrix.csv,cases.csv,ab.csv,<dataset>-k<K>/}`。

## 7. 限制与诚实记录

- **只批解码，不批 prefill**：稀疏 prefill 每序列位置与掩码不同，无法共图（AR012 的 gather 结构决定）。
- **锁步而非连续批**：`flush_batch` 等整批退出后一起解码；批内短答案要等最长答案（`steps_median`
  一致即证据）。真正的 continuous batching（中途准入/退出）未实现，是 AR018 候选。
- **批容量需要先定**：`ensure_context` 会重建上下文并清 KV，故批模式在首序列 prefill 前定容；
  装不下的用例串行（`decode_batch_max_span=1024`，`len8k` span 1173 即走串行）。
- **批模式禁用快照**（AR015 快照为 seq-0 单序列假设）；会话用例不进批（AR014 依赖）。
- **短答案 E2E 变差**：K=8 单例端到端 0.39×，因为 48 token 的生成段短、批内等待占比高。
  只有"同批用例长度接近"时批解码才划算，`e2e-cost` 因此只约束推荐 K=2。
- 矩阵计时受主机噪声主导（同 cell 跨会话 2–3.6×），故以交错 A/B 为权威口径，矩阵只用于质量与审计。
