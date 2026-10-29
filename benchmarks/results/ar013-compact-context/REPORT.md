# AR013 验收报告：紧凑上下文（KV 按准入量分配 + 位置压缩）

| 字段 | 内容 |
|------|------|
| AR | AR013 |
| 日期 | 2026-10-29 |
| 模型 | Qwen2.5-1.5B-Instruct Q4_K_M，sha256 `6a1a2eb6…9407e` |
| 主机 | Xeon Gold 6234（8C/16T），WSL2，8 线程，`repeats=5` |
| 命令 | `./repro.sh ar013`（内部：`ar013_matrix.py` + `ar013_report.py --gate`） |
| 结论 | 质量不降（多事实**提升**）、KV 分配 16–41 倍下降、64K 文档从不可运行变为可运行；TTFT 中位聚合 0.96×（不劣化） |

## 1. 问题（需求阶段实测）

`benchmark.cpp` 把 `ctx_opts.n_ctx` 硬写成 0，llama.cpp 于是按 `n_ctx_train=32768`
申请上下文：**4K 文档只用 146 个准入 token（实际写入 3.5 MB），却分配 896 MiB KV**。
后果是端侧内存直接翻倍（1.5B Q4 权重本身 ≈1 GB），且文档长度超过 32K 训练上下文时
稀疏准入也无法运行。

## 2. 两个机制

| 机制 | 开关 | 作用 |
|---|---|---|
| 上下文按跨度增长 | 默认（`n_ctx=0`） | bench 起始 256 格，按每个 case 实际写入的最大位置 `ensure_context`；显式 `--n-ctx` 不足即报错 |
| 块级位置压缩 | `position_compaction = 1` | 准入后消除块间空隙、**块内偏移原样保留**；gather 索引仍用原始坐标，映射到 decode 行号 |

分配量 `kv_alloc_bytes` 由模型几何（`n_layer × head_dim × n_head_kv × dtype × n_ctx × 2`）计算，
与 llama.cpp 自带日志 `CPU KV buffer size` 逐项对齐（4K：112 MiB ↔ 计算值 112 MiB）。

## 3. 结果（同会话，逐数据集）

| 数据集 | 准入 token | 位置跨度 AR012→AR013 | q_mean | q_min | KV 分配 | TTFT min |
|---|---:|---|---:|---:|---|---:|
| explen/len4k | 146 | 3946 → **146** | 1.000 → 1.000 | 1.000 → 1.000 | 112 → **7 MiB** | 3311 → **2163 ms** |
| explen/len8k | 188 | 7812 → **188** | 1.000 → 1.000 | 1.000 → 1.000 | 217 → **7 MiB** | 1131 → 1100 ms |
| explen/len16k | 280 | 15576 → **280** | 1.000 → 1.000 | 1.000 → 1.000 | 430 → **14 MiB** | 1641 → 1789 ms |
| deep/len32k | 525 | 31077 → **525** | 1.000 → 1.000 | 1.000 → 1.000 | 854 → **21 MiB** | 2980 → 3073 ms |
| deep/multineedle | 247 | 7857 → **265** | 1.000 → 1.000 | 1.000 → 1.000 | 217 → **14 MiB** | 1551 → 1518 ms |
| deep/multifact8k | 509 | 8571 → **515** | 0.887 → **0.975** | 0.429 → **0.600** | 238 → **21 MiB** | 3131 → 3156 ms |
| deep/multifact8k-hard | 513 | 8681 → **525** | 0.739 → **0.746** | 0.621 → **0.643** | 244 → **21 MiB** | 2482 → 2603 ms |
| **explen/len64k** | 1163 | 63900 → **1163** | AR012 无法运行 → **1.000** | — → 1.000 | — → 35 MiB | 6436 ms |

对照 AR013 之前的 bench（恒定 896 MiB）：4K **128×**、8K 128×、16K 64×、32K 42×、
多针 64×（全部同时把 `rss` 从 2.6 GB 降到 1.8 GB 量级）。

## 4. 为什么多事实会变好

确定性审计（`check_facts_admission.py`）在同一份 dump 上给出**完全相同**的
`facts_in_context`（easy 1.000、hard 0.990），即准入内容没有变化；变好的是**模型对
已准入证据的利用**：压缩把相隔数千个填充位置的证据块放到相邻位置，跨块组合变得容易。
这是"位置压缩"作为独立机制的收益，而不是又一次准入调参。

## 5. 闸门（22 项全 PASS）

- quality：7 个数据集 q_mean/q_min 不降（多事实提升）；
- time：逐数据集 `ttft_min` ≤1.10×，聚合几何均值 **0.958**（≤1.03）；
- memory：逐数据集分配严格下降，4K ≤8 MiB、32K ≤24 MiB；
- capability：len64k 上 AR012 **无样本**（`context-too-long`，6 s 退出），AR013 4/4 例 q=1.000。

注入劣化必失败（已实测）：
`--inject explen/len4k:no-compaction` → memory FAIL；
`--inject explen/len64k:no-capability` → capability FAIL。

## 6. 方法与注意事项

- 主机在本次会话中受外部负载影响极大：同一 cell 的 TTFT 中位数在两次会话间差到 5×，
  16 线程甚至比 8 线程慢 12 倍（`ttft_min` 在同一条件下稳定）。因此
  **闸门用 `ttft_min`（每次运行的最快重复）**，并在 CSV 中同时保留 median。
- len8k 在 5 次重复的矩阵里曾出现 1.15× 超限，用 7 次重复、两轮交错复测为
  1.009×/0.973×，判定为噪声，最终结果取复测值（`len8k-*/bench.conf` 为复测配置）。
- GPU（RTX 5000 / CUDA 12.0，99 层卸载）：4K 与 multifact8k 两个臂的
  `selected_tokens`、`position_span`、逐例 quality 与 CPU **完全一致**；
  multifact8k 在 GPU 上同样是 0.975。

## 7. 复现

```bash
./repro.sh ar013                    # 8 数据集 × 2 臂 + 闸门 + 注入
python3 benchmarks/scripts/ar013_report.py benchmarks/results/ar013-compact-context --gate
python3 benchmarks/scripts/check_facts_admission.py \
    benchmarks/results/ar013-compact-context --dataset deep/multifact8k --gate-arm ar013
```

原始数据：本目录 `<dataset>-<arm>/{bench.conf,*_raw.csv}`、`matrix.csv`；
逐 case 准入文本用 `ar013_matrix.py --dump` 再生（约 8 MB，不入库）。
