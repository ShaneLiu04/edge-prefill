# [AR013] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| AR 主题 | 紧凑上下文稀疏 prefill（KV 按准入量分配 + 位置压缩） |
| 关联 SR | 无（延续 AR001 端侧稀疏 prefill 目标） |
| 日期 | 2026-10-29 |
| 状态 | Draft |

## 1. 背景与目标

AR012 之后质量层面已收敛：5 个单事实数据集 q=1.000，多事实 easy/hard 0.887/0.739，
事实完整率 0.990–1.000。但**资源层面有一个从未被审视的量级问题**。

需求阶段用现有二进制做的实测（真实 GGUF，`benchmarks/data/explen/len4k`，146 准入 token）：

| 观测 | 数值 | 来源 |
|---|---:|---|
| `llama_context` 申请的 KV 缓冲区 | **896.00 MiB** | llama.cpp 日志 `CPU KV buffer size` |
| 实际写入的序列状态 | 3.5 MB（122 token） | `raw.csv` 的 `kv_bytes` |
| 准入 token | 146（4K 文档 3962 token） | `raw.csv` |
| 原因 | `benchmark.cpp:278` 硬写 `ctx_opts.n_ctx = 0` → llama.cpp 取 `n_ctx_train = 32768` | 代码 |

即：**稀疏准入把计算量降到 1/27，却仍然按文档全长分配 KV**。端侧后果：

1. 内存：4K 文档也要 896 MiB KV，32K 文档同样 896 MiB；1.5B Q4 权重本身才 ~1 GB，
   KV 占了同等量级，4 GB 内存的端侧设备直接出局。
2. 能力上限：文档超过模型 `n_ctx_train`（本模型 32768）时，稀疏准入也无法运行——
   稀疏化本该让"长文档 + 小 KV"成立，现在被上下文上限卡死。
3. TTFT：准入 token 只 146–509，但每次 prefill 仍按 32K 上下文走
   （实测 5.5–9.6 ms/token，dense 4K 7.7 ms/token）。

**目标**：让 KV 与上下文规模**跟随实际准入量**，而不是跟随原始文档长度；
并借此解锁"文档长度 > 模型训练上下文"的能力。

参考的核心思想（均已在 SOTA 项目中验证，本项目按其思想重新实现，不引入依赖）：

| 来源 | 借用的思想 | 本项目差异 |
|---|---|---|
| vLLM PagedAttention / prompt caching | KV 按实际 token 分配与管理 | 我们的 token 集合是"稀疏+乱序"的，先压缩位置再分配 |
| StreamingLLM / attention sink | 位置重映射不破坏结构 token | 只压缩**块间空隙**，块内偏移严格保留 |
| RoPE position interpolation（YaRN 等） | 位置可被压缩/重标定 | 我们不缩放也不插值，直接采样原始位置（保序、保块内距离） |
| CacheBlend | 复用已算 KV、只重算少量 token | 本 AR 只做"分配与位置"，跨请求复用留给后续 AR |

## 2. 需求范围

**In Scope：**
- `src/benchmark.cpp`：上下文按实际准入量申请，n_ctx 真实生效并进入审计
- `src/selector.cpp` / `include/edge_prefill/selector.h`：位置压缩（块级）与索引重映射
- `src/sparse_prefill.cpp`、`src/attention_sparsity.cpp`：gather 索引、chunk 边界随压缩位置更新
- `src/metrics.cpp`、`include/edge_prefill/metrics.h`：`ctx_cells`、`kv_alloc_bytes`、`position_span` 审计列
- `src/main.cpp`、`src/config.cpp`：`--position-compaction off|block`、conf 同名键
- 测试、矩阵脚本、闸门、数据集（64K）、文档、repro

**Out of Scope：**
- 跨请求 KV 复用 / CacheBlend 式拼接（独立 AR）
- 改变 RoPE 本身（不修改 llama.cpp 权重或 rope 参数）
- 量化 KV（`kv_type` 已有，不在此 AR 调参）
- 修改 `budget=1.0` 的 dense 等价路径

## 3. 功能需求

### 3.1 FR-1 上下文按准入量分配（n_ctx 真实生效）

**描述**：bench 不再硬写 `n_ctx=0`，默认按"本 case 实际会写入的最大位置 + 生成余量"申请上下文。

**触发条件**：`edge-prefill bench` 任意 case。

**期望行为**：
- 默认 `n_ctx = max_position + max_new_tokens + 64`（压缩开启时 `max_position` 为压缩后位置）；
- `--n-ctx N` / `n_ctx = N` 显式覆盖优先；
- `ctx_cells`、`kv_alloc_bytes`（= `ctx_cells × 每 token KV 字节`，由模型属性计算）写入 `raw.csv`；
- 日志中 llama.cpp 的 `CPU KV buffer size` 与 `kv_alloc_bytes` 一致性由 ST 断言。

**异常处理**：显式 n_ctx 小于所需位置 → 明确报错（`InvalidArgument`），不静默截断。

**验收标准**：
- Given 4K NIAH case（准入 146 token）、默认压缩，When 跑 bench，Then `kv_alloc_bytes ≤ 8 MiB`（现 896 MiB）且 q 不变；
- Given 显式 `--n-ctx 128` 而文档需要 4000 位置，When 跑 bench，Then 返回非零状态并给出清晰错误。

### 3.2 FR-2 块级位置压缩

**描述**：准入后把保留 token 的位置重映射为紧凑递增序列，**块内偏移严格保留**，块间空隙被压掉。

**触发条件**：`position_compaction = block`（默认 off，库默认保持不变）。

**期望行为**：
- 对每个保留块 `b`（按原位置升序），新位置 = `Σ_{b'<b} len(b') + (原位置 − block_start(b))`；
- `block_indices`、`mandatory_blocks`、`evidence_blocks` 语义不变（仍是原块号），只有 token 位置变化；
- 依赖位置的下游全部重映射：`attention_indices`、chunk 边界、`prefix_len`、dump 里的 `positions`；
- 位置跨度 `position_span = max_position+1` 写入审计，便于对比压缩率。

**异常处理**：压缩后位置必须严格递增（断言）；任何重映射失败返回 `Status` 错误并降级为不压缩（记录 `compaction_reason=fallback`）。

**验收标准**：
- Given 一个 4K 文档与 AR012 组合配置，When 开启 block 压缩，Then 7 个数据集中每个数据集的 `q_mean`/`q_min` **不下降**；
- Given 同一 case，When 压缩开启，Then `position_span` ≤ 准入 token 数（即块间空隙为 0），且块内相邻 token 的位置差与原文档一致（单测）；
- Given 压缩开启，When 跑 4K/8K/16K/32K/多针/多事实，Then TTFT 相对 AR012 **不增**（中位数容差 3%），
  有改善则如实记录（需求阶段的微实验表明时间收益有限，本 AR 的主要收益是内存与能力上限，不虚报）。

### 3.3 FR-3 超训练长度的文档承载

**描述**：文档长度超过模型 `n_ctx_train` 时，稀疏准入 + 位置压缩应能正常推理。

**触发条件**：`bench` case 的 token 数 > `n_ctx_train`（新增 64K NIAH 数据集）。

**期望行为**：
- 准入后位置跨度落在申请上下文内，`llama_decode` 正常；
- 记录 `prompt_tokens`（原始）与 `position_span`（压缩后）的对比；
- dense 路径在同样输入下必然失败（n_ctx 不足），需要明确报错而不是崩溃。

**验收标准**：
- Given `benchmarks/data/explen/len64k`（真实生成的 NIAH，4 例），When 用 AR012 组合+block 压缩跑，Then 全部 case 成功且 `q_mean ≥ 0.80`；
- Given 同数据 dense（`budget=1.0`），When 跑，Then 返回明确错误（上下文不足），不崩溃、不 OOM 杀进程。

### 3.4 FR-4 组合闸门与回退

**描述**：任何一项创新若不改善组合性能，必须回退到 AR012 组合。

**期望行为**：`benchmarks/scripts/ar013_report.py --gate`：
- 质量：7 个数据集 `q_mean` 不比 AR012 低（容差 0，逐数据集）；
- 时间：TTFT 中位数不增；若质量有提升，允许 token 增长（沿用 AR012 规则）；
- 资源：`kv_alloc_bytes` 在 4K 上至少 32× 下降；32K 上至少 100× 下降（文档更长则更多）；
- 注入 `deep/len4k:baseline`（把压缩关掉、n_ctx 打回全量）必须 FAIL。

**验收标准**：Given 生成的矩阵，When 跑闸门，Then 4 项全 PASS；Given 注入劣化，Then FAIL 且报告指出违规项。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 内存 | 4K case 的 KV 分配 | ≤ 8 MiB（基线 896 MiB） |
| 性能 | TTFT | 不劣于 AR012（中位数容差 3%） |
| 质量 | 7 数据集 q_mean / q_min | 不下降 |
| 兼容 | `budget=1.0` + 全部开关关闭 | 与 dense 数值等价（现有单测保持） |
| 确定性 | 同一输入两次运行 | `token_indices` 与位置完全一致 |
| 可移植 | 压缩后位置 | 不依赖 GPU/CPU 差异（选择阶段在主机侧） |

## 5. 约束与假设

**约束：**
- 不修改 llama.cpp 上游代码中 RoPE/位置处理逻辑（只改我们写进 batch 的 `pos`）；
- 不引入新依赖；C++17、`-Werror` 零告警、无跨 API 异常；
- 不得用 mock 或合成分数：所有质量数字必须来自真实 GGUF 推理。

**假设：**
- 块内 token 的相对距离对注意力最重要（块是语义/句子的自然单位），块间空隙是"填充噪声"；
- 模型对绝对位置的依赖弱于相对距离（Qwen2.5 使用 RoPE，长上下文训练充分）；
- 64K 文档在 32K 训练上下文下，稀疏+压缩后有效上下文 ~1K token，位置跨度也随之缩小。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 位置跨度 `position_span` | 写入 KV 的最大位置 + 1，决定最小可用上下文 |
| 块级压缩 | 保留块内原始偏移、跨块累计偏移重排位置 |
| KV 分配 `kv_alloc_bytes` | `ctx_cells × 每 token KV 字节`（模型属性计算，可与 llama.cpp 日志对齐） |
| n_ctx_train | 模型训练上下文上限（本模型 32768） |
