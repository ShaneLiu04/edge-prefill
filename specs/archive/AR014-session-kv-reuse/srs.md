# [AR014] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR014 |
| AR 主题 | 会话级 KV 复用（append-only 上下文 + 增量准入） |
| 关联 SR | 无（延续 AR001 端侧稀疏 prefill 目标） |
| 日期 | 2026-10-29 |
| 状态 | Draft |

## 1. 背景与目标

AR013 之后单次请求已经很快（4K 750ms–2.1s，KV 7 MiB），但**每次请求都从零重建上下文**：
`sparse_prefill.cpp` 在每次 `prefill()` 前调用 `reset_sequence()`，bench 也把
`enable_prefix_cache` 固定为 false。真实端侧场景（多轮问答、RAG 同一文档连续提问）里，
第 2..N 轮消化的内容与第 1 轮高度重叠，却每一轮都付一次全量 prefill。

需求阶段可量化的依据（全部来自已入库的 AR013 实测，真实 GGUF）：

| 观测 | 数值 | 来源 |
|---|---:|---|
| 每 token 的 prefill 成本 | ≈ 5.5–6 ms | AR013 matrix（515 token ↔ 3.1 s） |
| 多事实单轮准入 | 509 token | `ar013-compact-context/matrix.csv` |
| 每轮 `reset_sequence()` | 1 次 | `sparse_prefill.cpp` |
| 复用已写入 KV 的代码路径 | 只有 `PrefixCache`，且仅 dense、bench 里关闭 | `prefix_cache.cpp` |

SOTA 参考（思想，不引依赖）：

| 项目 | 核心思想 | 本项目差异 |
|---|---|---|
| vLLM automatic prefix caching | 前缀相同 → 直接复用 KV，只 prefill 增量 | 我们的"前缀"由准入决定，每轮增量还要经过证据准入 |
| SGLang RadixAttention | 请求间共享前缀树、LRU 淘汰 | 单会话顺序 append，不做树 |
| CacheBlend | 复用的 KV 需要少量重算来修正上下文漂移 | 我们的复用是**严格同前缀**（append-only），无需重算 |
| RAGCache | 文档级 KV 缓存 + 查询只算增量 | 我们缓存的是"准入后的证据子集"，不是整篇文档 |

**目标**：让多轮会话的每轮成本与该轮**新增证据**成正比，而不是与"已累积上下文"成正比；
且**不牺牲质量**（同一 token 序列、同一位置，答案必须与重算一致）。

## 2. 需求范围

**In Scope：**
- `include/edge_prefill/session.h` + `src/session.cpp`：会话状态（append-only log、已入 log 的原始 token 集合、下一个位置）
- `src/benchmark.cpp`：会话模式（数据集 `index.tsv` 第 6 列 `session`），同一 session 的 case 共享 KV
- `src/sparse_prefill.cpp`：`Options.append`（不清序列、只解码增量、位置从会话游标继续）
- `include/edge_prefill/metrics.h`、`src/metrics.cpp`：`session_id`/`reused_tokens`/`appended_tokens` 审计列
- 多轮真实数据集（`deep/multiturn`）+ 生成器
- 矩阵脚本、闸门、文档、repro

**Out of Scope：**
- 跨进程/跨模型的 KV 落盘（只做进程内会话）
- 多会话并发调度与 LRU 淘汰（会话数由矩阵控制，内存用 AR013 的紧凑布局约束）
- 修改 llama.cpp 上游

## 3. 功能需求

### 3.1 FR-1 会话级 KV 复用

**描述**：同一 `session` 的多个 case 在同一个 llama sequence 上顺序执行，已写入的 KV 不再重算。

**触发条件**：`index.tsv` 带第 6 列 `session`，且 conf `session_reuse = 1`。

**期望行为**：
- 每个 session 维护 `log`（append-only 的 token 序列）与 `admitted`（已入 log 的原始文档 token 下标）；
- 新一轮开始时，只把"本轮准入但尚未入 log 的证据 token"（保持原文档顺序）与本轮问句追加到 log；
- 追加部分用 `llama_decode` 从 `next_pos` 连续写入，**不清空序列、不重解码前缀**；
- 审计 `session_id`、`reused_tokens`（log 中已存在、本轮未重算的 token 数）、`appended_tokens`。

**异常处理**：同一 session 的 case 若 context 不一致 → 明确报错（`InvalidArgument`），不静默跨文档复用；`append` 失败时序列标记为不可复用并重新开始（记录 `session_reset_reason`）。

**验收标准**：
- Given `deep/multiturn`（8 会话 × 5 轮），When 开启会话复用，Then 第 k 轮 `appended_tokens` 单调不增且恒 < log 长度；
- Given 同一 session，When 连续 5 轮，Then 序列只重置一次（会话首轮）。

### 3.2 FR-2 质量等价（同前缀、同位置）

**描述**：会话臂与"每轮重算同一条 log"的对照臂必须产生**完全相同的 token 序列与位置**。

**触发条件**：两种模式跑同一数据集。

**期望行为**：
- 对照臂（`session_reuse = 0`）：每轮把当前 log 从位置 0 起完整 prefill；
- 会话臂（`session_reuse = 1`）：只解码增量，位置从上一轮末尾连续；
- 两臂写出 `session_log_hash`（token 序列 + 位置的哈希），作为"输入相同"的硬证据。

**验收标准**：
- Given 多轮数据集，When 两臂都跑，Then 每轮 `session_log_tokens` 相同且 `session_log_hash` 相同
  （f1 翻转造成的级联轮次除外）；
- Given 多轮数据集，When 比较 f1，Then 会话臂均值不低于重算臂（容差 0.005），
  且偏离 1e-9 以上的轮次 ≤5%，单轮下降不超过 0.25。

**实测偏差（ST 阶段记录）**：40 轮中 39 轮 f1 逐位相同；1 轮（`mt-03-t4`）hash 相同但 f1
0.200→0.800。根因为整段分块 prefill 与增量 decode 在 ggml 中的 tiling 不同导致的末位比特差异，
被 greedy 解码放大，两次独立运行可复现。因此 FR-2 的"逐例相同"以"log 相同"为硬判据、
以 f1 的统计界为质量判据，理由见 `design.md` §7.4 与 REPORT §4。

### 3.3 FR-3 延迟收益

**描述**：每轮 TTFT 与该轮新增 token 成正比。

**验收标准**：
- Given 5 轮会话，When 对比两臂，Then 第 5 轮 TTFT ≤ 0.5× 对照臂，且 5 轮平均 ≤ 0.6×；
- Given 单轮数据集（8 个），When 会话关闭，Then 与 AR013 逐位一致（回归保护）。

### 3.4 FR-4 组合闸门与回退

**描述**：`ar014_report.py --gate` 同时检查质量等价、多轮延迟、单轮回归；不达标项必须回退。

**验收标准**：
- Given 多轮矩阵 + 单轮回归矩阵，When 跑闸门，Then 全部 PASS；
- Given 注入 `deep/multiturn:no-reuse`（把会话臂的 TTFT 换成对照臂），Then FAIL；
- Given 注入 `deep/multiturn:quality-drift`（人为改一个答案），Then FAIL。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 延迟 | 第 5 轮 TTFT | ≤ 0.5× 重算臂；平均 ≤ 0.6× |
| 质量 | 逐例 f1 | log/hash 相同；均值不降，翻转轮 ≤5% |
| 内存 | 会话 KV | 5 轮累计 ≤ 1.5× 单轮准入量（AR013 紧凑布局下 span = log 长度） |
| 兼容 | 无 `session` 列 / `session_reuse=0` | 与 AR013 行为逐位一致 |
| 确定性 | 同输入两次运行 | token 序列与 TTFT 分布稳定 |

## 5. 约束与假设

**约束：**
- 严格同前缀才会复用（append-only），不做 CacheBlend 式的"拼接任意 KV + 重算修正"；
- 不引入新依赖；C++17；`-Werror`；无跨 API 异常；
- 所有质量数字来自真实 GGUF，不用 mock。

**假设：**
- 多轮场景中，后续问题所需的证据与已准入证据有重叠，增量远小于全量；
- 位置连续追加不改变模型对已入 log 内容的注意力语义（同前缀、同位置）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| log | 会话的 append-only token 序列（准入证据 + 问句 + 分隔符） |
| 增量 | 本轮需要新写入 log 的 token |
| 复用 | 已写入 KV 的前缀被直接沿用，不计入本轮 TTFT |
| 重算臂 | 每轮从 0 重新 prefill 同一 log 的对照配置 |
