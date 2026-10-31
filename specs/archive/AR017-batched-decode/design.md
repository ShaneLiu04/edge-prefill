# [AR017] 详细设计

## 1. 结构

```
bench (K>1)
  ├─ 建上下文：n_seq_max=K，n_ctx = K*1536（可配），一次性定容
  ├─ 逐用例（串行）：准入 → 在自己的 seq 上 prefill → 取首 token → 入队
  │     └─ 该用例 span+gen 超过 n_ctx/K 时：先冲批，再用串行路径重算容量
  ├─ flush：锁步（lockstep）批解码
  │     decode_batch([tok_i], [seq_i], [pos_i])  → 每序列一个 logits
  │     token_i ← greedy(logits_i)；EOS 或达 max_new 的序列退出后续步
  └─ 收尾：quality = f1(generated, answer)，generated_hash 写入审计
```

新增/修改：

| 位置 | 内容 |
|---|---|
| `include/edge_prefill/backend.h` | `prefill(..., int32_t seq_id=0)`、`reset_sequence(int32_t)`、`decode_batch(...)` |
| `src/backend.cpp` | 多序列 batch 构造（每 token 一个 `seq_id`/`pos`/`logits=1`）、重复 seq 拒绝、logits 逐 token 取回；`kv_pos_*` 按 `seq_id` 读 |
| `src/benchmark.cpp` | `BatchJob` 队列、`flush_batch` 锁步循环、每用例独立 seq、批容量检查、快照禁用、`generated_hash` |
| `include/edge_prefill/{benchmark,metrics,config}.h`、`src/{config,main,metrics}.cpp` | `decode_batch_size`/`decode_batch_ctx` + 5 个审计列 |

## 2. 为什么这样划分

**只批解码，不批 prefill。** prefill 走的是 AR002/AR012/AR013 的稀疏路径：token 子集 +
位置压缩 + （可选）gather 索引。这些状态是**上下文级**的（`set_attn_indices` 等），
多个序列共用一张图做不到；而解码是 dense 的、每序列一个 token，天然可以合并。
把批处理放在"解码"这一层，准入/预算/KV 布局全部不动——这也是"不改算什么、只改怎么算"。

**每用例一个 seq，而不是共享 seq。** 共享 seq 会让第二个用例的 KV 覆盖第一个
（不同文档、不同位置），输出必然错；`decode_batch` 显式拒绝重复 seq_id。

**容量先定后批。** `ensure_context` 会重建上下文，KV 全清。批内任何一次增长都会
毁掉已 prefill 的兄弟序列，所以批的 `n_ctx` 必须在第一个序列 prefill 之前定好，
装不下的用例退回串行（先冲批，再按串行路径扩容）。

## 3. 关键实现细节

1. `decode_batch` 的 logits：`batch.logits[i]=1`，用 `llama_get_logits_ith(ctx, i)`
   顺序取回——与输入顺序一致，因此不需要位置反查。
2. EOS 语义与串行路径一致：**先追加 piece，再判断 EOS**，否则最后一个 EOS 会多/少一个 token。
3. 位置：`job.pos = stats.kv_pos_max + 1`，每步 +1；`stats.kv_pos_max` 已按 `seq_id` 读。
4. 批内序列数可以小于 K（数据跑完或容量降批）：`batch_effective` 记录真实值。
5. 快照（AR015）与批处理互斥：`save_state/load_state` 是 seq-0 专用，批模式禁用并告警，
   不做"悄悄恢复到错误的序列"。
6. 会话（AR014）不走批：多轮之间有数据依赖，且它们有自己的串行代码路径；
   `c.session.empty()` 是批分支的前置条件。
7. 审计：`batch_size`、`batch_steps`、`batch_wall_ms`、`seq_id`、`decode_ms_per_token`、
   `generated_hash`。`generated_hash` 让"批不改答案"可以被逐字节断言，而不是只看 f1。

## 4. 度量口径（最容易出错的地方）

| 口径 | 定义 | 用在哪 |
|---|---|---|
| 单例 E2E | `ttft + 该批的 wall`（批内所有序列共享同一批 wall） | 用户感知延迟 |
| 解码段 wall | `batch_wall × ceil(cases / K_eff)` | 吞吐（**不能**把每例的 batch_wall 相加，那会重复计 K 次） |
| TTFT | prefill 时间，批处理不参与 | 首 token 延迟 |

`ar017_report.py` 的 throughput 检查用的是"解码段 wall"，并同时报告单例 E2E 比值——
两者分开，避免用"总 e2e 相加"把 K 倍重复计算成"提速 K 倍"。

## 5. 测试

| 层级 | 用例 |
|---|---|
| 真机 | `batched_decode_matches_per_sequence_decode`：同前缀 + 同 next token，批解码 logits 与逐序列 **max|diff|=0**，argmax 相同 |
| 真机 | `batched_decode_rejects_duplicate_sequence`：重复 seq → `InvalidArgument` |
| 真机 | `batched_bench_matches_serial`：临时数据集 K=1 vs K=2，逐例 quality 与 `generated_hash` 相同 |
| 集成 | 199→203 例全绿；K=1 路径与 AR016 完全一致 |
| 集群 | `ar017_matrix.py` + `ar017_report.py`（质量/答案哈希/解码吞吐/单例 E2E/TTFT/容量）+ 两种注入 |

## 6. 风险与取舍

| 风险 | 处理 |
|---|---|
| 批解码引入浮点差异 | 实测 max|diff| = 0（同一图、同一 ubatch 形状），仍保留 `generated_hash` 闸门 |
| KV 容量不够导致降批 | 显式 `K_eff` + 审计；剩余用例进下一批，不丢用例 |
| 批内短序列被长序列拖慢 | 退出机制：先结束的序列不再占 batch 位置；报告中给出单例 E2E |
| 主机抖动下吞吐测量不稳 | 用进程 wall + 解码段 wall 两个口径，并记录 K_eff |
| 与快照/会话的语义冲突 | 批模式禁用快照、会话不进批，都在审计里可见 |

## 7. 组合关系

- AR012/AR013：决定每序列长度 → 决定批容量；
- AR016：提供线程/设备计划与 SLA；批处理不改变 TTFT，因此与 SLA 正交；
- AR014：会话串行，不进批；
- AR015：批模式禁用快照（seq-0 限制），文档写明。

**回退条件**：若 `ar017_report.py` 的质量或单例 E2E 不达标，则 `decode_batch_size` 保持 1
（默认值），AR017 只保留"多序列解码原语"，并在 portfolio 记负结果。

## 8. 实现差异与实测约束（as-built）

### 8.1 多序列 KV 的容量系数是 2，不是 1

在 `n_seq_max=K` 的统一 KV 里，一个序列实际占用的单元格约为其 span 的 **2 倍**
（每个 stream 会预留一个 ubatch 窗口）。实测（len8k，span 1173）：

| decode_batch_ctx | per-seq 份额 | 结果 |
|---:|---:|---|
| 3072（K=2，1536/seq） | 1536 | prefill 失败：`failed to find a memory slot for batch of size 512` |
| 4096（2048/seq） | 2048 | 同上 |
| 6144（3072/seq） | 3072 | prefill 通过 |

因此默认 `decode_batch_ctx = K * 4096`，并把可同批的 span 上限设为
`decode_batch_max_span = 1024`（`required_ctx <= 1024` 才进批），超出即串行。
这是**显式的容量边界**，不是静默降级：审计列 `batch_size=1` 可见。

### 8.2 解码位置必须来自缓存，不能来自 stats

批内第一个解码 token 的位置必须等于该序列的 `last_pos + 1`，llama.cpp 会断言。
初版用 `stats.kv_pos_max + 1`，在重复/回收序列时出现 1–20 个位置的漂移
（`Y = X + 1` 断言失败）。改为 `LlamaBackend::seq_pos_max(seq) + 1`（直接读 KV）。

### 8.3 批结束后必须释放所有序列

不释放会在统一缓存里留下死单元格，跨批累积后使后续 prefill 找不到 512 的 ubatch 槽位。
`flush_batch` 结束即对 0..K-1 全部 `reset_sequence`。

### 8.4 失败用例必须可见

初版把 `prefill-failed:<case>` 写进 warning 就丢掉该用例（len8k 只出了 3/4 行）。
现在 warning 带 `Status.message` 并落 stderr；同时用上面的 span 上限把"跑不了的用例"
明确路由到串行，`cases` 数与数据集一致（闸门的 `capacity` 检查）。

### 8.5 度量口径

- `decode_wall = batch_wall * ceil(cases / K_eff)`（不能把每例的 batch_wall 相加，会重复 K 次）；
- 单例 E2E = `ttft + batch_wall`，与串行同义；
- 交错 A/B（`ar017_ab.sh`，同数据集、K 交错、取各臂最小值）用于抵消主机漂移。
