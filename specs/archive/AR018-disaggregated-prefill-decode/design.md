# [AR018] 设计说明（prefill / decode 分离与跨上下文 KV 交接）

## 1. 结构

```
bench (main thread)                          prefill worker thread
─────────────────────                        ──────────────────────
case i: selection (纯 CPU, 毫秒级)
   │
   ├─► PipeTask{tokens, selection, opts, required_ctx}
   │        └──► 队列（长度由 pipeline_ahead 控制）
   │                                    ▼
   │                          prefill_backend（独立 model + context）
   │                            ensure_context(required × 2 × K)
   │                            reset_sequence(0)
   │                            SparsePrefiller::prefill(...)   ← AR012/13 稀疏路径
   │                            get_seq_state(0)  → KV + 位置
   │                            copy_last_logits() → 首 token 分布
   │                                    │
   │        ◄──── PipeResult{state, logits, stats, prefill_ms}
   │
   ├─► BatchJob{deferred, pipe_index, answer}
   │
   └─► 队列 ≥ K + ahead 时 flush_batch(K)：
         wait(pipe_result) → set_seq_state(seq, state)
         pos = seq_pos_max(seq) + 1
         next = greedy(logits)
         decode_batch(AR017 锁步) → 采样 → 计分 → 释放序列
```

## 2. 关键决定与依据

| 决定 | 依据 |
|---|---|
| prefill 与 decode 各用一个 `llama_model` 实例 | llama.cpp 的 `n_gpu_layers` 是**模型加载参数**，同一模型无法给不同 `llama_context` 指定不同设备。两个实例共享同一文件页（mmap），代价是 1 GB RAM + 1 GB VRAM（实测 RSS/VRAM 见 REPORT） |
| 只把 **KV 状态**（`llama_state_seq_*`）跨上下文搬运 | 这是上游为 server slot 迁移提供的接口，语义就是"另一上下文可精确续写"；不用 patch、不依赖内部布局。交接后做 logits 等价断言（测试） |
| 首 token 分布随状态一起搬（`copy_last_logits`） | 解码上下文必须从 `seq_pos_max+1` 开始写（llama.cpp 断言），不能重放最后一个 prompt token，否则位置冲突 |
| prefill 上下文 `n_seq_max = K` | `llama_state_seq_get_data` 的状态按 **stream（=seq）** 序列化；两侧 stream 数不一致时 `set_data` 直接报 `n_stream mismatch`（实测） |
| prefill 上下文 `n_ctx = required × 2 × K` | 关闭 `kv_unified` 后每个 stream 独占 `n_ctx/n_seq_max` 格，且一个 ubatch 需要**连续**的一段（AR017 实测的 ~2× 因子）。按 `required` 申请会立刻 `failed to find a memory slot`（实测） |
| 解码槽位在 flush 时才分配 | 队列里可能同时存在 K+ahead 个用例，入队时分配会写出 `seq_id ≥ n_seq_max`，直接 GGML_ASSERT 崩溃（实测一次，已修） |
| flush 只取前 K 个（`flush_batch(limit)`） | 队列刻意保持比 K 长（lookahead 才是重叠的来源），但解码上下文只有 K 条序列 |
| 队列触发线 `K + pipeline_ahead`（默认 2K） | 若等 K 个就 flush，worker 在解码期间无事可做，重叠为 0 |
| 流水线模式禁用快照与会话；dense 臂显式跳过 | 快照/会话绑定单上下文；dense 臂要整篇文档，批上下文装不下。两者都写告警，**不静默**（FR-4） |
| 线程异常必须在 worker 内捕获 | llama.cpp 会 `throw std::runtime_error`；逃出线程 → `std::terminate`（实测一次，已修） |
| worker 用 RAII 保证 join | 任何 early return 都会析构 joinable `std::thread` → `std::terminate`（实测一次，已修） |

## 3. 审计列

| 列 | 含义 |
|---|---|
| `pipeline_mode` | 该样本是否走流水线 |
| `prefill_device` / `prefill_device_state` | 预填充上下文实际所在的设备与 AR016 放置判定（`gpu` / `cpu` / `cpu(fallback)`） |
| `queue_wait_ms` | 预填充完成到真正开始解码之间的排队时间 |
| `handoff_ms` / `handoff_bytes` | 状态搬运的耗时与字节数（~16 MB / 1K token 上下文） |
| `ttft_ms`（流水线） | 预填充自身耗时（设备时间） |
| `e2e_ms`（流水线） | `prefill + queue_wait + batch_wall`（产品视角，含排队） |

## 4. 与 AR017 的关系

流水线复用 AR017 的锁步批解码与 `BatchJob`，只是把"谁做 prefill"换成了另一个上下文的 worker。
`pipeline_mode=0` 时行为逐位不变（AR017 闸门原样通过）。

## 5. 失败模式（都必须显式）

| 失败 | 表现 |
|---|---|
| 预填充失败 / 异常 | `Status` 经 `pipe_wait` 回传，run 返回错误，不产出半成品样本 |
| KV 交接失败（截断/流数不符） | `set_seq_state` 返回 `Cache` 错误，run 终止 |
| 放置与请求不符 | 两个后端各自跑 AR016 的 `PlacementReport::resolve` |
| 上下文装不下用例 | 预填充上下文按 `required×2×K` 增长；仍失败则报错 |
| dense 臂 / 快照 / 会话 | 显式跳过 + 告警，不当成成功 |
