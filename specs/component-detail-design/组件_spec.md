# EdgePrefill 组件详细设计（组件级基准）

| 字段 | 内容 |
|------|------|
| 组件名称 | edge-prefill / EdgePrefill |
| 版本 | 0.2（AR002） |
| 日期 | 2026-10-20 |
| 语言 | C++17 |
| 构建 | CMake ≥ 3.20 |
| 推理后端 | llama.cpp（固定 commit）+ GGUF |

## 1. 组件职责与边界

EdgePrefill 负责端侧 LLM 的 **prefill 阶段优化**：在真实 GGUF 模型上，通过
**draft 模型语义打分**、**块级上下文选择**、**attention 级全局/分层 top-k**、
**跨请求/多轮前缀 KV 复用（Radix 树）**、**分块 prefill**、**prefill-only FFN 跳过**、
**按层与相位感知的 KV 精度规划**、**自动调优与 Pareto 基准**，降低 TTFT 与 KV 写入量，
同时以真实基准验证质量保持。

**不负责：** 模型训练/微调、decode 算法优化、GPU/NPU 自定义 kernel、服务化多请求调度。

## 2. 架构分层

```text
┌──────────────────────────────────────────────────────────────┐
│ CLI / Benchmark Runner / TuneRunner                          │
├──────────────────────────────────────────────────────────────┤
│ Planner（预算/模式/attention top-k/自适应精度/回退）             │
│   ├─ DraftScorer                                            │
│   ├─ BlockSelector                                          │
│   ├─ AttentionSparsityController                            │
│   └─ AdaptivePrecisionController（策略规划层）                 │
├──────────────────────────────────────────────────────────────┤
│ PrefixCache(Radix)   ChunkScheduler                          │
├──────────────────────────────────────────────────────────────┤
│ SparsePrefiller（位置保持、chunk、审计）                        │
├──────────────────────────────────────────────────────────────┤
│ LlamaBackend（model/context/vocab/KV/state）                  │
│   ├─ FFNSkipController（patched llama.cpp）                   │
│   ├─ AttentionTopKController（patched llama.cpp）             │
│   └─ PerLayerKV（patched llama.cpp）                          │
├──────────────────────────────────────────────────────────────┤
│ llama.cpp / ggml（固定上游 + 最小 patch）                       │
└──────────────────────────────────────────────────────────────┘
```

## 3. 核心数据结构

| 结构 | 作用 | 关键字段 |
|------|------|----------|
| `Config` | 运行配置 | model、draft_model、budget、mode、block/chunk、attention_top_k、skip mask、KV 类型/profile、hardware_profile |
| `Plan` | 规划器输出 | passes、budget、skip mask、dense_fallback、reasons、attention_top_k、kv_profile、ffn_skip_mask、draft_enabled |
| `Selection` | 块选择结果 | block/token indices、signals、mandatory、selected_tokens、attention_indices、attention_top_k |
| `CacheHit` | 前缀命中 | prefix_len、state blob、age、node_id |
| `CacheStats` | 缓存统计 | cache_hit_rate、reused_tokens、saved_ttft_ms、saved_kv_bytes |
| `KvSignature` | KV 兼容签名 | model_key、n_ctx、type_k、type_v、per_layer_profile |
| `KvLayerProfile` | 单层 KV 类型 | type_k、type_v |
| `AdaptiveLayerProfile` | 按层调度结果 | kv_profile、ffn_skip_mask |
| `HardwareProfile` | 硬件条件 | n_threads、has_cuda、available_memory、n_ctx |
| `TuneSearchSpace` | 自动调优搜索空间 | budgets、block_sizes、attention_top_ks、kv_types、ffn_masks、chunk_sizes |
| `ChunkStats` | 分块统计 | tokens、elapsed_us、kv_pos_max |
| `Sample` | 基准样本 | ttft/e2e、tokens/s、kv_bytes、quality、quality_retention、cache_hit_rate、reused_tokens |

## 4. 关键接口契约

- 所有**可失败**的公共 API 返回 `Status`，不向调用方抛异常；纯查询/getter
  （如 `find` 返回 `optional`、`stats` 返回值类型）按签名约定返回。
- `LlamaBackend` 独占 model/context 生命周期，RAII 释放。
- `SparsePrefiller` 是稀疏 prefill 编排层：负责前缀恢复、chunk、位置保持 batch、
  FFN 模式 RAII、attention top-k RAII、首 token logits 与 `PrefillStats`。
- `BlockSelector` 不调用 llama.cpp 运行时；仅把 `llama_token`/`llama_pos` 当作整型别名，
  接收 token 与文本片段，便于单测与复用。
- `LlamaDraftScorer` 实现真实 GGUF draft 模型语义打分；draft 缺失或加载失败时回退
  lexical-only，并在 `Plan.reasons` 记录 `draft-disabled`，不伪造语义分数。
- `AttentionSparsityController` 把块选择结果映射为严格升序的 token 级
  `attention_indices`；`top_k=0` 时关闭 attention 稀疏并完全恢复原路径。
- `PrefixCache` 只管理 byte blob 与签名，不直接操作 llama context；内部使用 Radix 树，
  `find` 返回最长可复用前缀，`insert` 在关键边界写入 state snapshot，LRU 淘汰叶子节点；
  `find/insert/stats` 线程安全，并输出 `cache_hit_rate/reused_tokens/saved_ttft_ms/saved_kv_bytes`。
- `Planner` 为纯函数；相同输入必须产生相同 `Plan`。
- FFN skip 通过 `llama_set_prefill_mode` / `llama_set_prefill_ffn_skip` /
  `llama_get_prefill_ffn_skip` 控制；patch 缺失时返回 `Unsupported`，其余功能不受影响。
- Attention top-k 通过 `llama_set_prefill_attn_top_k` /
  `llama_set_prefill_attn_indices` 控制；C API 返回 `int32_t` 错误码
  （`0=Ok, 2=InvalidArgument, 9=Unsupported`），由 `LlamaBackend` 转换为 `Status`；
  indices 必须升序且在 KV 范围内，patch 缺失时返回 `Unsupported`，其余功能不受影响。
- `AdaptivePrecisionController` 输出 `AdaptiveLayerProfile`，同时管理按层 KV 类型
  （`kv_profile`）与按层深/位置的 FFN 跳过（`ffn_skip_mask`）；quality 模式自动回退
  全 F16 且关闭 FFN skip。启用 adaptive 时 `ffn_skip_mask` 覆盖 `ffn_skip.h/cpp` 的
  静态配置，未启用时完全回退 AR001 行为。
- KV 精度支持全局 `llama_context_params.type_k/type_v` 与按层
  `llama_set_kv_type_per_layer`；相同类型走单 context；
  首版 prefill/decode 类型不同时**不创建第二 context**，直接 `Unsupported` 并回退统一类型，
  `Plan.reasons` 记录 `kv-phase-fallback`；双 context handoff 为后续 AR 扩展点。
- `TuneRunner` 必须基于真实 `BenchmarkRunner` 搜索配置，输出 Pareto 前沿与
  `recommended.conf`；禁止代理指标与伪造结果。
- `Reporter` 输出 `raw.csv`、`report.md`、`env.json`，每行样本可追溯到命令与环境。

## 4b. 错误、审计与 CLI 契约

| 项 | 契约 |
| --- | --- |
| `Status` / `ErrorCode` | `Ok=0, InvalidArgument=2, ModelLoad=3, Tokenize=4, ContextInit=5, Prefill=6, Decode=7, Cache=8, Unsupported=9, OutOfMemory=10, Io=11, Internal=12`；公共 API 返回 Status，不抛异常 |
| `KvSignature` | `model_key,n_ctx,type_k,type_v,per_layer_profile`；全字段相等才可复用 |
| `Metrics` | `ttft_ms,e2e_ms,prefill_tokens_per_s,kv_bytes,quality,quality_retention,attention_top_k,per_layer_attn_stats,kv_profile,draft_enabled,cache_hit_rate,reused_tokens,saved_ttft_ms,saved_kv_bytes` |
| `EnvironmentInfo` | OS/CPU/核数/编译器/CMake/llama commit/model_sha/draft_model_sha/data_sha/threads/kv/command/timestamp |
| `Reporter` | 输出 `raw.csv`、`report.md`、`env.json`；空样本仍写表头 |
| `AttentionSparsity` | indices 必须严格升序且小于 KV 长度；`top_k=0` 关闭；C API 返回 `int32_t` 错误码，patch 缺失返回 `Unsupported` |
| `PerLayerKV` | `type_k/type_v` 数组长度必须等于 `n_layer`；长度不匹配返回 `InvalidArgument` |
| `TuneRunner` | 单个候选配置失败记录原因但不终止搜索；推荐配置必须满足质量门禁 |
| CLI 退出码 | 与 ErrorCode 一一对应；`plan/inspect` 不返回 5/6/7/8；未知参数=2 |
| 警告出口 | `Plan.reasons`、`Status.message`、CLI stderr 三者之一，测试必须断言具体字符串 |

## 5. 设计约束

1. **无 mock**：所有质量/延迟数据来自真实 llama.cpp 推理；测试可在无模型时 skip，
   但不得返回伪造结果。
2. **位置保持**：稀疏 prefill 必须使用原始 position id；budget=1.0 且关闭 attention top-k、
   FFN skip 时与 dense 等价，logits max abs diff < 1e-5。
3. **可复现**：固定 llama.cpp commit、模型/数据/draft 模型 SHA256、线程数、种子、重复次数；
   CPU 与 CUDA 实验均需可从干净克隆复现。
4. **端侧优先**：单请求、batch=1、CPU 可运行；内存额外开销 ≤20%（不含可选 draft）。
5. **可禁用**：FFN skip、块稀疏、attention top-k、前缀缓存、KV 量化、自适应精度、
   draft 模型均可独立关闭，便于消融与故障隔离。
6. **可扩展**：`DraftScorer`、`AttentionSparsityController`、`AdaptivePrecisionController`
   为本版核心接口；逐 query 动态 top-k、双相位 KV handoff 为后续 AR 扩展点。
7. **最小 patch**：llama.cpp 上游改动必须最小、可审查、可禁用，并在关闭时完全恢复原路径。
8. **真实复用**：跨请求/多轮前缀复用必须基于真实 KV state，禁止用重建 prompt 冒充缓存命中。

## 6. 对外交付

- `libedge_prefill.a` + `edge-prefill` CLI。
- 公共头文件 `include/edge_prefill/*.h`。
- 基准配置、脚本、结果报告。
- `README.md`、`docs/architecture.md`、`docs/experiments.md`、`docs/research.md`。

## 7. 变更记录

| 日期 | 变更 | 来源 |
|------|------|------|
| 2026-10-20 | 初始组件设计，建立 AR001 架构基线 | AR001 |
| 2026-10-20 | 升级为 AR002：加入 draft 模型、attention 全局/分层 top-k、Radix 前缀缓存、按层 KV 精度、自动调优与 Pareto 基准 | AR002 |
