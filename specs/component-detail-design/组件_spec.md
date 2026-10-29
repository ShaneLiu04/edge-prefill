# EdgePrefill 组件详细设计（组件级基准）

| 字段 | 内容 |
|------|------|
| 组件名称 | edge-prefill / EdgePrefill |
| 版本 | 0.6（AR008） |
| 日期 | 2026-10-26 |
| 语言 | C++17 |
| 构建 | CMake ≥ 3.20 |
| 推理后端 | llama.cpp（固定 commit）+ GGUF |

## 1. 组件职责与边界

EdgePrefill 负责端侧 LLM 的 **prefill 阶段优化**：在真实 GGUF 模型上，通过
**draft 模型语义打分**、**嵌入模型（bge-small）组×块语义门控**、**块级上下文
选择**、**attention 级全局/分层 top-k**、**head 级自适应稀疏（离线 head 分类 +
任务内校准 + 层×头 top-k 矩阵）**、**在线质量风险控制**、**跨请求/多轮前缀 KV
复用（Radix 树）**、**分块 prefill**、**prefill-only FFN 跳过**、**按层与相位
感知的 KV 精度规划**、**自动调优与 Pareto 基准**，降低 TTFT 与 KV 写入量，
同时以真实基准验证质量保持。

**不负责：** 模型训练/微调、decode 算法优化、GPU/NPU 自定义 kernel、服务化多请求调度。

## 2. 架构分层

```text
┌──────────────────────────────────────────────────────────────┐
│ CLI / Benchmark Runner / TuneRunner                          │
├──────────────────────────────────────────────────────────────┤
│ Planner（预算/模式/attention top-k/自适应精度/回退/风险控制）    │
│   ├─ DraftScorer（多态：LlamaDraftScorer / EmbeddingSelectionScorer）│
│   ├─ EmbeddingScorer（bge-small GGUF，多 seq 并行块编码）        │
│   ├─ BlockSelector                                          │
│   ├─ AttentionSparsityController                            │
│   ├─ AdaptivePrecisionController（策略规划层）                 │
│   ├─ RiskController（三级风险动作）                            │
│   └─ HeadProfileLoader（离线 profile 校验与加载）              │
├──────────────────────────────────────────────────────────────┤
│ PrefixCache(Radix)   ChunkScheduler                          │
├──────────────────────────────────────────────────────────────┤
│ SparsePrefiller（位置保持、chunk、审计）                        │
├──────────────────────────────────────────────────────────────┤
│ LlamaBackend（model/context/vocab/KV/state）                  │
│   ├─ FFNSkipController（patched llama.cpp）                   │
│   ├─ AttentionTopKController（patched llama.cpp）             │
│   ├─ PerHeadAttnSparsity（ranked 池 + 层×头矩阵，patched）      │
│   ├─ PerLayerKV（patched llama.cpp）                          │
│   └─ HeadClassifier（单 head 消融探测，classify-heads 工具）     │
├──────────────────────────────────────────────────────────────┤
│ llama.cpp / ggml（固定上游 + 最小 patch）                       │
└──────────────────────────────────────────────────────────────┘
```

## 3. 核心数据结构

| 结构 | 作用 | 关键字段 |
|------|------|----------|
| `Config` | 运行配置 | model、draft_model、embedding_model、budget、mode、block/chunk、attention_top_k、skip mask、KV 类型/profile、hardware_profile |
| `Plan` | 规划器输出 | passes、budget、skip mask、dense_fallback、reasons、attention_top_k、kv_profile、ffn_skip_mask、draft_enabled、per_head_top_k（层×头矩阵序列化）、head_adaptive、risk_level |
| `Selection` | 块选择结果 | block/token indices、signals（lexical/diversity/position/draft/**phrase**）、mandatory、selected_tokens、attention_indices、attention_top_k、per_group_block_scores（G×B 语义矩阵，b 为原始块序号）、score_source（lexical/draft/embedding/混合来源审计）、anchor_blocks（结构锚点块数）、mmr_penalized（被 MMR 降权的候选块数）、phrase_blocks（短语命中块数）、specific_terms（判别词数）、lexical_concentration（top-1 词法占比，组合 profile 特征） |
| `PortfolioTable` | AR008 组合路由表 | 行式 `key = value`（`model_sha256`/`default`/`<discriminability>:<length_bucket>`），指数量级 O(1) 查表、mtime 缓存、指纹校验；输出 combo 与命中的 key 供审计 |
| `PromptProfile` | 组合 profile | n_tokens、specific_terms、phrase_blocks、lexical_concentration（全部来自选择器一次 pass，无模型调用） |
| `CacheHit` | 前缀命中 | prefix_len、state blob、age、node_id |
| `CacheStats` | 缓存统计 | cache_hit_rate、reused_tokens、saved_ttft_ms、saved_kv_bytes |
| `KvSignature` | KV 兼容签名 | model_key、n_ctx、type_k、type_v、per_layer_profile |
| `KvLayerProfile` | 单层 KV 类型 | type_k、type_v |
| `AdaptiveLayerProfile` | 按层调度结果 | kv_profile、ffn_skip_mask |
| `HardwareProfile` | 硬件条件 | n_threads、has_cuda、available_memory、n_ctx |
| `TuneSearchSpace` | 自动调优搜索空间 | budgets、block_sizes、attention_top_ks、kv_types、ffn_masks、chunk_sizes、head_adaptive |
| `ChunkStats` | 分块统计 | tokens、elapsed_us、kv_pos_max |
| `Sample` | 基准样本 | ttft/e2e、tokens/s、kv_bytes、quality、quality_retention、cache_hit_rate、reused_tokens、attention_top_k、repeats、risk_level、head_adaptive、retrieval_heads、semantic_score_ms（语义打分开销独立计时） |
| `HeadProfile` | 离线 head 分类结果 | model_sha、n_layer、n_head、probe 参数、heads[]（layer/head/echo_score/induction_score/retrieval）|
| `RiskSignals` / `RiskDecision` | 风险输入与决策 | separation/coverage/anchor_ratio → Ok/Tighten/DenseFallback + reason |
| `HeadProbeParams` | head 探测参数 | block_len、repeats、sink、recent、probe_k、seed、probe_source（random/calibration_text，任务内校准用真实文本重复段） |

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
- `EmbeddingScorer`（AR005）为独立块编码嵌入引擎（bge-small GGUF，池化=GGUF
  自带池化头：bge=CLS、通用=MEAN；无池化头的 causal GGUF 拒绝 Unsupported）：
  `load_model` 独占嵌入模型生命周期，`encode` 支持多 seq 并行编码，组文本取组内
  行的原始连续块跨度（含未选 token 文本）；输出 G×B 余弦矩阵，b 为**原始提示块
  序号**（非行空间块）；引擎输出矩阵含非有限元素返回 `Internal`（无静默截断），
  消费方 `build_query_gated` 对输入矩阵的尺寸/非有限校验返回 `InvalidArgument`。
  零 patch 改动，纯规划层，检索目标与 NIAH 任务对齐。
- `EmbeddingSelectionScorer`（AR005）把 `EmbeddingScorer` 适配为 `DraftScorer`
  多态实现，`BlockSelector` 零改动；语义分数优先参与选择权重，保证 needle 块
  进选择集（真实模型守护测试，32K budget 0.15）。
- Query-chunk 门控 v2（AR005）：`build_query_gated` 提供 2D 矩阵重载，每组独立
  对**本组**分数做因果 top-k（v1 的 1D 同源分数各组必相同，退化为全局 top-k）；
  空矩阵回退 1D 与 v1 **字节等价**；矩阵尺寸不匹配返回 `InvalidArgument`。
- 降级链（AR005）：embedding 模型缺失 → qg 回退 v1 1D → scalar → dense，每级
  在 `Plan.reasons` 记录原因；**显式配置**的 embedding 路径无效为硬错（ModelLoad），
  不降级。
- Head 任务内校准（AR005）：`classify-heads --calibration-file` 从真实任务文本
  取重复段作为探针（`ProbeSource::CalibrationText`），替代通用随机探针；
  三分数机制与 profile 格式不变，通用随机路径保留为默认。
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
- Head 级自适应稀疏（AR003）通过 `llama_set_prefill_attn_ranked_rows`（按分数
  排名的候选行池，不要求升序）与 `llama_set_prefill_attn_head_top_k`（层×头
  top-k 矩阵，`-1`=该 head 稠密直通）控制；矩阵长度必须等于 n_layer×n_head，
  元素为 -1 或 ≥1，矩阵非空要求池非空，违反返回 `InvalidArgument`；graph 每层
  按 distinct top-k 值分组（≤4 组，超限该层安全降级 dense），组内取池前 k、
  过滤未写行、内部升序 gather；矩阵优先于标量路径，清空矩阵恢复 AR002 行为；
  patch 缺失返回 `Unsupported` 并回退标量路径，`Plan.reasons` 记录
  `per-head-unsupported`；flash attention 下矩阵路径禁用（同标量限制）。
- `HeadClassifier`（classify-heads 子命令）通过 `LlamaBackend` 公共接口做单
  head 消融探测（单 head 施加 sink+recent 稀疏、其余 dense，真实前向 logits
  散度），输出确定性 JSON profile；不引入 attention dump，不产生合成分数；
  探测长度超过 n_ctx 返回 `InvalidArgument` 并给出建议参数。
- 压缩上下文分支（AR004，NSA 式）通过 `llama_set_prefill_attn_pool_blocks`
  （池化块行区间 + 块内最大 position，n≤1024）控制：已选未 gather 的满块经
  graph 内 `pool_1d(AVG)` 均值 + `get_rows` 池化进注意力集；选择级被丢块由
  规划层注入每块 1 个最高 draft 分代表 token（`compressed_rows_cap` 上限）；
  `compressed_branch=off|sample|pool` 三档，off 时与 AR003 行为逐字段一致；
  patch 缺失回退代表注入并记录 `compressed-branch-unsupported`；flash
  attention 下禁用。
- Query-chunk 门控（AR004，MoBA 式）通过 `llama_set_prefill_qg_count/
  qg_rows/qg_bounds` 控制：graph 按查询分片（默认 128 token/片，≤16 片）循环
  gather-attend-concat；每分片行集必含 sink+recent+**该分片自身查询跨度行**（AR005 修复：
  缺自身局部上下文会使隐藏态失真并污染更高层 KV cache，实测 4K needle 由 0.5 提升至 1.0）；
  与 head 矩阵路径互斥
  （矩阵优先，`query-gating-skipped`）；超限/无 draft 分数回退全局集合并审计；
  patch 缺失回退全局集合；flash attention 下禁用。
- Head 模式路由与金字塔先验（AR004，MInference/PyramidKV 式）为纯规划层：
  HeadProfile 三模式分类（锚点/局部/检索）映射 k+必留模板；无 profile 时
  `pyramid_prior` 生成层凸函数矩阵（中层最疏），输出 AR003 兼容矩阵，无新 patch。
- `RiskController` 为纯函数：draft 分数区分度/覆盖率/锚完整率 →
  `Ok/Tighten/DenseFallback` 三级动作；evaluate 失败按 `Tighten` 保守处理；
  `Plan.risk_level` 与 metrics `risk_level` 列全程审计；`--risk-guard off` 时
  决策器旁路（行为与无控制器逐字段一致）。
- `HeadProfile` 加载时校验 n_layer/n_head 与加载模型一致，不符返回
  `InvalidArgument`，不静默使用错误 profile；`--head-adaptive off` 时规划与
  AR002 全局路径逐字段一致。
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


## 4c. 图内页门控契约（AR006）

**所有权与互斥**：一个 context 同一时刻只有一个稀疏机制拥有 attention 图。
`llama_set_prefill_qg_bound(ctx, enabled, sel_pages, local_pages, pool_pages)` 成功
arming 时清空行集 qg（qg_count/qg_rows/qg_bounds）、scalar attn_top_k、head 矩阵、
ranked pool、pool blocks，并 bump `prefill_edge_epoch`；任意 `enabled=0` 调用清空字段。
`llama_get_prefill_qg_bound` 返回 enabled（0/1）。预算校验：`sel_pages<=128`、
`local_pages<=1024`、`pool_pages>=0`，越界返回 9（Unsupported）之前的 2（InvalidArgument）。

**图语义**：门控在 `build_attn`（非 FlashAttn、非 SWA、`n_head_kv(il)==k->ne[1]`、
无 sinks/`kq_b`、单流）内生效。页宽固定 32 行。打分 `score(q,p)=q·mean(K_page)`，
q 取 ubatch 末 token 在每组首个 query head 的投影；归约到全头并集（MAX）。
页选择 `top_k(score + bias, sel_pages)`，bias 把 local 页与第 0 页（sink）设为 -inf。
visible 集合 = local（sink 页 + 尾部窗口，窗口 = `local_pages*32` 或 ubatch 跨度）
∪ selected；K/V 与掩码各一次 `get_rows`，K/V 沿 ne2、mask 沿 ne0 concat。

**退化与等价**：
- `n_kv < 32`（不足一页）或 `local_start_page == 0`（local 覆盖全部缓存）→ 不建
  门控节点，走 stock attention（数学等价，避免 gather 开销）；这是 4K/8K 短提示的
  预期行为。
- `sel_pages=0` → local-only。
- `qg_bound=0` 或未 arming → 图与 dense 逐字节一致（`bound_gating_execution_and_restore`
  验证恢复后与原 dense 逐位相同）。

**审计**：`Sample.qg_source="attention-bound"`、`semantic_score_ms=0`（无嵌入）、
`attn_fallback` 反映是否回退 dense。规划侧 `Options.enable_qg_bound` 失败时降级到
行集 qg → scalar → dense 并记录 reason，绝不静默。

## 4b. 错误、审计与 CLI 契约

| 项 | 契约 |
| --- | --- |
| `Status` / `ErrorCode` | `Ok=0, InvalidArgument=2, ModelLoad=3, Tokenize=4, ContextInit=5, Prefill=6, Decode=7, Cache=8, Unsupported=9, OutOfMemory=10, Io=11, Internal=12`；公共 API 返回 Status，不抛异常 |
| `KvSignature` | `model_key,n_ctx,type_k,type_v,per_layer_profile`；全字段相等才可复用 |
| `Metrics` | `ttft_ms,e2e_ms,prefill_tokens_per_s,kv_bytes,quality,quality_retention,attention_top_k,per_layer_attn_stats,kv_profile,draft_enabled,cache_hit_rate,reused_tokens,saved_ttft_ms,saved_kv_bytes` |
| `EnvironmentInfo` | OS/CPU/核数/编译器/CMake/llama commit/model_sha/draft_model_sha/embedding_model_sha/data_sha/threads/kv/command/timestamp |
| `Reporter` | 输出 `raw.csv`、`report.md`、`env.json`；空样本仍写表头 |
| `AttentionSparsity` | indices 必须严格升序且小于 KV 长度；`top_k=0` 关闭；C API 返回 `int32_t` 错误码，patch 缺失返回 `Unsupported` |
| `PerLayerKV` | `type_k/type_v` 数组长度必须等于 `n_layer`；长度不匹配返回 `InvalidArgument` |
| `TuneRunner` | 单个候选配置失败记录原因但不终止搜索；推荐配置必须满足质量门禁 |
| CLI 退出码 | 与 ErrorCode 一一对应；`plan/inspect` 不返回 5/6/7/8；未知参数=2；`classify-heads` 额外契约：3=ModelLoad、2=参数非法（含探测长度>n_ctx）、11=输出不可写 |
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

- `libedge_prefill.a` + `edge-prefill` CLI（含 `classify-heads`、`tune` 子命令）。
- 公共头文件 `include/edge_prefill/*.h`。
- 基准配置、脚本（`setup.sh`/`download_data.sh`/`run_experiments.sh`/
  `sync_gitee.sh`）、结果报告。
- `README.md`、`docs/architecture.md`、`docs/experiments.md`、`docs/research.md`。

## 7. 变更记录

| 日期 | 变更 | 来源 |
|------|------|------|
| 2026-10-29 | 升级为 AR013：**紧凑上下文**。`LlamaBackend::ensure_context` 让 bench 的上下文按每个 case 实际写入的位置跨度增长（起始 256 格，超 `n_ctx_train` 或显式 `--n-ctx` 不足即报错，不静默截断），不再恒定申请模型训练长度（4K 文档 896 MiB → 7 MiB）；**块级位置压缩** `position_compaction`（Selection 新增平行的 `positions`/`position_span`/`compaction_reason`，块内偏移原样保留、块间空隙消除，gather 索引仍用原始坐标并在 `sparse_prefill` 映射到 decode 行号）；新增审计 `ctx_cells`/`kv_alloc_bytes`（按模型几何计算，与 llama.cpp `CPU KV buffer size` 对齐）/`kv_used_bytes`/`rss_mb`/`position_span`/`compaction_reason` 与噪声鲁棒的 `ttft_min_ms`；`ar013_matrix.py`/`ar013_report.py`（质量/延迟/KV 分配/超训练长度能力四项闸门 + 两种注入自测）；解锁 64K 文档在 32K 训练上下文模型上的运行；记录三条实测结论：只调小 n_ctx 无效（跨度仍由文档长度决定）、16 线程比 8 线程慢 12 倍（主机 SMT 争用）、`llama_state_get_size` 只反映已写入状态不能当分配量 | AR013 |
| 2026-10-20 | 初始组件设计，建立 AR001 架构基线 | AR001 |
| 2026-10-20 | 升级为 AR002：加入 draft 模型、attention 全局/分层 top-k、Radix 前缀缓存、按层 KV 精度、自动调优与 Pareto 基准 | AR002 |
| 2026-10-21 | 升级为 AR003：加入 head 级自适应稀疏（ranked 池 + 层×头 top-k 矩阵 patch、消融式 head 分类器）、在线质量风险控制器、TuneSearchSpace 增 head_adaptive 维度、Sample 增 risk_level/head_adaptive/retrieval_heads、classify-heads CLI 契约、sync_gitee.sh 交付 | AR003 |
| 2026-10-22 | 升级为 AR004：加入压缩上下文分支（pool_blocks patch + 代表注入）、query-chunk 门控（qg_count/qg_rows/qg_bounds patch）、head 模式路由与金字塔先验、chunk_size=0 自动对齐 n_ubatch 语义、Gitee 开放复现契约（download_model.sh 双通道+SHA256、结果 CSV/图入库） | AR004 |
| 2026-10-28 | 升级为 AR012：**句子完整准入**（证据 token 拉入整句、可跨块闭包补齐、`sentence_completion`/`sentence_trim`/`span_max_tokens`，仅多事实启用）；新增审计 `sentences_completed`/`sentences_extended` 与可解释性出口 `--dump-selected-text`、bench `dump_selection_dir`（逐 case `_selected.txt`/`_selection.json`）；新增 `check_facts_admission.py`（不经过模型打分的"事实是否完整在上下文"确定性指标）、`ar012_matrix.py`/`ar012_report.py`（"没有免费收益"闸门 + 注入自测）；**修复超限安全网按 mandatory 优先截断导致丢证据的 bug**；记录"加预算负收益"与"裁剪伤质量"两条负结果 | AR012 |
| 2026-10-27 | 升级为 AR011：**结构锚点预算与证据预算分离**（`structure_budget` 绝对锚点预算 + `anchor_ratio` 只作用于证据预算，且仅当`anchor_tokens > target/2` 时介入；`anchor_span_radius` 可配窗口）；**准入粒度与执行粒度解耦**（`span_admission` 单事实下把证据块裁成"证据峰所在句子"的 token 段，`block_indices` 不变、`token_indices` 变窄，多事实自动禁用）；新增 `coverage_reserve`（每事实保底，实测无稳定收益，默认关）与审计字段 `structure_tokens`/`evidence_tokens`/`span_tokens`/`uncovered_facts`（Selection/Sample/raw.csv/dump JSON）；新增 CLI `--structure-budget`/`--anchor-span-radius`/`--span-admission`/`--span-min-tokens`/`--coverage-reserve`；修复 `run` 路径未把 phrase/coverage/prior/fill 传入选择器的遗留 bug；新增 `ar011_matrix.py`/`ar011_report.py`（不劣于 AR010 的质量+token+TTFT 闸门与注入自测）；记录 embedding 准入零收益负结果 | AR011 |
| 2026-10-27 | 升级为 AR010：**修复 `minmax` 归一化别名 bug**（`*lo/*hi` 为被写向量的迭代器，导致 diversity 归一化到 3.2、position 到 1.157，先验可对零证据块打分）；**证据/先验分离**（`evidence = w_lex·lexical + w_phrase·phrase + w_draft·draft`、`prior = w_div·diversity + w_pos·position`、`fused = evidence + sat(evidence)·prior`，`prior_gate` 开关）；**显式停止语义**（`stop_reason` ∈ budget/evidence-exhausted/no-signal/dense，`fill_priors=0` 内容定预算）；新增 `evidence_blocks`/`stop_reason` 审计列与 `--prior-gate`/`--fill-priors` CLI；新增组合脚本 `ar010_matrix.py`/`ar010_report.py`（含 Pareto 闸门与注入自测）与 CUDA 一致性验证；**更正 AR009 多事实地板**（原数字含归一化 bug 红利） | AR010 |
| 2026-10-27 | 升级为 AR009：加入**多事实覆盖准入**（子查询=稀有查询词 df≤2、问句回声块排除、每事实一锚点 `idf_p+lexical[b]`、facility-location 覆盖增益 `Σ max(0,idf_p−covered_p)/idf_max`、`coverage_weight` 配置键/CLI、`coverage_queries` 审计列、覆盖启用时取代 AR007 entity 锚点启发式），修复单事实族零回归（cw=0 或多事实结构不成立时选择逐位等于 AR008）；**修复 AR008 遗留越界**（短语外溢 `per_pos` 用 `flat_seq` 下标当 token 下标 → `flat_tok` 词→token 映射，ASan 确认 heap-buffer-overflow）；新增 `gen_multifact.py` 多事实族（拼写数字键 + 部分分）、`ar009_ablation.py`、`check_expected_ar009.py` 闸门 | AR009 |
| 2026-10-26 | 升级为 AR008：加入**位置短语准入**（问句相邻词对作独立词项、保留单字符数字判别符、跨块 bigram、块级短语 IDF、`signals.phrase`、短语块锚定+裁剪保护、短语命中外溢到 `per_pos_scores` 供 AR004/AR006 共享），修复多针检索"唯一判别符被过滤"根因（多针 q 0.33→1.00）；加入**实测驱动的组合路由**（`portfolio` 表 + prompt profile：判别词数/短语块数/词法集中度；combo ∈ selection/gate-p16/gate-p24-local32/dense；`portfolio_choice`/`portfolio_reason`/`phrase_blocks`/`specific_terms` 审计列）与 **Pareto 组合回归闸门**（`check_portfolio.py`，路由结果不得劣于最佳固定组合） | AR008 |
| 2026-10-26 | 升级为 AR007：加入**预算-质量曲线驱动的长度自适应预算**（`budget_curve` 行式 `key = value` + `model_sha256` 指纹校验、ceiling 语义、`budget_source=model|override|default|clamp` 审计列、bench 的 `budget=0` 哨兵表示"让曲线决定"）、**结构锚点与 span growing**（sink/末块/问句词锚点/位置采样，裁剪优先级 sink>recent>quote>stride，相邻已选块 +0.5 保持证据 span 连续）、**增量 MMR 去冗余**（`max_sim[]` 只在新增入选块时刷新，`mmr_threshold`/`mmr_lambda` 可配）、**页打分去 K 拷贝**（patch 侧 `q_heads=cont(view_3d(q))` + `mul_mat(permute(k))` + 页内 `sum_rows`，替代 `cont(permute(k))+pool_1d`，32K 门控 67.9s→25-31s）、Selection/Sample 增 `anchor_blocks`/`mmr_penalized` 审计列与 `sweep_budget.py`/`ar007_accept.sh`/`check_expected_ar007.py` 复现资产 | AR007 |
| 2026-10-25 | 升级为 AR006：加入图内查询感知页门控（patch 侧 `llama_set_prefill_qg_bound`，每 ubatch 每层 K 页均值打分 + 图内 top-k + single-union gather，无嵌入、`semantic_score_ms=0`）、页→行 row_table 查表（ggml 无 F32→I32 cast）、每图一次 mask 预转置、local 覆盖全缓存时自动跳过门控（短提示等价 dense）、稀疏路径互斥单一 owner 与 RAII 恢复；新增 `qg_bound/qg_sel_pages/qg_local_rows/qg_pool_pages` 配置键与 CLI | AR006 |
| 2026-10-24 | 升级为 AR005：加入嵌入模型语义引擎（bge-small GGUF，EmbeddingScorer + EmbeddingSelectionScorer 多态适配，零 patch）、qg v2 组×块 2D 门控（每组本组分数因果 top-k，1D 回退字节等价）、head 任务内校准（--calibration-file 真实文本探针）、Selection 增 per_group_block_scores/score_source、Sample 增 semantic_score_ms 净值记账、语义降级链（embedding→qg v1→scalar→dense 逐级审计） | AR005 |
