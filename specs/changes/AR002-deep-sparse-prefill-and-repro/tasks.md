# [AR002] 任务跟踪：深度稀疏 Prefill 与自适应复用创新

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-20 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | AR001 基线审计与兼容性回归：确认现有块稀疏、前缀缓存、FFN skip、KV 精度在 AR002 开发期间不回归；修复 QA `qa-02` 质量回退与 `test_planner.cpp` balanced FFN skip 语义不一致 | - | passing | 2026-10-20 完成 |
| T002 | llama.cpp attention 级稀疏最小 patch：实现全局/分层 top-k gather（`ggml_get_rows`）、可审查运行时开关与 `llama_set_prefill_attn_top_k`/`llama_set_prefill_attn_indices` API | T001 | passing | 2026-10-21 完成 |
| T003 | Draft 模型与分层稀疏选择器集成：真实 GGUF draft 语义打分、块级选择 + attention 级 top-k 候选排序与 `Plan` 新字段 | T002 | passing | 2026-10-21 完成 |
| T004 | 分层稀疏 prefill 执行路径：位置保持、因果性、dense 回退、审计统计与 benchmark `repeats/warmup` | T003 | passing | 2026-10-21 完成 |
| T005 | 跨请求与多轮前缀 KV 复用：Radix 树缓存、最长前缀命中、共享 system prompt、多轮追加 | T001 | pending | 真实 state 复用 |
| T006 | KV 分页与快照管理：state 序列化、LRU、签名校验、内存统计与恢复失败回退 | T005 | pending | 防错误复用 |
| T007 | 按层 KV 精度与相位感知调度：`llama_set_kv_type_per_layer` patch、首尾层保守策略、prefill/decode 不一致返回 `Unsupported` 并回退 | T004 | pending | 质量门禁约束 |
| T008 | 自动调优与 Pareto 基准框架：基于真实 `BenchmarkRunner` 搜索 budget/top-k/KV/FFN/chunk，输出前沿与 `recommended.conf` | T007 | pending | 不用代理指标 |
| T009 | CPU/CUDA 实验环境与数据集扩展：支持 GPU offload、多模型、多数据集 | T001 | pending | 服务器复现基础 |
| T010 | 复现脚本与文档：setup、download、run、tune 脚本与 README/实验文档 | T008, T009 | pending | 干净克隆可复现 |
| T011 | 质量保障与回归测试：dense 等价、无 mock、质量门禁、真实模型集成测试与 PPL 综合指标 | T008 | pending | 硬门禁 |
| T012 | 8K/4K 主实验矩阵与参数调优：CPU + CUDA 验证 TTFT ≥2.5x/1.8x、质量 ≥97%、共享前缀 ≥5x | T011 | pending | 主验收 |
| T013 | 性能与质量迭代：针对未达标项继续优化稀疏率、精度与复用策略 | T012 | pending | 直到达标或明确不可达 |
| T014 | 架构文档与研究分析：分层稀疏、自适应复用、先进项目对比与创新总结 | T013 | pending | 体现深度创新 |
| T015 | 代码审查与规范合规：Status/RAII/命名/警告/测试覆盖复核 | T014 | pending | Review 阶段 |
| T016 | ST 验收与结果归档：端到端真实模型验收、报告归档、验收记录 | T015 | pending | ST 阶段 |
| T017 | Gitee 同步与 release tag：推送仓库、README 校验、版本标记 | T016 | pending | 交付阶段 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-20 会话记录

- 完成任务：T001 AR001 基线审计与兼容性回归
- 修改文件：
  - `tests/test_planner.cpp`
  - `tests/test_selector.cpp`
  - `src/selector.cpp`
  - `.gitignore`
- 测试结果：
  - CTest 无模型：1/1 通过
  - CTest 真实模型：1/1 通过
  - QA 8K：dense 4/4，sparse 4/4，质量保持率 100%，平均 TTFT 加速 2.33x
  - NIAH 8K：dense 4/4，sparse 4/4，质量保持率 100%，平均 TTFT 加速 2.43x
- 备注：
  - 修复 `planner_balanced_8k` 断言与实现的 Balanced FFN skip 语义不一致
  - 修复 selector `best_lex` 被问题后缀自我匹配污染的问题
  - 新增 `selector_query_suffix_does_not_dominate_lexical` 回归用例

### 2026-10-21 会话记录

- 完成任务：T002 llama.cpp attention 级稀疏最小 patch
- 修改文件：
  - `patches/llama-prefill-attn-top-k.patch`（新增，268 行）
  - `scripts/apply_llama_patch.sh`（支持两个 patch 依次应用）
  - `include/edge_prefill/backend.h`（新增 `set_attn_top_k`/`set_attn_indices` 等 4 个方法）
  - `src/backend.cpp`（C API 的 Status 包装与错误码转换）
  - `tests/test_integration.cpp`（新增 `integration_attn_top_k` 测试）
- patch 内容：
  - `include/llama.h`：4 个 C API（`llama_set_prefill_attn_top_k`、`llama_get_prefill_attn_top_k`、`llama_set_prefill_attn_indices`、`llama_get_prefill_attn_indices_size`），返回 int32_t 错误码
  - `src/llama-cparams.h`：`prefill_attn_top_k` + `prefill_attn_indices` 配置
  - `src/llama-context.cpp`：API 实现与输入校验（严格升序、< n_ctx、负值返回 2）
  - `src/llama-graph.h/.cpp`：`llm_graph_input_attn_kv` 新增 `self_attn_idx` 输入张量；`build_attn` 在 prefill 模式下对 K/V/KQ mask 执行 `ggml_get_rows` gather；graph `can_reuse` 检查 top-k 一致性
- 关键实现决策：
  - gather 方式：K/V/KQ mask 分别 permute 到 n_kv 在 dim1 的布局，用 `ggml_get_rows` 按 indices gather，再 permute 回原布局并 `ggml_cont`
  - 因果性：直接 gather 原 KQ mask 的行，天然保持因果语义
  - V 布局：trans-V 和 non-trans V 统一 gather-back 为非 trans 布局，`build_attn_mha` 的 `v_trans` 检测与 `!v_trans` 分支自动处理
  - `ggml_get_rows` 支持 F16/Q8_0 等量化输入（输出 F32），无需 cast
  - 限制：首版不支持 flash attention（F16 mask），文档记录
- 调试修复：
  - `ggml_permute` 语义：`ne[axis_i] = a->ne[i]`，trans-V 的 permute 参数曾写反
  - `ggml_soft_max` 要求 mask 连续，gather 后需 `ggml_cont`
  - `ggml_mul_mat` 要求非 transposed 输入，gather-back 后需 `ggml_cont` 统一为非 trans 布局
- 测试结果：
  - CTest 无模型：1/1 通过
  - CTest 真实模型：1/1 通过（37 个用例全 PASS）
  - `integration_attn_top_k`：API 校验（负值/非升序/越界返回 InvalidArgument）、边界等价（top_k=n 时与 dense argmax 相同）、gather 激活（半数 indices 时 logits 有限且与 dense 不同）、禁用恢复（top_k=0 后与 dense max diff < 1e-5）
  - 全部原有测试无回归（dense 路径在 top_k=0 时完全不经过 gather 代码）

### 2026-10-21 会话记录（2）

- 完成任务：T003 Draft 模型与分层稀疏选择器集成
- 修改文件：
  - `include/edge_prefill/draft_scorer.h` + `src/draft_scorer.cpp`（新增）：`DraftScorer` 基类（自 selector.h 拆出）、`DraftScorerOptions`、`LlamaDraftScorer`（真实 GGUF 语义打分）
  - `include/edge_prefill/attention_sparsity.h` + `src/attention_sparsity.cpp`（新增）：`AttentionSparsityController::build` 从 `Selection` 派生 attention gather indices
  - `include/edge_prefill/selector.h` + `src/selector.cpp`：`Selection` 新增 `attention_indices`/`attention_top_k`；draft 打分失败回退 lexical-only（reason "draft-disabled"），绝不伪造分数
  - `include/edge_prefill/config.h` + `src/config.cpp`：`Config.attn_top_k`/`attn_top_k_layers`、`Plan.attention_top_k`、`parse_u32_list`
  - `src/planner.cpp`：sparse 路径规划 `attention_top_k`，passes 加 "attn_top_k"
  - `src/main.cpp`：CLI `--draft-model`/`--attn-top-k`/`--attn-top-k-layers`；run_command 加载 draft、构建 indices
  - `tests/test_draft_scorer.cpp`、`tests/test_attention_sparsity.cpp`（新增，13 个用例）
- 关键实现决策：
  - Draft 打分：draft tokenizer 编码 query → 全 token embedding mean-pool 归一化；prompt 单次 chunked forward，块边界取 final hidden state，与 query 向量 cosine，per-block 分数广播到 per-token
  - llama.cpp 本 commit 语义：decoder 生成模型取 per-token embedding 需 context `embeddings=true` + `LLAMA_POOLING_TYPE_NONE`（qwen2 graph 设 `res->t_embd`）；此时 batch 层强制所有 token 为 outputs，`llama_get_embeddings_ith` 的入参是 **batch 内 token 索引**（经 `output_ids` 映射到输出行），边界提取必须传 chunk-local 行号而非输出计数
  - vocab 覆盖检查：token id 超出 draft vocab → Unsupported，selector 回退 lexical-only
  - indices 语义：dense → 空 "dense"；top_k=0 → 空 "attn-disabled"；kept 覆盖全部 selected → 空 "attn-top-k-covers-all"（gather 全量=纯开销）；否则 kept 块 ∩ token_indices 严格升序，mandatory 锚（sink/recent/coverage/best-lex）永不因 cap 牺牲，最后 token 必在内
  - 执行路径未接入 backend：多 chunk prefill 时 indices 指向尚未写入 KV 的位置非法，per-chunk 调整留 T004
- 调试修复：
  - 真实模型 SEGFAULT 根因：`test_attention_sparsity.cpp` `make_selection(256, ...)` 中 `pieces[37*32+5]`（下标 1189）越界写造成 heap corruption 延迟崩溃——加 `needle_pos < n` 守卫
  - 边界 embedding 索引错误（同上"关键实现决策"第 2 条）：曾按输出计数 `embeddings_ith(o)` 读到 chunk 前 o 个 token 的 embedding
  - 测试文本 tokenize 不足 64 token：加长文本
- 测试结果：
  - CTest 无模型：1/1 通过（真实模型用例 SKIP）
  - CTest 真实模型：1/1 通过（50 个用例全 PASS、0 SKIP，含 `draft_scorer_load_and_score` 真实 draft 打分与 8 个 attention sparsity 用例）

### 2026-10-21 会话记录（3）

- 完成任务：T004 分层稀疏 prefill 执行路径
- 修改文件：
  - `include/edge_prefill/attention_sparsity.h` + `src/attention_sparsity.cpp`：新增 `validate_indices`（严格升序/范围/⊆token_indices/含最后 token）与 `chunk_indices`（位置前缀语义）
  - `include/edge_prefill/sparse_prefill.h` + `src/sparse_prefill.cpp`：`Options.enable_attn_sparse`；新增私有 `attn_prefill`——验证→patch 探测（不可用→attn_fallback 降级块级稀疏）→per-chunk 行号 gather→结束恢复 dense；`PrefillModeGuard` 复用
  - `include/edge_prefill/backend.h`：`PrefillStats` 新增 `attn_indices/attn_top_k/attn_fallback` 审计字段
  - `include/edge_prefill/chunk_scheduler.h`：`ChunkStats.attn_rows`（逐 chunk 审计）
  - `include/edge_prefill/benchmark.h` + `src/benchmark.cpp`：`BenchmarkConfig.attn_top_k/attn_top_k_layers`；**repeats/warmup 真正落地**（此前仅解析未参与循环）：warmup 全量 prefill 不计时 → repeats 次 timed prefill → TTFT 取中位数；quality/generation 仅最后一次 timed 后执行（贪心确定性）；E2E 覆盖最后一次 timed 端到端；attention indices 经 AttentionSparsityController 接入 benchmark 执行路径
  - `include/edge_prefill/metrics.h` + `src/metrics.cpp`：`Sample.attn_top_k/repeats`，raw CSV 与 report.md 新增列
  - `src/main.cpp`：bench_command 传递 attn 配置；run_command stderr 汇总输出 attn_top_k/attn_fallback
  - `tests/test_sparse_attn_exec.cpp`（新增，8 用例）
- 关键实现决策：
  - **patch 索引语义修正**：llama.cpp gather 的索引是 **KV cell 行号（decode 顺序枚举）**，非绝对位置——T002 稠密测试 row==position 掩盖了该语义；稀疏选择下必须做 位置→行号 映射（`lower_bound` 于 active_positions），行号继承升序
  - per-chunk 调整：chunk 覆盖行 [begin,end) 执行时仅行 < end 已写入；gather 前缀 = 映射行号 < range.end；空前缀 → 该 chunk top_k=0 禁用 gather（对已写行稠密注意力，因果安全）
  - dense selection 永不启用 gather；非法 indices 在任何模型状态变更前拒绝（不留半写序列）
  - graph 每chunk 重建（indices 变化触发 can_reuse 失配），可接受开销
- 调试修复：
  - 真实执行 SIGABRT（`GGML_ASSERT(i01 < ne01)` in `ggml_get_rows`）：位置值当行号用导致越界——即上述索引语义修正的根因
  - 测试场景退化：512 token/budget 0.5 仅 ~8 selected blocks，top_k_blocks=8 触发 covers-all 返回空 indices → 用例静默 SKIP；改为 top_k_blocks=2 且模型在场时空 indices 直接 CHECK 失败（大声失败）
- 测试结果：
  - CTest 无模型：1/1 通过（SKIP 行为正常）
  - CTest 真实模型：1/1 通过（58 用例全 PASS、0 SKIP，8 个新用例全部真实执行：engages 审计字段/非递减 attn_rows/确定性 argmax、late-only 前缀禁用、disabled 等价 <1e-5、非法拒绝、dense 忽略、benchmark repeats 中位数+CSV 审计列）




## 阶段门控记录

### 2026-10-20 req 门控记录（第 1 轮）

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项
- 修复的问题：G3/G4/G7 的 Minor 级建议已在审查后顺手修复（量化容差、明确并发语义、补充 T010 依赖）
- 审查代理：sdd-gate-reviewer

### 2026-10-20 design 门控记录（第 1 轮）

- 门控结果：FAIL（第 1 轮）
- 失败项：G8 覆盖率目标缺失；G14 6 处 srs 验收标准无追溯用例；G15 接口/CLI 参数边界用例缺失；G2/G9 缓存指标与动态 FFN 跳过设计缺失、scripts/README/docs 未列入变更清单；G12/G13 draft 融合、dense fallback、质量门禁回退、LRU 淘汰等分支用例缺失
- 严重性：Important
- 审查代理：sdd-gate-reviewer

### 2026-10-20 design 门控记录（第 2 轮）

- 门控结果：FAIL（第 2 轮）
- 失败项：G14 §6.6 追溯矩阵错配且缺 FR-6 AC3；G3 C API void 签名与错误返回矛盾；G9 变更清单漏列 selector 相关文件；Minor 包括 top_k=1 边界、tune 不可达原因报告、线程安全判据、draft 融合 oracle、setup.sh 命名、组件图分层、indices_size patch 缺失用例
- 严重性：Important
- 审查代理：sdd-gate-reviewer

### 2026-10-20 design 门控记录（第 3 轮）

- 门控结果：PASS（第 3 轮）
- 审查项数：15 项
- 修复的问题：G2 补充 per-layer 稀疏统计；G3 补充 PrefixCache 签名与枚举定义；G9 补充 metrics.h 与 ffn_skip 衔接；G14 补充一键复现逐 token 断言并修订 srs FR-1 AC2 基线；G15 补充非数字/最大值/NULL 指针用例
- 审查代理：sdd-gate-reviewer
