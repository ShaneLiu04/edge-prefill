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
| T003 | Draft 模型与分层稀疏选择器集成：真实 GGUF draft 语义打分、块级选择 + attention 级 top-k 候选排序与 `Plan` 新字段 | T002 | pending | 核心创新 |
| T004 | 分层稀疏 prefill 执行路径：位置保持、因果性、dense 回退、审计统计与 benchmark `repeats/warmup` | T003 | pending | 必须真实推理 |
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
