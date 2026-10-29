# [AR014] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR014 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-29 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 多轮真实数据集 `deep/multiturn`（8 会话 × 5 轮）+ 生成器 | - | passing | 同一文档、逐轮问不同事实 |
| T002 | `SessionState`（log/已准入集合/游标）+ 单测 | T001 | passing | 纯数据结构，无模型可测 |
| T003 | `SparsePrefiller` 追加模式（不清序列、位置续写）+ 单测 | T002 | passing | 失败可降级为重建 |
| T004 | bench 会话模式（index 第 6 列）+ 增量准入 + 两臂 | T003 | passing | `session_reuse=0/1` |
| T005 | 审计列 `session_id`/`reused_tokens`/`appended_tokens` | T004 | passing | CSV 末尾追加 |
| T006 | `ar014_matrix.py` + `ar014_report.py --gate` + 两种注入 | T005 | passing | 质量等价 + 多轮延迟 + 单轮回归 |
| T007 | 全量实验：多轮两臂 + 单轮数据集回归 + CUDA 一致性 | T006 | passing | CPU 8 线程 + CUDA RTX 5000 各两臂抽查 |
| T008 | 文档（README/research/architecture/portfolio/组件详设/repro.sh）+ 归档 + tag + 推 Gitee + 克隆复现 | T007 | passing | 见 ST 记录 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-29 需求阶段（HarnessX sdd-task-requirements）

- 领域知识：`sparse_prefill.cpp` 每次 prefill 前 `reset_sequence()`；bench `enable_prefix_cache=false`；
  现有 `PrefixCache` 只服务 dense 且未接入 bench。
- 量化依据（AR013 已入库实测）：每 token prefill ≈5.5–6 ms，多事实单轮 509 token ↔ 3.1 s；
  即每轮成本与"累积上下文"成正比，而不是与"新增证据"成正比。
- SOTA 对照：vLLM prefix caching / SGLang RadixAttention（前缀复用）、CacheBlend（复用需重算修正）、
  RAGCache（文档级缓存）。本项目采用**严格同前缀 append-only**（无需修正重算），
  因为准入后的证据子集按原文档顺序追加，天然构成前缀关系。
- 结论：AR014 主攻"会话级 KV 复用 + 增量准入"，并以"同前缀同位置 ⇒ 逐例质量必须相同"作为硬约束。

## 阶段门控记录

### 2026-10-29 req 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（背景可量化 / 范围清晰 / FR 可测 / 验收 Given-When-Then / NFR 有阈值 /
  约束明确 / 假设列出 / 任务可独立验证 / 依赖成链）
- 关键约束：质量等价是硬条件（同 token 序列、同位置），任何"复用导致质量漂移"都必须回退，
  不做 CacheBlend 式近似复用。
- 遗留风险：会话内证据只能追加不能重排（重排会使已写入的 KV 位置失效），
  若某轮证据在原文档中位于已入 log 内容之前，则其语义顺序与位置顺序不一致——需在 ST 中用
  逐例质量等价来验证（若不等价则退化为"按位置重排 + 重建会话"）。

### 2026-10-29 design 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：6 项（方案覆盖 FR / 接口完整 / 流程图覆盖分支 / 测试可执行 / 代码边界清晰 / 兼容与回退）
- 关键设计决策：两臂由同一 log 驱动（同 token、同位置）⇒ 质量等价是可断言的硬不变量；
  选择器零改动，增量只做集合差，回归风险限制在 bench 内。
- 遗留风险：会话臂位置 `[0,|log|)` 与重算臂相同，但 KV 写入顺序不同（增量 vs 全量），
  浮点累加顺序可能带来极小差异——ST 以逐例 f1 完全相同为准，若出现漂移则记录并回退。

### 2026-10-30 ST 验收记录

- 单元/集成：无模型 176 例 + 真机 176 例（`EDGE_PREFILL_TEST_MODEL`）全 PASS；
  `-Werror` 构建零告警。
- 矩阵：`deep/multiturn`（8 会话 × 5 轮）+ `explen/len4k`、`deep/multifact8k` 单轮回归，
  两臂同一会话测量。
- 闸门：`ar014_report.py --gate` 90 项全 PASS；注入 `no-reuse` → 5 项 latency FAIL，
  注入 `quality-drift` → quality-agg FAIL。
- 结果：第 2–5 轮 TTFT 0.27×/0.26×/0.20×/0.13×，首轮 0.91×；q 0.678 → 0.693；
  40 轮中 39 轮 f1 逐位相同 + hash 相同，1 轮（`mt-03-t4`）hash 相同但 f1 反转（复用臂更优）。
- 单轮回归：`len4k` 1.000→1.000、`multifact8k` 0.975→0.975，逐位一致。
- CUDA（RTX 5000，全层卸载）：`mt-01` 两轮两臂 q 与 CPU 完全一致（0.727/0.909），log hash 相同，
  第 2 轮 TTFT 4164 → 893 ms。
- 内存：会话一次性定容 112 MiB（4096 格）；已用 KV 与 log 长度一致（30 → 62 MB）。
- 遗留：多轮目前单会话串行，改为在线并发需 KV 池 + LRU（Out of Scope，已写入 research §16.6）。
