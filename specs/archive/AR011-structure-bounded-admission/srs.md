# [AR011] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| AR 主题 | 结构锚点预算与证据预算分离（structure-bounded admission） |
| 关联 SR | 无（沿用 AR001 的端侧 prefill 目标） |
| 日期 | 2026-10-27 |
| 状态 | Draft |

## 1. 背景与目标

AR010 之后，准入（admission）已经做到"零证据块不进入上下文"，端到端 TTFT 近似等于
`selected_tokens / 预填吞吐`（1.5B q4_k_m、16 线程实测 ≈ 155 token/s）。也就是说
**剩余的全部优化空间都在"准入 token 数"上**。

本 AR 起点是两次实测审计（真实 GGUF、无 mock）：

1. **结构锚点吃掉了预算**：4K NIAH（b=0.08，预算 314 token）实际准入 310 token / 10 个
   32-token 块，其中 **9 个是 mandatory 锚点**（sink、recent、anchor_ratio=0.35 的结构锚点），
   真正的证据块只有 1 个；多事实 hard（b=0.05，预算 429）准入 422 / 14 块，**13 个是 mandatory**。
   即"预算"这一概念已经被锚点占满，"证据预算"名存实亡。
   `anchor_ratio` 按**总预算**比例分配，而不是按固定结构成本分配，导致预算越小、
   锚点占比越高（b 越小越糟），这与"预算越紧越应该把 token 留给证据"的直觉相反。
2. **语义（embedding）信号不值得**：在 mf/mfh/multineedle/len4k/len8k 上对比
   `score_source=lexical` 与 `embedding`：多事实 hard 0.334→0.337、easy 0.576→0.573、
   multineedle 与 len 系列 1.000→1.000，**质量无变化**，而编码扫描成本
   4K +996 ms、8K +1958 ms、8K-hard +2192 ms（TTFT 的 45–85%）。
   → AR005 的 embedding 选路在当前数据集上只买成本不买质量，应当保持默认关闭，
   并以此作为 AR011 的"负结果"记录，避免后继重复投入。

**目标**：在不降低任何数据集质量的前提下，让准入 token 数显著下降；并让多事实场景
的每个被问事实都有保底证据（收紧 `q_min`）。

## 2. 需求范围

**In Scope：**
- `src/selector.cpp` 的锚点/预算记账与填充策略；`Selection` 新增审计字段
- `include/edge_prefill/{selector.h,config.h,benchmark.h}`、`src/{config.cpp,main.cpp,metrics.cpp}`
  的开关与审计列（`--structure-budget`、`--span-admission`、`--coverage-reserve`）
- 单测 + 组合矩阵脚本 + 报告闸门 + `repro.sh ar011`
- 文档：README、`docs/research.md` §13、`docs/architecture.md`、`docs/server-repro.md`、
  组件详设 v0.9

**Out of Scope：**
- 语义/embedding 选路（本 AR 实测无收益，不投入）
- llama.cpp 侧 KV 复用 / CacheBlend 类跨请求复用（工程量大，另立 AR）
- 解码阶段优化（本项目是 prefill 项目）
- 任何形式的 mock 或合成质量分

## 3. 功能需求

### 3.1 FR-1 结构锚点与证据预算分离

**描述：** 结构锚点（sink、recent）使用**绝对 token 预算**，证据/覆盖填充使用剩余预算。

**触发条件：** 任意非 dense 选择（`budget < 1.0`）。

**期望行为：**
- 新增 `structure_budget`（token 数，默认 0 = 沿用 AR010 行为）：
  - `>0` 时，sink（首块）与 recent（尾块）按绝对量计入，且 `anchor_ratio` 只作用于
    `budget - structure_budget`；
  - `=0` 时保持 AR010 语义（锚点按总预算比例）。
- 审计：`Selection.structure_tokens`、`Selection.evidence_tokens`，
  以及 raw.csv 列 `structure_tokens,evidence_tokens`。

**异常处理：** `structure_budget >= budget_tokens` 时钳制为 `budget/2` 并 `log_warn`，
不得返回空上下文；`budget=1.0` 时该开关无效（dense 等价）。

**验收标准：**
- Given 4K NIAH、b=0.08、structure_budget=64，When 选择，Then
  `structure_tokens<=64` 且 `evidence_tokens>0`；
- Given `structure_budget=0`，Then 选择结果与 AR010 **逐 token 一致**（回归测试）。

### 3.2 FR-2 覆盖保底（per-fact reservation）

**描述：** 多个被问事实（`coverage_queries>=2`）时，先保证每个事实至少一个证据块，
再跑常规填充。

**触发条件：** `coverage_reserve=1`（**as-built 默认 0**：实测无稳定收益，见 §7；仅在 coverage 激活时生效）。

**期望行为：**
- 对每个子查询，`argmax(idf_p + lexical)` 的块若未被选中，则强制准入；
- 若预算不足以覆盖全部事实，按 `idf` 降序覆盖尽可能多的事实，并记录
  `Selection.uncovered_facts`（审计列 `uncovered_facts`）；
- 全部覆盖后停止（`stop_reason=facts-covered`）。

**异常处理：** 子查询数 > 8 时按 idf 取前 8 个（防止预算被碎片事实吃光），并记录。

**验收标准：**
- Given 6 事实 hard 数据、b=0.05，When 选择，Then `uncovered_facts==0`（预算足够时）；
- Given 预算不足，Then `uncovered_facts>0` 且与 `coverage_queries` 一致可解释。

### 3.3 FR-3 子块（span）准入

**描述：** 证据块不再整块准入，而按"证据峰附近的连续 token 段"准入。

**触发条件：** `span_admission=1`（**as-built 默认 0**，由组合表启用；`0` 时逐字节等价 AR010）。

**期望行为：**
- 对每个入选证据块，取块内 `per_pos_scores` 峰值，向两侧扩展到
  `span_min_tokens`（默认 24）或到块边界，可跨块合并（与 AR007 span-grow 一致）；
- sink/recent 锚点仍按整块（保证 chat 模板完整）；
- 审计：`Selection.span_tokens`（被裁掉的 token 数）。

**异常处理：** 若某块内无正分 token，则整块准入（不裁）；span 结果必须升序唯一。

**验收标准：**
- Given 单针 4K，When span 准入，Then 质量不变（q=1.000）且 `selected_tokens` 下降 ≥15%；
- Given `span_admission=0`，Then 与 AR010 逐 token 一致。

### 3.4 FR-4 组合再择优与回退闸门

**描述：** 必须证明新机制让**组合**变好，否则回退。

**期望行为：**
- `ar011_matrix.py`：7 数据集 × {AR010 基线, AR011 全开, 仅 FR-1, 仅 FR-3, gate-p16} 五臂；
- `ar011_report.py`：沿用 AR010 的表推导规则（质量地板下最快者）+ Pareto 闸门 +
  `--inject` 自测；
- 任一数据集质量下降 > 0.01 → 该数据集对应开关默认关闭，并写入 REPORT 的"回退记录"。

**验收标准：**
- Given 7 数据集矩阵，Then 组合表 7/7 Pareto 不劣于最佳固定臂；
- Given 注入劣化，Then 闸门 FAIL。

### 3.5 FR-5 审计完整性

**描述：** `--dump-selection` 的 JSON 必须包含 AR008–AR011 的全部审计字段。

**验收标准：** Given 任一非 dense 选择 dump，Then 含
`anchor_blocks,phrase_blocks,specific_terms,coverage_queries,evidence_blocks,stop_reason,
structure_tokens,evidence_tokens,span_tokens,uncovered_facts`。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4K/8K/16K/32K TTFT | 相对 AR010 下降 ≥15%（质量不降） |
| 质量 | 所有规范数据集 q_mean | 不低于 AR010（容差 0.01） |
| 质量 | 多事实 hard `q_min` | 相对 AR010 提升（≥0.087 → 目标 ≥0.15） |
| 兼容 | `budget=1.0` / 开关全关 | 与 dense 逐字节一致 |
| 兼容 | AR010 语义 | 三个新开关全关时逐 token 等价 |
| 可复现 | 干净克隆 | `./repro.sh ar011` 一条命令跑通 |

## 5. 约束与假设

**约束：**
- 无 mock：全部指标来自真实 GGUF 前向（qwen2.5-1.5b q4_k_m + bge-small 仅在对比实验中使用）
- `include/` 公共接口只增不改（新字段默认值保持旧行为）
- 不接受"质量换速度"的隐性折衷：任何质量下降必须显式记录并回退

**假设：**
- 端侧场景：单机 CPU（16 线程）/ 单 GPU，模型 ≤2B，上下文 4K–32K
- 结构锚点（sink/recent）对质量是必要的（AR001/AR004 已验证），本 AR 只调整其**预算记账**方式

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 结构锚点 | 为保持上下文可用性而强制准入的块：sink（首块）、recent（尾块/问题所在）、结构锚点（anchor_ratio） |
| 证据块 | 归一化词法/短语/覆盖信号 > 0 的块 |
| 证据预算 | `budget_tokens - structure_tokens`，真正用于证据的 token 数 |
| 覆盖保底 | 每个子查询（稀有查询词）至少一个证据块的硬约束 |
| span 准入 | 以 token 段（而非 32-token 块）为单位的准入粒度 |

## 7. As-built 偏离记录

| 需求 | 偏离 | 原因 |
|------|------|------|
| FR-2 默认 1 | 实现为默认 0 | 预算扫描实测无稳定收益（b=0.03 +0.019、b=0.02 −0.007、b=0.04 0），按"不提升组合即回退"规则关闭；`uncovered_facts` 审计保留 |
| FR-3 默认 1 | 实现为默认 0 + 单事实硬门控 | 多事实裁剪实测 easy −0.036 / hard −0.225 / `q_min`→0；默认关闭避免影响未测场景，组合表对单事实启用 |
| FR-1 无条件生效 | 仅当 `anchor_tokens > target/2` 生效 | 16K 无条件启用会把 280 token 变成 320（质量不变），属于负向改动 |
