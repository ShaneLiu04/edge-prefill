# [AR011] 详细设计：结构锚点预算与证据预算分离

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| 关联 srs.md | ./srs.md |
| 状态 | Draft |
| 日期 | 2026-10-27 |

## 1. 设计目标与不变量

**目标**：把"预算"从"锚点配额"变成"证据配额"，并让准入粒度和证据保底可配置，
使 `selected_tokens`（端到端 TTFT 的唯一主导项）在质量不降的前提下下降。

**必须守住的不变量（逐 token 等价，回归测试固定）**：

| 编号 | 不变量 | 验证方式 |
|------|--------|----------|
| I1 | `budget=1.0` 且所有稀疏开关关闭 → 与 dense 完全一致 | 既有 `bound_gating_*` / dense 等价用例 |
| I2 | `structure_budget=0 && span_admission=0 && coverage_reserve=0` → 选择结果与 AR010 **逐 token 相同** | 新增 `ar011_off_is_ar010_identical` |
| I3 | 上一条同时要求 `mandatory_blocks`、`block_indices`、`attention_indices` 也相同 | 同上 |
| I4 | `token_indices` 始终严格升序、无重复、必须含 `n_tokens-1` | `selection_indices_invariants`（既有 + 扩展） |

## 2. 现状与问题（实测）

`src/selector.cpp` 目前把锚点与证据混在一个 `std::set<uint32_t> mandatory` 里：

1. `mandatory` 预置 `{0, n_blocks-1}`（sink + recent）；
2. 若有查询词，插入 `best_lex` **及其 ±2 邻块**（5 个块 = 4K 下 160 token）；
3. 再并入 `phrase_anchors`（受 `target/2` 限制）、`entity_anchors`、`stride_anchors`；
4. 用 `anchor_cap = ceil(target * anchor_ratio)` 修剪 stride→entity，**但 phrase 与
   `best_lex±2` 永不修剪**；
5. `mandatory` 先全部计入 `selected_tokens`，剩余预算才给证据填充。

实测后果（srs §1）：4K b=0.08 预算 314 token → 实际 310 token / 10 块，其中 **9 块是
mandatory**；多事实 hard 429 → 422 / 14 块，13 块 mandatory。也就是说
`anchor_ratio=0.35` 这个"锚点上限"在真实的预算区间（b=0.05–0.08）完全失效：
上限被不可修剪的锚点突破，而证据只剩 1 个块。

## 3. 方案总览

```text
                    ┌─────────────────────────────────────────────┐
 prompt blocks ────► │ 1. 结构层 structure (绝对预算, 不可修剪)     │
                    │    sink_blocks ∪ {n_blocks-1}               │
                    ├─────────────────────────────────────────────┤
                    │ 2. 证据锚点层 (可修剪, 受证据预算约束)        │
                    │    best_lex(±r) / phrase / entity / stride  │
                    ├─────────────────────────────────────────────┤
                    │ 3. 覆盖保底层 (每子查询 ≥1 块)               │
                    ├─────────────────────────────────────────────┤
                    │ 4. 证据填充层 (fused 贪心, 证据耗尽即停)      │
                    ├─────────────────────────────────────────────┤
                    │ 5. 粒度层 span: 证据块裁成证据峰附近的 token 段│
                    └─────────────────────────────────────────────┘
```

四层各自记账：`structure_tokens`（第 1 层）、`evidence_tokens`（第 2–4 层，按 token 计），
`span_tokens`（第 5 层裁掉的 token 数），`uncovered_facts`（第 3 层未覆盖的子查询数）。

## 4. 接口变更

### 4.1 `SelectionOptions`（include/edge_prefill/selector.h）

```cpp
// AR011 FR-1: absolute token budget for structural anchors (sink + recent).
// 0 = AR010 semantics (anchors are capped as a fraction of the whole budget).
uint32_t structure_budget = 0;
// AR011 FR-1: radius of the best-lexical anchor window. 0 = AR010 (+-2 blocks).
uint32_t anchor_span_radius = 0;
// AR011 FR-2: force one evidence block per uncovered sub-query.
int coverage_reserve = 0;
// AR011 FR-3: trim evidence blocks to a token span around the evidence peak.
int span_admission = 0;
uint32_t span_min_tokens = 24;
```

默认全 0 ⇒ I2/I3 成立（AR010 逐 token 等价），组合表负责启用。

### 4.2 `Selection` 审计字段（新增，默认 0）

```cpp
uint32_t structure_tokens = 0;
uint32_t evidence_tokens = 0;
uint32_t span_tokens = 0;         // tokens trimmed by span admission
uint32_t uncovered_facts = 0;
```

### 4.3 配置 / CLI / CSV

| 层 | 名称 | 默认 | 说明 |
|----|------|------|------|
| conf/CLI | `structure_budget` / `--structure-budget` | 0 | token 数 |
| conf/CLI | `anchor_span_radius` / `--anchor-span-radius` | 0 | 块半径 |
| conf/CLI | `coverage_reserve` / `--coverage-reserve` | 0 | 0/1 |
| conf/CLI | `span_admission` / `--span-admission` | 0 | 0/1 |
| conf/CLI | `span_min_tokens` / `--span-min-tokens` | 24 | token 数 |
| raw.csv | `structure_tokens,evidence_tokens,span_tokens,uncovered_facts` | | 追加到末尾 |

## 5. 算法

### 5.1 结构层与锚点预算（FR-1）

替换 `selector.cpp` 锚点段（当前 ~L873–L910）：

```text
structure = {0 .. sink_blocks-1} ∪ {n_blocks-1}
struct_tokens = Σ block_tokens(structure)

if structure_budget > 0:
    budget = clamp(structure_budget, struct_tokens, max(struct_tokens, target/2))
    evidence_budget = max(block_size, target - budget)          // 证据可用额度
    anchor_cap = budget + ceil(evidence_budget * anchor_ratio)  // 锚点总上限
    radius = anchor_span_radius > 0 ? min(anchor_span_radius, 2) : 2
else:
    anchor_cap = ceil(target * anchor_ratio)                    // AR010
    evidence_budget = target
    radius = 2
```

`best_lex` 窗口用 `radius`（AR011 组合默认 1，把 160 token 降到 96）。
修剪顺序不变（stride → entity，phrase 与结构层受保护），但比较基准改为
`anchor_tokens(anchors) > anchor_cap`，于是"锚点上限"对 phrase/`best_lex` 也生效：
若仍超限，先放弃 stride/entity，再放弃 phrase，最后放弃 `best_lex` 窗口外的邻块，
只保留 `best_lex` 自身与结构层。

**为什么这样安全**：结构层（sink/recent）永不放弃，所以首 token logits 与注意力 sink
始终存在；`best_lex` 自身也永不放弃，所以单针证据始终在上下文里。降低的只是
"邻域保护"与"位置采样"的冗余。

### 5.2 覆盖保底（FR-2）

在 `covered`/`sub_hit` 建好、`mandatory` 定型之后、填充之前插入：

```text
if coverage_reserve > 0 && !sub_idf.empty():
    // 子查询已按 idf 降序排列；预算够时全部覆盖
    for p in order(sub_idf desc):
        if covered[p] > 0: continue
        if selected_tokens + block_size > target: break
        best = argmax over b of (sub_hit[p][b] ? fused[b] : -inf)
        if best is none: continue
        mandatory.insert(best); mark_covered(best)
        selected_tokens += block_tokens(best)
    uncovered_facts = count(covered[p] == 0)
    if uncovered_facts == 0 && 填充前已满足: stop_reason = "facts-covered"
```

与 AR009 的 facility-location 贪心互补：贪心负责"顺带覆盖"，保底负责"保证覆盖"。

### 5.3 子块准入（FR-3）

`token_indices` 按块展开之后（当前 ~L1050–L1054），若 `span_admission > 0`：

```text
for each selected block b not in structure:
    peak = argmax_{i in block b} per_pos_scores[i]        // 无正分则整块保留
    half = span_min_tokens / 2
    lo = max(block_begin, peak - half); hi = min(block_end, peak + half + 1)
    // 邻块已在选择集内时向邻块扩展，避免把跨块答案切断
    while lo > block_begin && 前一个 token 已在选中集: lo--
    while hi < block_end  && 后一个 token 已在选中集: hi++
    keep [lo, hi)
drop everything else outside structure ∪ kept spans
```

约束：结果仍升序去重；`n_tokens-1` 必须保留（既有代码保证）；`span_tokens` 记录裁剪量。
`block_indices` 保持不变（它描述"哪些块的证据被准入"，供 attention pool 与审计使用），
`token_indices` 才变窄 —— 这正是本 AR 的架构解耦：**准入粒度（token 段）与执行粒度（块）分离**。

### 5.4 停止语义

`stop_reason` 增加两个取值：`facts-covered`（FR-2 提前完成）、`span-trimmed`
（仅当 FR-3 生效且裁剪后低于预算，用于审计口径解释）。原 `budget` /
`evidence-exhausted` / `no-signal` / `dense` 语义不变。

## 6. 数据流影响

| 下游 | 影响 | 处理 |
|------|------|------|
| `SparsePrefiller` | 用 `token_indices` 构造 batch，位置取原 token 下标 | 无需改动（AR007 stride 已支持非连续） |
| `AttentionSparsityController` | 用 `block_indices`/`per_pos_scores` 建 pool | 不变；span 后块内 token 变少，pool 按实际 token 裁剪 |
| `benchmark.cpp` 质量口径 | 依赖 `selection.token_indices` 计算 ppl/overlap | 不变 |
| 组合路由 portfolio | 输入 `selected_tokens`/耗时 | 不变；新增审计列可用于后续路由 |

## 7. 错误处理

| 情况 | 行为 |
|------|------|
| `structure_budget >= target` | 钳制为 `max(struct_tokens, target/2)`，`log_warn` |
| `span_min_tokens < block_size/4` | 钳制到 `block_size/4`（默认 8） |
| 证据块内无正分 token | 整块保留（不裁） |
| `coverage_reserve` 预算不足 | 按 idf 降序覆盖能覆盖的，记 `uncovered_facts` |
| `per_pos_scores` 长度不符 | 整块保留，`log_warn`，不 fail |
| 任何新开关关闭 | 走原代码路径（if/else 分支隔离，不做无用计算） |

## 8. 测试设计

| 类别 | 用例 | 断言 |
|------|------|------|
| 等价 | `ar011_off_is_ar010_identical` | 三开关全关：block/token/mandatory 向量与 AR010 基准一致 |
| FR-1 | `structure_budget_bounds_anchors` | `structure_tokens <= structure_budget`；证据块 ≥1 |
| FR-1 | `structure_budget_zero_keeps_legacy` | 0 时 `anchor_blocks` 与旧值一致 |
| FR-2 | `coverage_reserve_covers_all_facts` | 6 事实、预算充足 → `uncovered_facts==0` |
| FR-2 | `coverage_reserve_reports_shortfall` | 预算不足 → `uncovered_facts>0` 且 ≤ 子查询数 |
| FR-3 | `span_admission_trims_to_peak` | token 数下降、`span_tokens>0`、索引升序 |
| FR-3 | `span_admission_keeps_last_token` | `token_indices.back()==n_tokens-1` |
| 不变量 | `selection_indices_invariants` | 升序/唯一/非空（既有用例扩展） |
| 真机 | tests 现有 158 例 | 全 PASS（无回归） |

## 9. 风险与回退

| 风险 | 影响 | 缓解 |
|------|------|------|
| span 切断跨块答案 | 质量下降 | 峰值居中 + 邻块扩展 + `span_min_tokens`；逐数据集质量闸门 |
| 锚点变少导致首 token 不稳 | 生成退化 | sink/recent 永不裁剪；`q_min` 闸门 |
| 覆盖保底吃光预算 | 单针质量下降 | 仅 `sub_idf.size()>=2` 时生效 |
| 组合无提升 | 白做 | FR-4 闸门：任一数据集质量降 >0.01 → 该开关默认关闭并记录 |

**回退动作**：把 `ar011_matrix` 中对应开关置 0，重跑矩阵，确认回到 AR010 数字后
在 REPORT 写"回退记录"。仓库默认值本来就是 0，因此回退 = 不加 `--span-admission` 等参数。

## 10. 实施顺序

1. T002/T003/T004 三处算法（`selector.cpp`）+ 审计字段；
2. T005 接线（config/CLI/CSV），T007 dump 完整性；
3. T006 单测（先写测试再落实现，I2 等价用例必须最先绿）；
4. T009 矩阵脚本 + 闸门；T010 质量/性能验收；
5. T011 GPU；T012 文档；T013 Review；T014 ST。
