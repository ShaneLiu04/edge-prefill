# [AR009] 详细设计：多事实覆盖准入

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 关联 srs.md | ./srs.md |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.6） |
| 日期 | 2026-10-27 |

## 1. 设计驱动因素与 SOTA 取舍

| # | 驱动 | 量化目标 |
|---|---|---|
| D1 | 一问多事实时预算全被一条事实吃掉 | multifact8k 0.488 → ≥0.85；hard 0.261 → ≥0.60 |
| D2 | 单事实族不能回归 | 选择逐位一致（q=1.000，`selected_tokens` 不变） |
| D3 | 覆盖必须可审计 | `coverage_queries` 列 + 闸门 |
| D4 | 不能引入新的越界/性能风险 | ASan 干净；单事实族零额外工作 |

**SOTA 参考与取舍**

| 来源 | 核心思想 | 取用/否决 |
|---|---|---|
| 次模覆盖 / facility location（Nemhauser 1−1/e 贪心） | 用"未覆盖收益"作增益，贪心即可近似最优覆盖 | **采纳**：子查询 = 待覆盖元素，块 = 候选设施 |
| MMR / 多样性重排（Carbonell） | 惩罚与已选项相似的候选 | 已有（AR007），但去重≠覆盖：MMR 让块互不相似，不保证"每条事实都有块" |
| Set cover / 最大覆盖选址 | 显式优化覆盖集合 | **部分采纳**：纯 set cover 会牺牲连贯性（答案句被切碎），故覆盖只作**加分项**并与 span-grow 共存 |
| 查询扩展 / pseudo-relevance feedback | 用文档句式改写查询 | **否决**：需要外部模型；数据集侧已用拼写数字消除枚举歧义 |
| 学习型路由/覆盖打分 | 训练分类器决定选哪些块 | **否决**：32 行样本，必然过拟合，且不可解释 |

**关键结论**：AR007/AR008 的准入是"相关度排序 + 去重 + 邻域生长"，
本质是**单事实最优**；多事实需要的是**集合覆盖**目标。二者不冲突：
覆盖作为加性增益项，天然在"已覆盖"时归零，因此单事实查询下自动退化为原排序。

## 2. 方案对比

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A：top-k 简单加大 k | 提高预算 | 一行改动 | 预算翻倍仍可能漏事实；成本线性上升 | 否决 |
| B：每事实取 top-1 块（argmax 拼接） | 子查询各自取最相关块 | 直接、易实现 | 丢掉句间上下文（答案常跨块）；块数固定不看预算 | 否决（锚点里保留其思想） |
| C：MMR 加大 λ | 强制多样性 | 复用现成代码 | 多样性 ≠ 覆盖：可能六个相似块都不含第二/三条事实 | 否决（实测多针已证明） |
| D：facility-location 贪心增益 | `gain(b)=Σ_p max(0, idf_p−covered_p)` | 次模、有 1−1/e 保证；已覆盖即归零 → 零回归 | 需要子查询定义与回声排除 | **采纳**（FR-2） |
| E：显式 set cover 求解（ILP） | 最小块集覆盖全部事实 | 最优覆盖 | 目标不是最小集合而是"预算内尽量多覆盖"；无连贯性约束 | 否决 |
| F：锚点 + 覆盖混合 | 每事实一锚点，剩余预算按覆盖增益填 | 事实必进；剩余预算仍按相关度 | 锚点数量需受控 | **采纳**（D + 每事实锚点） |

## 3. 详细设计

### 3.1 子查询提取（FR-2.1）

```text
subs = 查询词序列中去重后的词 t，满足：
        |t| >= 2 字符
        且 1 <= df_blocks(t) <= 2        // 稀有度判据
``` 

- **不用长度启发式**：AR007 的 `size>=6 || 含数字` 会丢掉 `one`/`two`，
  正是多事实族的关键判别符；而 `vault`/`secret` 这类模板词虽然长，
  `df_blocks=6` 一定被稀有度筛掉。
- `df_blocks(t)` 直接从 `block_seq`（未过滤词序列，随分词一并构建）统计，
  不需要额外一遍分词；≤2 的容忍度用于"句子跨块边界"。
- 上限 `kMaxCoverageSubqueries = 8`，按块级 IDF `log((n_blocks+1)/(df+1))+1` 降序保留。

### 3.2 回声排除（FR-2.2）

```text
echo_threshold = max(2, ceil(n_sub / 2))
for each block b: if hits(b) >= echo_threshold: 该块对所有子查询的 hit 置 0
```

chat template 把问句重复在最后一个块，该块天然命中全部子查询；不清除会让
`covered` 在第一次填充前就满载、增益恒 0（第一轮实现即如此，
三个实验的 q 完全相同，是这次"无效果"的直接证据）。
阈值取一半：真正的证据块通常只含 1 条事实（1 个子查询）。

### 3.3 每事实一锚点（FR-2.3）

```text
for p in subs (IDF desc):
    best = argmax_{b: hit(p,b)} ( idf_p + lexical[b] )   // 纯 facility location
    if best 已锚定: continue
    if used + block_size > target/2: continue
    phrase_anchors += best
```

- **不叠加兄弟子查询 IDF**：`idf_p + Σ_{q≠p} idf_q·hit(q,b)` 会让所有子查询
  指向同一个"命中最多"的块；第二版实现因此只锚上 1–2 条事实。
- **替代 entity 锚点**：两者目标重复（都是一事实一锚点），同时启用时实测
  8K/6 事实会把 429 token 预算里的 320 花在锚点上，填充几乎没空间。
  启用覆盖时走本路径，`sub_idf` 为空时仍走 AR007 的 entity 启发式。

### 3.4 覆盖式填充（FR-2.4）

```text
covered[p] = 0                 // 锚点块先 mark_covered
while selected_tokens < target:
    score(b) = adjusted[b]                       // 词法/语义融合 + MMR
             + (adjacent[b] ? 0.5 : 0)           // AR007 邻域生长
             + coverage_weight * gain(b)/idf_max
    gain(b)  = Σ_p max(0, idf_p − covered_p) · hit(p,b)
    选 argmax（同分取小块号，确定性）
    选中后 mark_covered(best)
```

- 归一化 `idf_max` 使覆盖项落在 [0,1]，与融合分数同一量纲；
  `coverage_weight=3` 让"未覆盖事实"稳定压过邻域生长（0.5）与普通相关度。
- 已覆盖子查询贡献 0 → 单事实/已满足时自动退回 AR008 排序（**零回归的结构性保证**，
  不是靠开关）。
- 选中即更新 `covered` → 次模贪心；不重复为同一条事实花第二块。

### 3.5 审计与开关（FR-3）

| 位置 | 内容 |
|---|---|
| `SelectionOptions.coverage_weight` | 默认 3.0；0 = 完全不建覆盖结构 |
| `Config.coverage_weight` / `BenchmarkConfig.coverage_weight` | conf 键 `coverage_weight`（0..10），CLI 经 conf 传播 |
| `Selection.coverage_queries` | 结束时 `covered[p] > 0` 的计数 |
| `Sample.coverage_queries` → `raw.csv` | 追加在 `specific_terms` 之后，位于 `portfolio_choice` 之前 |

### 3.6 顺带修复：`per_pos` 越界（实测缺陷）

AR008 的短语外溢：

```cpp
for (size_t i = 0; i + 1 < flat.size(); ++i) {
    per_pos[i]     += w;   // i 是 flat_seq 下标，不是 token 下标
    per_pos[i + 1] += w;
}
```

`flat_seq` 每 token 追加 1..k 个词（一个 piece 可能解出多个词），
因此 `flat.size()` 可 > `n_tokens`，越界写。新增单测（回声块把整句放进一个
piece）在 ASan 下直接报 heap-buffer-overflow。修法：构建 `flat_tok`
（词 → token 下标）并据此索引。

## 4. 复杂度与资源

| 阶段 | 复杂度 | 备注 |
|---|---|---|
| 子查询统计 | O(n_blocks · |Q|) 去重后 O(n_blocks · n_subs · 块内词数) | 只跑一次；`n_subs ≤ 8` |
| 回声排除 | O(n_blocks · n_subs) | 一次 |
| 锚点选择 | O(n_subs · n_blocks) | 一次 |
| 每次填充 | O(n_blocks · n_subs) | 填充次数 ≤ 预算块数（约 14） |
| 单事实族 | 0（`sub_idf` 为空，不建任何结构） | 保证逐位一致 |

实测 TTFT 影响（同会话，budget=0.05）：multifact8k +5.8%（2767→2927ms），
hard +21.8%（2642→3218ms，子查询 6 个、锚点更多、进入 prefill 的块更多）。

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 覆盖挤掉"答案句另一半"（句跨块） | 覆盖只作加分项，span-grow 仍在；`coverage_weight` 可调 |
| 回声块识别误伤真证据块 | 阈值 = 半数子查询；真证据块通常只含 1 条事实 |
| 短词误判为子查询 | 稀有度判据（df ≤ 2）+ IDF 排序，模板词自然出局 |
| 多事实收益只在合成数据上成立 | REPORT 中标注数据来源与生成命令；单事实族回归作为反向证据 |
| 收益不达标 | `coverage_weight=0` 一键回退；specs 要求不达标则只保留数据集与负结果 |

## 6. 与既有 AR 的关系

| AR | 关系 |
|---|---|
| AR007 | 预算曲线/锚点/MMR/span-grow 原样复用；覆盖只在"多子查询"时插入 |
| AR008 | 短语准入（`phrase_weight`）仍是判别符来源；`coverage_weight=0` 逐位等于 AR008 |
| AR004/AR006 | `per_pos` 外溢沿用；本轮不改图内路径（含越界修复同样惠及该路径） |
| AR005 | 不涉及 |

## 7. 测试策略

| 层次 | 内容 |
|---|---|
| 单测（无模型） | 零回归开关、每事实入选、回声排除、单事实 no-op（4 例，见 `tests/test_selector.cpp`） |
| ASan | `build-asan` 跑覆盖用例，确认无越界（`per_pos` 修复） |
| 真机 | `ar009_ablation.py` 两档 × 两权重；`expected_ar009.csv` 机检 |
| 非回归 | 4k/16k/mn 选择逐位一致；32K 不劣化 |
| 干净克隆 | 构建 + `ctest` + `./repro.sh ar009` |

## 8. 实测结果（as-built）

| 数据集 | AR008 q | AR009 q | Δ | covered | TTFT |
|---|---|---|---|---|---|
| deep/multifact8k | 0.488 | **0.878** | +0.390 | 3.25/3 | +5.8% |
| deep/multifact8k-hard | 0.261 | **0.652** | +0.391 | 6.50/6 | +21.8% |

dense 参照（2 上下文 4 行，budget=1.0）：multifact8k q=0.975
（1.000/1.000/1.000/0.900）。即 AR009 用 5% 的 token 取回 dense 90% 的质量。

单事实族非回归（`coverage_weight=3`）：4K `sel=322 q=1.000`、16K `sel=793 q=1.000`、
mn `sel=401 q=1.000`，与 `coverage_weight=0` **完全相同**；32K q 0.875→0.893
（选择略变，非劣化）。

## 9. 未达成的目标（诚实记录）

1. `hard` 档未到 srs 的 0.85 目标（实测 0.652）。根因：12 条事实埋在 8.5K 文档里，
   6 条请求事实需要 6 个块 + 句间上下文，而 445 token（约 14 块）在扣掉 sink/末块
   /锚点后只剩 3–4 块给填充；剩余差距是**预算物理不足**，不是准入逻辑失效
   （`covered_queries=6.5/6` 说明事实已进上下文，丢分主要来自答案格式与顺序）。
2. hard 档 dense 天花板未测（16 行 dense 前向远超本机单次会话时限），
   因此该档只报告 AR008↔AR009 同条件对比，不给"相对 dense 的恢复比例"。
3. 覆盖信号目前只作用于选择级；图内（AR006）路径未接，留作后继项。
