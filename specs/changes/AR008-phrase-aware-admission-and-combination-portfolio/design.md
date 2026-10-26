# [AR008] 详细设计：短语感知准入 + 组合路由

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 关联 srs.md | ./srs.md |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.5） |
| 日期 | 2026-10-26 |

## 1. 设计驱动因素与 SOTA 取舍

**驱动因素（按优先级）**

| # | 驱动 | 量化目标 |
|---|---|---|
| D1 | 判别符丢失导致多针检索失效 | mn q 0.33 → ≥0.80（b=0.05） |
| D2 | 没有"按 prompt 选组合"的机制 | 路由后每个数据集 Pareto 不劣于最佳固定组合 |
| D3 | 组合变差无人阻止 | 机器无关闸门，注入劣化必须 FAIL |
| D4 | 不能引入新的图内风险 | 零 patch 变更；`phrase_weight=0` 与 AR007 逐位一致 |

**SOTA 参考与取舍**

| 来源 | 核心思想 | 本 AR 取用/否决 |
|---|---|---|
| BM25 / 短语检索（位置 bigram、邻近度） | 词项顺序与邻近携带独立信息，短语命中比单词命中更可信 | **采纳**：块级短语 IDF 作独立信号 + 锚点 + `per_pos` |
| FrugalGPT / RouteLLM 级联路由 | 用小模型/便宜路径打分决定是否升级到大模型/贵路径 | **采纳思想**：实测表 + profile 特征做组合路由（不做概率模型，样本不足） |
| ColBERT / 多向量检索 | 细粒度晚期交互提升判别力 | **否决**：需要额外编码器，违背"无外部依赖、无嵌入关键路径" |
| SnapKV / H2O / Scissorhands | decode 期按累积注意力淘汰 KV | **否决**：目标是 prefill 准入；decode 期优化不在组件范围 |
| Submodular coverage / facility location | 多事实覆盖的 1-1/e 贪心 | **暂缓**：仓库缺少"一问多事实"数据集，无法诚实评估（记入 srs §5 之外的后继项） |
| LLMLingua 粗到细压缩 | 先粗筛再精筛 | **部分采纳**：短语优先级 = 粗筛，词法/锚点填充 = 精筛 |

**关键结论**：AR007 把 mn 失败归因为"块粒度覆盖不足"，本轮用数据推翻了该假设
（`mn-02.txt` 中 `vault 5` 唯一出现，代码紧随；三条针只差一个数字，而数字被过滤掉）。
因此 AR008 的第一个功能是**修 IR 信号**，而不是继续加预算或加锚点。

## 2. 方案对比

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A：把单字符数字直接加入查询词 | 放宽过滤条件 | 一行改动 | "1" 在文档里到处出现（df 极高），IDF 近似 0；仍无法区分 "vault 1" 与 "1" | 否决 |
| B：位置短语（bigram）+ 短判别符保留 | 查询相邻词对作独立词项，块内按顺序匹配 | 直接编码"vault 5"这一判别单元；零依赖；无命中时零影响 | bigram 数量膨胀需要过滤（只保留有判别力的词对） | **采纳**（FR-1） |
| C：窗口邻近度（查询词共现窗口内加权） | 统计块内查询词的最小覆盖窗口 | 更鲁棒（允许插入词） | O(n_tokens×|Q|)，阈值难定，且与短语高度重叠 | 暂缓（后继项，B 已解决 D1） |
| D：学习型路由器 | 训练分类器选组合 | 潜在最优 | 5×5=25 个样本，必然过拟合；不可解释 | 否决 |
| E：实测表 + 规则路由 + Pareto 闸门 | 由测量结果生成表，规则匹配；闸门校验 | 可解释、可复现、可回退；直接满足"组合最优 + 不许变差" | 表随模型/数据变化需重测（用指纹 + 生成脚本约束） | **采纳**（FR-2/3） |

## 3. 详细设计

### 3.1 短语感知准入（FR-1）

**数据结构**（`selector.cpp` 内部 + `Selection` 审计）

```cpp
// 块内顺序词序列（未过滤，含单字符数字），用于 bigram
std::vector<std::vector<std::string>> block_seq(n_blocks);
// 查询短语（"a b"），只保留有判别力的词对
std::vector<std::string> query_phrases;
// 块级短语 IDF 分
std::vector<double> phrase(n_blocks, 0.0);
```

新增 `BlockSignal::phrase`（0..1，min-max 归一化后）与
`Selection::phrase_blocks`（短语命中的块数，审计）。

**算法**

1. 收集块内词序列：token 顺序 → `terms_from_text(normalize_piece(piece))` 全部词
   （**包含**单字符数字），追加到 `block_seq[b]`；
2. 查询短语：对问句词序列（同样不过滤）取相邻对 `(a,b)`，保留条件：
   `is_discriminative(a) || is_discriminative(b)`，其中
   `is_discriminative(t) = 含数字 || 长度≥5 || 首字节≥0x80`；去重、排序（确定性）；
3. 短语 IDF：`df_phrase = #{b : "a b" ∈ bigrams(block_seq[b])}`，
   `idf = log((n_blocks+1)/(df_phrase+1)) + 1`；
   `phrase_raw[b] = Σ_{p ∈ query_phrases, p ∈ bigrams(b)} idf(p)`，
   再除以 `sqrt(block_tokens)`（与词法同构，避免长块占优）；
4. `minmax(&phrase)` → `signals.phrase[b]`；
5. 融合：`fused[b] += opts.phrase_weight * phrase[b]`（默认 0.5；`0` 表示完全关闭，
   此时不构建短语、逐位等于 AR007）；
6. 锚点：`phrase_best`（`phrase_raw` 最大且 >0 的块）进入 mandatory，
   优先级 **sink > recent > phrase > quote(entity) > stride**；裁剪顺序相应改为
   stride → entity → phrase（永不裁剪 sink/末块/短语块，见 §5）；
7. `per_pos_scores`：对每对命中短语的两个 token 位置各加 `idf(p)`，
   使 AR004 qg / AR006 门控 2D 信号同样短语感知（共享同一信号，组合时一致）。

**复杂度**：短语集合 ≤ 2×|Q|；块 bigram 构建 O(n_tokens)；匹配 O(n_blocks×|P|)
（用 `unordered_set<string>` 缓存块 bigram，查询侧 O(1)）。32K/970 块实测 <5ms。

**零回归开关**：`phrase_weight=0` → 跳过步骤 2-5，`fused` 与锚点集合与 AR007 完全一致
（单测用同一输入比较 `block_indices`、`anchor_blocks`、`selected_tokens` 逐位相等）。

### 3.2 组合路由（FR-2）

**profile**（`PromptProfile`，host 侧，无模型）

| 特征 | 计算 | 用途 |
|---|---|---|
| `n_tokens` | 提示长度 | 长度桶：`lt4k` / `8k` / `16k` / `32k+` |
| `n_specific` | `is_discriminative` 的查询词数（含短语中的数字） | 判别力：0-1 → `low`，≥2 → `high` |
| `phrase_hits` | `phrase_raw > 0` 的块数 | 短语是否可定位（0 = 无判别符命中） |
| `concentration` | `top1_lexical / Σ lexical` | 词法是否唯一指向（>0.5 = 单点） |

**组合表**（`benchmarks/matrix/portfolio/portfolio.txt`）

```
# 由 portfolio_matrix.py 从实测 CSV 生成，勿手改
model_sha256 = <sha>
generated = 2026-10-26T…
default = selection
low:lt4k  = selection
low:8k    = selection
high:8k   = selection
high:16k  = selection
high:32k  = selection
flat:any  = gate-p24-local32   # 无判别符命中且词法不集中 → 图内门控兜底
```

行键 `<discriminability>:<length_bucket>`，`flat` 表示 `phrase_hits==0 且 concentration<0.3`。
加载器与 `BudgetCurve` 同风格（行式严格解析、指纹校验、mtime 缓存）。

**路由决策顺序**（不改变用户显式意图）

1. conf/CLI 显式 `qg_bound` / `budget` / `budget_curve` → 用户意图优先，路由只记录不建议；
2. 否则按 profile 查表 → `combo`；
3. combo = `gate-*` → 置 `enable_qg_bound=true` + `qg_sel_pages/qg_local_rows`；
   `selection` → 现状；
4. 表缺失/指纹不符/无匹配 → `selection` + reason `portfolio-fallback`。

**审计**：`Sample.portfolio_choice`（`selection`/`gate-p16`/`gate-p24-local32`/`dense`/
`none`）、`Sample.portfolio_reason`（`table`/`fallback`/`user-override`）。

### 3.3 组合回归闸门（FR-3）

`check_portfolio.py <matrix.csv> <expected_portfolio.csv>`：

1. 按数据集分组，取每个组合的 (q_mean, q_min, ttft_median)；
2. `best_fixed` = argmax q_mean（同质量取 ttft 最小）；
3. 对 `routed` 行：`q_mean ≥ best_fixed.q_mean - 0.01` 且 `ttft ≤ best_fixed.ttft × 1.05`；
4. 任一违规 → 打印违规表并非零退出；
5. `--inject <dataset>=<combo>` 用于自测（把 routed 替换成指定组合，验证闸门会 FAIL）。

### 3.4 组合矩阵实测（FR-4）

`portfolio_matrix.py`：对每个 (数据集 × 组合) 调 bench，产出
`matrix.csv`（列：dataset,combo,q_mean,q_min,ttft_median,speedup,selected,portfolio_choice）
+ `matrix.md` + `REPORT.md`（含 PHA 前后 mn 对比、淘汰理由）。
组合定义放 `matrix/portfolio/conf/*.conf`，全部复用现有能跑的配置，不新造开关语义。

## 4. 关键流程

```
bench: 读 conf → [profile] → [portfolio 查表] → Plan(combo,budget,curve)
      → BlockSelector(phrase_weight, anchors: sink>recent>phrase>entity>stride)
      → (combo==gate-* ? SparsePrefiller(qg_bound) : 选择级稀疏)
      → Sample(+portfolio_choice, phrase_blocks)
```

## 5. 错误处理与降级

| 场景 | 行为 |
|---|---|
| 组合表缺失/解析失败/指纹不符 | 记 `portfolio-fallback` → `selection`，不失败 |
| profile 无匹配行 | 用 `default` 行；无 `default` → `selection` |
| 短语集合为空（无判别力词对） | 跳过短语信号（等价 AR007），`phrase_blocks=0` |
| `phrase_weight` 越界 | 配置层 `InvalidArgument`（0..2） |
| 锚点超预算 | 裁剪顺序 stride→entity→phrase；sink/末块/短语块保留（短语块是唯一判别证据） |
| `per_pos` 短语加成溢出 | 与词法同尺度（IDF），无额外缩放，保持非负有限 |

## 6. 测试设计

| ID | 类型 | 内容 |
|---|---|---|
| UT-1 | 无模型 | 短语提取：`vault 5` 保留；`what is` 丢弃；CJK 相邻保留；确定性排序 |
| UT-2 | 无模型 | 判别符：单字符数字不被过滤；含数字词对必保留 |
| UT-3 | 无模型 | `phrase_weight=0` 与 AR007 选择逐位一致（同一输入） |
| UT-4 | 无模型 | 短语块进锚点且优先于 entity/stride 被保留（构造超预算输入） |
| UT-5 | 无模型 | 组合表解析严格性 + 指纹不符回退 + profile 分桶边界 |
| UT-6 | 无模型 | 路由降级：无表 → `selection` + `portfolio-fallback` |
| UT-7 | 真机 | mn b=0.05：q≥0.8（AR007 0.33） |
| UT-8 | 真机 | 单针 8K/16K：q=1.00，TTFT ≤ AR007 ×1.05 |
| UT-9 | 真机 | 4K 复测：q≥0.95 |
| IT-1 | 真机 | 组合矩阵 25 格 + 闸门 PASS；注入劣化 FAIL |
| IT-2 | 真机 | dense 等价：`budget=1.0` 全关逐位一致 |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 短语信号在通用文本上过强（bigram 噪声） | 单针质量下降 | 只保留有判别力词对 + 权重 0.5 实测；UT-8/IT-1 闸门卡住 |
| 短语块占用锚点预算 | 高预算下稀释证据（AR007 教训） | 短语块数量受 `anchor_ratio` 约束，裁剪顺序表明其优先级高于 entity/stride |
| 路由表过拟合本机 | 换机器选错组合 | 表中只编码"结构性"规律（长度桶/判别力），闸门按数据集而非绝对延迟判定 |
| 表随模型失效 | 误用组合 | `model_sha256` 指纹 + 不符即回退 `selection` |
| 组合叠加导致相互抵消（AR004 分支 + 门控 + 短语） | 净收益下降 | 组合矩阵实测；闸门以"最佳固定组合"为基线，任一新组合不能反超即不入选 |

## 8. 组件详设影响

- `Selection` 增 `phrase_blocks`；`BlockSignal` 增 `phrase`；
- `Sample` 增 `portfolio_choice`/`portfolio_reason`/`phrase_blocks`（raw.csv 末尾追加）；
- `SelectionOptions` 增 `phrase_weight`；`Config` 增 `phrase_weight`/`portfolio_table`；
- planner 增组合路由步骤（`Plan.combo` 字段与 `portfolio:<combo>` reason）；
- 组件详设升级 v0.6（变更记录 + 数据结构表 + 分层图注记）。

## 9. 交付物

- 代码：selector（短语/锚点/审计）、profile+router、config、benchmark/main 接线、单测；
- 数据：`benchmarks/results/ar008-portfolio/{matrix.csv,matrix.md,REPORT.md,<cell>/raw.csv}`、
  `benchmarks/matrix/portfolio/{portfolio.txt,conf/*,expected_portfolio.csv}`；
- 脚本：`portfolio_matrix.py`、`check_portfolio.py`；`repro.sh portfolio`；
- 文档：srs/tasks/design（本目录）、README、research §10、architecture、server-repro §8、
  组件详设 v0.6；
- 交付：tag `ar008-v1.0` + Gitee 同步。

## 10. As-built 偏差（2026-10-26）

| 设计点 | 设计稿 | 实际落地 | 依据 |
|---|---|---|---|
| 短语锚点 | "短语命中的块进入锚点，优先级 sink>recent>phrase>entity>stride" | 锚定**最高 IDF 短语**命中的全部块（预算一半封顶），并在两处裁剪中保护 | 以"块短语总分"取 arg-max 会被模板短语（`for vault`，三个同级事实共享）带走；实测表现为答错兄弟事实（q=0.33→0.66 卡住），改成最高 IDF 短语后 1.00 |
| 短语跨块 | 未提及 | 增加**跨块 bigram**（块 b 的末词 + 块 b+1 的首词），同时记到两个块 | needle 句子起点常落在块尾，逐块 bigram 完全看不到判别短语；这是多针最后一个失败例的根因 |
| 组合闸门输入 | `expected_portfolio.csv` | `matrix.csv` + `portfolio.json`（由 `portfolio_matrix.py` 同一次实测生成） | expected 表等于把本轮结果抄一遍，闸门会永远 PASS；用"同会话最佳固定组合"作参照才是真正的 Pareto 检查 |
| 路由 combo 集 | selection/gate-p16/gate-p24-local32/dense | 相同，但 `selection` 臂固定走 AR007 曲线 + AR008 短语 | 曲线是 AR007 已验证的最优点，避免在 AR008 里再引入预算自由度 |
| 审计列 | `phrase_blocks`、`portfolio_choice`、`portfolio_reason` | 另加 `specific_terms`（组合 profile 的判别词数） | 路由表键 `<discriminability>:<bucket>` 需要可复核的输入；只报结果不报依据无法审计 |
| gate 臂预算 | 未固定 | `budget = 0.25`（与 AR006 canonical 一致） | 门控降低的是 attention 行数，不是准入 token 数；用 0.25 保持与 AR006 数字可比 |
| 回归发现 | — | bench 网格里 `qg_*` 属于 `BenchmarkConfig` 而非 `Config`，路由初版从 `run_cfg` 取导致门控静默不生效（qg_source 为空、质量等于纯选择级） | 由矩阵首轮 `qg_source=''` 暴露；修复后 4K gate q=1.00/18.5s、8K gate q=0.88/27.5s 与 AR007 观测一致 |
