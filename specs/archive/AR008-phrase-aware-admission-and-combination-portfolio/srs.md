# [AR008] 需求规格说明书：短语感知准入 + 组合路由

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 主题 | phrase-aware-admission-and-combination-portfolio |
| 日期 | 2026-10-26 |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.5） |
| 前序 | AR001–AR007（均已归档） |
| 状态 | 需求已定稿（用户授权自动化流程，自检替代交互确认） |

## 1. 背景与问题（来自实测，不是推测）

AR007 的验收报告里有一条刺眼的失败：多针数据集 `deep/multineedle` 在**选择级稀疏**下
只有 q=0.33（b=0.05）、0.39（b=0.25），而 AR006 图内门控能到 0.83。AR007 当时把根因
归为"选择级块粒度覆盖不了多事实邻域"。**本轮重新做根因分析，结论不同**：

```
问句：What is the secret passcode for vault 5?
文档：... The secret passcode for vault 5 is CRIMSON-LYNX-7624. ...
      （文档里 vault 4/5/6 各出现一次，只有数字不同）
```

文档中区分三条针的**唯一判别符是单个数字**，而 `selector.cpp` 的查询词过滤是
`term.size() >= 2 || CJK`——单个数字被**静默丢弃**。于是问句词退化为
{secret, passcode, vault}，三个块得分完全相同，选中哪个纯看块序号，正确率≈1/3，
与实测 q=0.33 完全吻合。多针"质量差"不是粒度问题，是**判别符被丢掉的 IR 缺陷**。

这不是多针专属问题：`chapter 7`、`case 42`、`step 3`、`v2`、`Q4` 这类
"短数字/短代码判别符"在任何文档检索里都是最强信号，当前实现一律丢弃。

第二个问题是**组合**：仓库现在有 6 条可用的执行路径（选择级稀疏、AR006 图内门控 p16/p24、
AR005 qg-2d、AR004 压缩分支、dense），同一份 prompt 用不同组合的结果差异很大
（32K：选择 7s/q=0.89，门控 31s/q=1.00）。目前没有任何机制按 prompt 特征选组合，
用户必须手工试。AR007 的曲线只解决"预算多少"，不解决"用哪条路径"。

## 2. 目标

1. **修掉判别符丢失**：让短语（相邻词对）与短数字/代码成为一等准入信号，
   使"vault 5"这种只有一个数字差异的检索在低预算下也能命中；
2. **组合可选且可审计**：给 planner 一个**实测驱动的组合路由**，按 prompt profile
   在已测量过的组合里选；选择结果写入 `portfolio_choice` 审计列；
3. **组合不许变差**：提供机器无关的**组合回归闸门**——路由后的每个数据集必须
   Pareto 不劣于"最佳固定组合"（质量不低于其 -0.01，TTFT 不高于其 1.05 倍），
   否则闸门失败，回退到上一版最优组合表。

## 3. 功能需求

### 3.1 FR-1 短语感知准入（PHA）

**描述**：在词法准入信号中引入**位置相邻短语**与**短判别符**，解决"只有数字不同"的检索。

**期望行为**：
1. 查询短语 = 问句词序列中相邻两词构成的 bigram，保留条件是其中至少一个词"有判别力"
   （纯数字、含数字、长度 ≥5、或含 CJK 字节）；单字符数字/短代码**不再被丢弃**；
2. 块短语集合 = 块内词序列的相邻 bigram（块内顺序来自 token 顺序），
   用块级 IDF 计算短语权重 `log((n_blocks+1)/(df_phrase+1))+1`；
3. 短语得分作为独立信号 `phrase[b]` 进入融合分数（默认权重 `phrase_weight=0.5`，
   0 表示关闭，等价于 AR007 行为）；
4. 短语命中的块同时进入**锚点**（优先级仅次于 sink，高于 stride/entity）与
   `per_pos_scores`（供 AR004 qg / AR006 门控使用），使下游路径共享同一信号；
5. 无短语命中（df=0）时行为与 AR007 **逐位一致**（零回归）。

**验收标准**：
- Given `deep/multineedle`（6 例，唯一判别符是 vault 编号），When budget=0.05 + PHA，
  Then q ≥ 0.8（AR007 为 0.33）；
- Given explen 8K/16K，When PHA 开启，Then q 保持 1.00，TTFT 变化 ≤5%；
- Given 同一输入，When 跑两次，Then 选择结果逐位一致；
- Given `phrase_weight=0`，When 跑，Then 与 AR007 选择结果逐位一致（单测）。

### 3.2 FR-2 组合路由（Portfolio Router）

**描述**：按 prompt profile 从"已实测过的组合"中选择执行组合，而不是固定一条路径。

**期望行为**：
- profile 特征（全部来自 token/词法，无模型调用）：`n_tokens`、`n_specific_terms`（判别力词数）、
  `phrase_hits`（短语命中块数）、`lexical_concentration`（top-1 词法分/总分）；
- 组合表 `benchmarks/matrix/portfolio/portfolio.txt`（行式 `key = value`，带
  `model_sha256` + `generated` 指纹），行格式：`<arity>:<length_bucket> = <combo>`，
  combo ∈ {`selection`, `gate-p16`, `gate-p24-local32`, `dense`}；
- 路由结果写入 `Sample.portfolio_choice`（与 `budget_source` 同级审计），
  `Plan.reasons` 增 `portfolio:<combo>`；
- 表缺失/指纹不符/无匹配行 → 回退 `selection`（当前最优组合），记
  `portfolio-fallback`，不失败；
- 显式 conf/CLI 指定 `qg_bound`/`budget` 时**用户意图优先**，路由不覆盖。

**验收标准**：
- Given 单针 8K/16K/32K，When 路由开启，Then `portfolio_choice=selection`（与 AR007 同级或更快）；
- Given multineedle，When 路由开启且表给出 `selection`，Then 走选择级 + PHA；
- Given 曲线指纹不符的模型，When 路由，Then 回退 `selection` 且质量不劣于 AR007。

### 3.3 FR-3 组合回归闸门（Pareto 不许回退）

**描述**：把"组合不能变差"变成可执行检查。

**期望行为**：
- `benchmarks/scripts/check_portfolio.py`：读组合矩阵 CSV + `expected_portfolio.csv`，
  对每个数据集取"最佳固定组合"（质量最高、同质量取最快）与"路由组合"比较；
- 判定：路由质量 ≥ 最佳固定 -0.01 且 路由 TTFT ≤ 最佳固定 ×1.05 → PASS；否则列出违规项并非零退出；
- 闸门接入 `repro.sh portfolio`。

**验收标准**：路由后 5 个数据集全部 PASS；注入一个劣化组合（人工把表中 16K 改成 gate-p16）
时闸门必须 FAIL（自测）。

### 3.4 FR-4 同会话组合矩阵实测

**描述**：所有组合在同一会话、同一 dense 参照下测一遍，作为路由表与闸门的唯一数据源。

**期望行为**：
- 组合矩阵：数据集（4K/8K/16K/32K/mn）× 组合（dense、selection、selection+pha、
  gate-p16、gate-p24-local32），写 `benchmarks/results/ar008-portfolio/`；
- 报告 `REPORT.md`：每数据集一张表（质量、TTFT、比值、审计列），并明确写出
  哪些组合被淘汰、为什么（含 PHA 前后的 mn 对比）。

### 3.5 FR-5 复现资产

- `benchmarks/matrix/portfolio/{portfolio.txt,conf/*.conf,expected_portfolio.csv}`；
- `benchmarks/scripts/{portfolio_matrix.py,check_portfolio.py}`；
- `repro.sh portfolio`（CPU 约 20-30 分钟，含 dense 参照）；
- README 结果表、research §10、architecture、server-repro §8、组件详设 v0.6 变更记录。

## 4. 非功能需求

| 项 | 要求 |
|---|---|
| 性能 | PHA 引入的额外开销 ≤2% TTFT（O(n_blocks×|phrases|) 集合查询）；路由为 O(1) 表查询 |
| 可靠性 | 表/指纹异常一律降级到 `selection`，绝不失败；`phrase_weight=0` 与 AR007 逐位一致 |
| 确定性 | 无随机源；同输入同输出逐位一致 |
| 可观测 | `signals.phrase`、`anchor_blocks`、`portfolio_choice`、`phrase_hits` 全部进 raw.csv |
| 兼容 | 新增列追加在 raw.csv 末尾；不改变既有列语义；不新增 llama.cpp patch |

## 5. 约束与不做的事

- **不新增 patch**：AR008 全部在 host 侧（选择器 + planner + 路由），避免再次引入图内
  形状类缺陷；图内路径直接复用 AR006/AR007 已验证的实现；
- **不做学习型路由器**：样本量不足（5 数据集 × 5 组合），表由实测直接生成，
  规则可解释、可审计，不做概率模型；
- **不做 token 级 masked FFN**（AR007 已用证据否决）；
- **不引入外部依赖**：短语/IDF/路由全部用现有 C++17 标准库与现有配置解析器。

## 6. 术语

| 术语 | 含义 |
|---|---|
| 判别符 | 能把目标块与相似块区分开的词：数字、含数字代码、长词、CJK |
| 短语（bigram） | 问句相邻两词，如 "vault 5" |
| 组合 | 一条完整的执行路径配置：选择级/门控/预算/窗口的组合 |
| Pareto 不劣 | 质量不低于且延迟不高于参照组合（允许 1%/5% 容差） |
| 最佳固定组合 | 同一数据集上质量最高（同质量取最快）的固定组合 |

## 7. 依赖的既有能力

- AR007：`budget_curve`、`anchor_blocks`、span growing、MMR、审计列约定；
- AR006/AR007：`qg_bound` 图内门控、去拷贝页打分；
- AR004/AR005：`per_pos_scores`、qg-2d 组×块矩阵（本 AR 只改信号来源，不改它们的实现）；
- 现有 `terms_from_text`/IDF/`minmax` 基础设施与 `config.cpp` 严格解析器。
