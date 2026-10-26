# [AR008] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-26 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 短语感知准入：查询 bigram 提取（保留短数字判别符）+ 块短语 IDF + `signals.phrase` + 锚点 + `per_pos` | - | passing | FR-1。实现含**跨块 bigram**（块 b 末词+块 b+1 首词）与**最高 IDF 短语锚定全部命中块**（两处裁剪均保护）；`per_pos` 同步外溢给 AR004/AR006 |
| T002 | `phrase_weight` 配置键（conf + CLI，0 = 关闭，范围校验） | T001 | passing | FR-1。conf `phrase_weight` + CLI `--phrase-weight`，范围 [0,2]；0 时完全不构建短语结构 |
| T003 | 单测：短语提取/判别符保留/`phrase_weight=0` 逐位等价/确定性 | T001,T002 | passing | FR-1。`selector_phrase_keeps_numeric_discriminator`、`selector_phrase_crosses_block_boundary`、`selector_phrase_weight_zero_matches_ar007` + 既有确定性用例 |
| T004 | prompt profiler（n_tokens/n_specific_terms/phrase_hits/concentration）+ `Sample.portfolio_choice` | T001 | passing | FR-2。profile 直接复用选择器一次 pass 的输出（`specific_terms`/`phrase_blocks`/`lexical_concentration`），零额外分词；审计列 `phrase_blocks`/`specific_terms`/`portfolio_choice`/`portfolio_reason` |
| T005 | 组合表加载器（行式 key=value + 指纹）+ planner 路由 + 显式意图优先 + 降级审计 | T004 | passing | FR-2。`portfolio.{h,cpp}`：严格解析、mtime 缓存、`model_sha256` 校验；显式 `qg_bound` 优先；缺表/指纹不符 → `fallback:*` + selection |
| T006 | 组合矩阵脚本 `portfolio_matrix.py` + `matrix/portfolio/*.conf` | T005 | passing | FR-4。脚本生成 matrix.csv/md + portfolio.txt/json；`matrix/portfolio/{selection,gate-p16,gate-p24-local32}.conf` |
| T007 | 同会话实测矩阵（dense/selection/selection+pha/gate-p16/gate-p24-local32 × 5 数据集） | T001,T006 | passing | FR-4。20 格点：4 长度 × {dense,selection,gate-p16,gate-p24-local32} + 多针 × 4；`results/ar008-portfolio/{matrix.csv,matrix.md,REPORT.md}` |
| T008 | 诚实评估：mn q≥0.8、单针不回归、PHA 开销 ≤2%；不达标则调权重或回退 | T007 | passing | FR-1/2。多针 0.333→**1.000**（b=0.05，6/6）；单针 4K/8K/16K q=1.000；PHA 开销 8K −1.8%、16K +1.7%（repeats=3，同会话） |
| T009 | `check_portfolio.py` Pareto 闸门 + `expected_portfolio.csv` + 注入劣化自测 | T007 | passing | FR-3。闸门以同会话**最佳固定组合**为基线（非 expected 抄录，见 design §10）；正常 5/5 PASS，`--inject=len16k:gate-p16` FAIL |
| T010 | 路由表定稿（由实测生成，不手填） | T007,T008 | passing | FR-2。`matrix/portfolio/portfolio.txt`：`high:32k = gate-p24-local32`，其余 `selection`，`default = selection`；路由端到端验证 5/5 命中且 32K `qg_source=attention-bound` |
| T011 | 复现资产：`repro.sh portfolio`、server-repro §8、README 结果表 | T010 | passing | FR-5。实际套件名 `ar008`（`./repro.sh ar008`）；server-repro §8；README 结果表 + CLI |
| T012 | 文档：research §10、architecture、组件详设 v0.6 变更记录、REPORT.md | T010 | passing | FR-5。research §10（判别符根因/短语三步/组合矩阵/闸门/矩阵抓到的缺陷）、architecture 增 AR008 节、组件详设 v0.6、REPORT.md |
| T013 | Review：-Werror、全量（无模型 + 真机）、dense 等价、无 mock 审计、列一致性 | T011,T012 | pending | |
| T014 | ST 验收 + 干净克隆 + tag `ar008-v1.0` + Gitee 推送 | T013 | pending | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复
- `cancelled`：经证据否决（备注写原因）

## 进度记录

### 2026-10-26 需求阶段

- **根因重做**（推翻了 AR007 的假设）：`multineedle` 三条针的唯一判别符是单个数字
  （`vault 4/5/6`），而 `selector.cpp:218` 的过滤条件 `term.size() >= 2 || CJK`
  把单字符数字丢掉，导致三块词法得分相同、正确率≈1/3，与实测 q=0.33 吻合。
  数据核对：`mn-02.txt` 中 "vault 5" 唯一出现（位置 50.1%），代码
  `CRIMSON-LYNX-7624` 紧随其后。
- **组合问题**：现有 6 条路径（选择级 / gate-p16 / gate-p24-local32 / qg-2d /
  压缩分支 / dense）在 32K 上质量与延迟差异显著（7s/q=0.89 vs 31s/q=1.00），
  但没有任何机制按 prompt 特征选组合；AR007 只解决预算不解决路径。
- SOTA 参考与取舍（详见 design.md §1）：BM25/短语检索（位置 bigram 提升判别符权重）、
  FrugalGPT/RouteLLM 级联路由（成本-质量折中，按输入特征选模型/路径）、
  Matryoshka/多向量检索（本 AR 不采纳：需要额外编码器）、
  SnapKV/H2O（decode 期重击者，与本 AR 的 prefill 准入目标不同）。
  本轮采纳：**位置短语 + 短判别符**（IR 经典做法，直接修根因）与
  **实测驱动的组合路由 + Pareto 闸门**（把"组合最优"变成可执行检查）。
- 授权记录：用户明确"自动化、无需逐步确认"，故 req 门控以自检替代交互式审查。
