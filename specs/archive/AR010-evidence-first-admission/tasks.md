# [AR010] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-27 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 32K 预算去向审计（导出选中块 + 各信号分量） | - | passing | 发现 50 块里 30 块零证据（62% 预算），全部位于文档尾部 |
| T002 | 定位根因：`minmax` 别名 bug（`*lo` 是正在被写入的向量的迭代器） | T001 | passing | 修复前 diversity∈[0,3.2]、position∈[0,1.157]；lexical 因 min=0 恰好正确 |
| T003 | 修复归一化（先拷贝上下界）+ 单测三分位覆盖 | T002 | passing | `selector_signals_are_bounded_after_normalisation` |
| T004 | 拆分 evidence / prior，加饱和门控 `gate(e)=min(1,e/0.25e_max)` | T003 | passing | `prior_gate` 开关，0 = 加法融合（消融用） |
| T005 | 显式停止语义（budget / evidence-exhausted / no-signal / dense）+ `fill_priors` | T004 | passing | 4K/多事实回归即由"填充依赖数值巧合"暴露 |
| T006 | 审计列 `evidence_blocks` / `stop_reason`（Selection→Sample→raw.csv 末尾） | T005 | passing | 旧脚本按列名读取不受影响 |
| T007 | 配置/CLI 接线（`prior_gate`/`fill_priors` + `--prior-gate`/`--fill-priors`） | T004 | passing | 两次踩到"main.cpp 未映射"的坑，已补 |
| T008 | 消融：prior_gate × fill_priors 四象限 × 7 数据集 | T006 | passing | 见 design §8；`fill_priors=0` 在 5 个数据集上 token 数降 1.3-3.2× |
| T009 | 硬负例抑制（稀有词错配惩罚）实验 | T008 | failed | 填充词同样稀有 → 误伤，多事实 0.576→0.566、hard 0.334→0.304；**已回退** |
| T010 | 组合矩阵重测（5 臂 × 7 数据集）+ 生成新表 + Pareto 闸门 | T008 | passing | 7/7 PASS；注入 `deep/len32k:selection-prior` 按 q_mean/q_min/TTFT 三项 FAIL |
| T011 | 多事实族复核：AR009 的 0.878/0.652 依赖同一归一化 bug | T008 | passing | 修正后 0.566/0.304（仍优于 AR008 基线 0.488/0.261），需在 REPORT 明示 |
| T012 | CUDA 构建 + GPU 一致性（稀疏/门控/质量） | T003 | passing | 稀疏 4K 与门控 8K 的 q/sel 逐例一致，GPU 快 4.8-5.5×；dense 路径 CUDA 上 graph split 异常，未下结论（记录为限制） |
| T013 | 文档：README/research §12/architecture/server-repro §10/组件详设 v0.8 | T010 | passing | 含 AR009 更正说明与失败实验记录 |
| T014 | Review：-Werror、无模型全量、CPU 真机全量、无 mock 审计 | T010 | passing | 干净构建 warnings=0；无模型 158 PASS；真机 158 PASS/0 FAIL；ASan 选择器用例干净 |
| T015 | ST + 干净克隆 + tag `ar010-v1.0` + Gitee 分片推送 | T014 | passing | Gitee 克隆（f14e44b）内容齐全、两个闸门通过；新增 `.gitattributes` 修 CRLF 破坏脚本的问题 |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`（含证据）/ `cancelled`

## 进度记录

### 2026-10-27 需求阶段

- 起点是 AR009 遗留的"32K 必须靠门控"：先做预算去向审计而不是加信号，
  结果发现 62% 预算给了零证据块。
- 逐层下钻：先验过范围（归一化 bug）→ 修复后填充提前停（停止语义缺失）→
  两者共同解释了"长上下文质量不稳"。
- 否决：学习型先验、token 级选择、KV 逐块淘汰；硬负例抑制实测无效已回退。

### 2026-10-27 开发阶段要点

- 第一次修 minmax 后 4K 从 1.000 掉到 0.500、多事实 0.878→0.579：
  这是"填满预算靠 bug"的直接证据，促成 FR-2 的显式停止语义。
- `fill_priors` 默认值一度设为 0，导致既有 `selector_budget_and_mandatory` /
  `selector_16k_budget` 等 3 个用例失败（它们断言"预算必须填满"）。
  按"组合优先、默认保守"的原则把默认恢复为 1（配额语义），
  内容定预算作为 `selection` 臂由组合表选择。

## ST 验收记录（2026-10-27）

| # | 用例 | 结果 |
|---|---|---|
| ST-1 | 预算审计：32K 零证据块占预算比例 | PASS（62% → 修复后 0 个零证据块进入非锚点位置） |
| ST-2 | 归一化边界：所有信号 ∈[0,1]（单测三位置覆盖） | PASS（`selector_signals_are_bounded_after_normalisation`） |
| ST-3 | 内容定预算停止语义：`fill_priors=0` → `stop_reason=evidence-exhausted` | PASS（单测 + 7 数据集实跑） |
| ST-4 | 配额语义保留：`fill_priors=1` → `stop_reason=budget` 且填满 | PASS（单测 + 既有 3 个预算用例不回归） |
| ST-5 | 先验门控不改变证据准入 | PASS（`selector_prior_gate_keeps_evidence_admitted`） |
| ST-6 | 4K/8K/16K/多针质量不回归（q=1.000） | PASS（同会话矩阵） |
| ST-7 | 32K 选择级达标（无门控 q≥0.99） | PASS（q=1.000，525 token，2260 ms；门控臂 5443 ms 被支配） |
| ST-8 | 组合表由实测重新生成 + Pareto 闸门 | PASS 7/7 |
| ST-9 | 闸门有效性：注入劣化必须 FAIL | PASS（3 项违规全部报出） |
| ST-10 | GPU 一致性（CUDA）：稀疏 4K、门控 8K | PASS（q/sel/`qg_source` 逐例一致；GPU 4.8-5.5×） |
| ST-11 | 多事实地板更正 + `repro.sh ar009` 复验 | PASS（0.836/0.667，闸门 2/2） |
| ST-12 | 全量测试：无模型 158 PASS；真机 158 PASS/0 FAIL | PASS |
| ST-13 | `-Wall -Wextra -Werror` 干净构建 | PASS（warnings: 0） |
| ST-14 | ASan：选择器用例无越界/未定义 | PASS |
| ST-15 | Gitee 干净克隆：内容齐全 + 两个闸门 + repro 语法 | PASS（并新增 `.gitattributes` 防 CRLF 破坏脚本） |

### 未达成 / 待办

- 多事实 easy 档 0.836（AR009 原报 0.878 含归一化 bug 红利），hard 档 0.667（原 0.652）；
- hard 档 `q_min=0.087`：6 条答案句 + 格式在 445 token 内仍偏紧；
- 32K / 多针的 dense 参照为 2 例子集（全量 dense 在本机需数小时 decode）；
- CUDA 上 dense 路径出现 391 个 graph split，未纳入结论，留作后继；
- `fill_priors=0` 需经组合表启用（库默认保持配额语义，避免破坏 attention/head 不变式）。
