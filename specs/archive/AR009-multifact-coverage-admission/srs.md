# [AR009] 需求规格说明书：多事实覆盖准入

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 主题 | multifact-coverage-admission |
| 日期 | 2026-10-27 |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.6） |
| 前序 | AR001–AR008（均已归档） |
| 状态 | 需求已定稿（用户授权自动化流程，自检替代交互确认） |

## 1. 背景与问题（来自实测，不是推测）

AR008 把多针数据集 `deep/multineedle` 的选择级质量从 0.333 修到 **1.000**，
但该数据集有一个结构性盲点：**每行只问一个事实**（"passcode for vault 5"）。
只要选中正确的那一块，得分就是 1.0。真实预填负载里更常见的是
**一问多事实**——"把 A、B、C 三个仓的密钥都列出来"——答案需要若干块同时在场。

为此本轮新增数据集生成器 `benchmarks/scripts/gen_multifact.py`：每个上下文埋
`--facts` 条独立事实，每行问其中 `--asked` 条，行质量按**命中事实比例**给分
（部分分可见，而不是全对才得分）。两档数据：

| 数据集 | 埋入事实 | 每行提问 | 行数 |
|---|---|---|---|
| `deep/multifact8k` | 6 | 3 | 16 |
| `deep/multifact8k-hard` | 12 | 6 | 16 |

**实测到的差距**（budget=0.05，Qwen2.5-1.5B-Instruct Q4_K_M，phrase_weight=0.5）：

| 数据集 | AR008 选择级 q | 覆盖事实数 |
|---|---|---|
| multifact8k | 0.488 | 0/3 |
| multifact8k-hard | **0.261** | 0/6 |

块预算约 14 块（443 token），三到六个事实每个只需一块就够，**预算不是瓶颈**；
问题是准入只看"整体最相关的块"，把整个预算花在一条最强的字面匹配上。

### 1.1 根因（逐条由调试与单测确认）

1. **判别符被长度启发式丢掉**。AR007 的 entity 锚点规则是
   `term.size() >= 6 || 含数字 || CJK`，于是 `one`/`two` 这类短键永远不会成为
   独立锚点；而 `vault`/`secret`/`passcode` 这类模板词又被 `df_cap` 之外的
   路径反复锚定，结果"一条事实吃满预算"。
2. **问句回声块冒充证据**。chat template 把问句重复在**最后一个块**里，
   该块因此包含**全部**子查询。任何"按查询词覆盖"的逻辑都会在第一条事实进入
   之前就宣布"全部已覆盖"。第一轮实现正是如此：覆盖恒为 0，选择结果与 AR008
   逐位相同（三个实验全部 q 相同）。
3. **锚点打分把子查询合并**。第二版实现用 `idf_p + Σ_{q≠p} idf_q·hit` 选锚点，
   于是每个子查询都指向同一个"命中最多"的块，后续子查询发现块已占位就跳过，
   真正的事实块反而一个都没锚上。
4. **数据集事实键必须经得住分词器**。用"仓库编号"当键看似自然，但
   `8491` 会被切成 `849`+`1`，数字键泄漏进答案串；用稀有全大写代号
   （`AURORA`）又会被切成 `AUR`+`ORA`，判别符根本到不了选择器。改用
   **拼写数字**（`vault one/two/three`）后两个问题都消失：单 token、不在
   填充词表、不在密钥串里。

### 1.2 顺带修掉的既有缺陷

新增单测首次覆盖了"一个 piece 含多个词"的输入，直接触发
`selector.cpp` 的**堆越界读**（ASan 确认）：AR008 的短语外溢用
`flat_seq` 的下标去索引 `per_pos`（长度 = n_tokens）。当某个 piece 解出多个词时
`flat_seq` 比 `n_tokens` 长，`per_pos[i+1]` 越界。已在 AR009 中修正为
`flat_tok` 映射（词 → 所属 token）。

## 2. 目标

1. **一问多事实要能答**：在多事实族上把选择级质量从 0.49/0.26 提到 ≥0.85/≥0.60
   （同等预算，同一模型）；
2. **单事实族零回归**：单针/长度族的选择结果与 AR008 **逐位一致**，
   `coverage_weight=0` 时整体等价于 AR008；
3. **可审计**：覆盖了几个请求事实必须写进 `raw.csv`（`coverage_queries`），
   不能只报总分；
4. **真实可复现**：数据集由确定性生成器产出并带 SHA256；实验用真实 GGUF 前向，
   无 mock、无合成分数。

## 3. 功能需求

### 3.1 FR-1 多事实数据集族（MFD）

**描述**：提供"一问多事实、按比例给分"的数据集族，用于诚实评估覆盖能力。

**期望行为**：
1. `gen_multifact.py --facts F --asked A --rows-per-context R --cases C --words W --seed S`
   产出 `contexts/*.txt` + `index.tsv` + `manifest.json`（逐上下文 SHA256）；
2. 事实键为**拼写数字**（`one..twelve`），单 token、不与填充词表/密钥串冲突；
3. 每行问句形如 `List the secret passcodes for the vaults one, three and five.`，
   答案按提问顺序给出 `COLOR-ANIMAL-NNNN`；
4. 质量口径沿用 token-F1（现有 `metrics`），部分命中自然体现在 F1 上。

### 3.2 FR-2 覆盖式准入（CQF, coverage-fair fill）

**描述**：把"一问多事实"的准入从"取最相关块"改成"保证每条请求事实都有证据"。

**期望行为**：
1. **子查询提取**：查询词序列中的**稀有词**（在提示中命中块数 1–2）作为子查询；
   稀有度是判据，不用长度启发式（长度会丢掉 `one`/`two` 这类短键）；
   上限 8 个，按块级 IDF 降序；
2. **回声排除**：命中子查询数达到 `max(2, ceil(n_sub/2))` 的块视为问句回声，
   从证据中剔除（chat template 的末块即属此类）；
3. **每事实一锚点**：每个子查询锚定它与整句最相关的那一块（`idf_p + lexical[b]`，
   纯 facility location，不叠加兄弟子查询 IDF），受 `target/2` 锚点预算约束；
   该路径**取代** AR007 的 entity 锚点启发式（两者目的重复，同时启用会挤爆预算）；
4. **覆盖式填充**：填充分 = `adjusted[b] + span_grow + coverage_weight · gain(b)/idf_max`，
   其中 `gain(b) = Σ_p max(0, idf_p − covered_p)·hit(p,b)`；选中后立即更新 `covered`；
5. **零回归开关**：`coverage_weight=0` 或子查询 <2 时**完全不建**覆盖结构，
   选择与 AR008 逐位一致；
6. 复杂度：每次填充 O(n_blocks·P)，P≤8；填充次数 ≤ 预算块数。

### 3.3 FR-3 审计与闸门

**期望行为**：
1. `Selection.coverage_queries` = 命中的子查询数，落到 `raw.csv` 的
   `coverage_queries` 列（**追加在末尾**，旧脚本按列名读取不受影响）；
2. `benchmarks/scripts/ar009_ablation.py` 跑 (数据集 × coverage_weight) 网格，
   产出 `results/ar009-multifact/{ablation.csv,ablation.md,ablation.json}`，
   记录模型 SHA256、预算、线程、max_new_tokens 与原始 raw.csv；
3. 闸门：`coverage_weight=3` 在两个档位上 q_mean 均须 ≥ `coverage_weight=0` + 0.20，
   且 `covered_queries` ≥ `asked − 1`；不满足则视为失败并回退。

### 3.4 FR-4 非回归与工程约束

**期望行为**：
1. 单针 `deep/multineedle`、长度族 `explen/len4k|len8k|len16k` 在 `coverage_weight=3`
   下选择结果与 AR008 逐位一致（`selected_tokens` 与质量均不变）；
2. `budget=1.0` 仍是 dense 路径，不受影响；
3. `-Wall -Wextra -Werror` 零警告；无模型单测与真机测试全通过；
4. 新增单测覆盖：零回归开关、每事实入选、回声排除、单事实 no-op、ASan 干净。

### 3.5 FR-5 复现资产

1. `repro.sh ar009`：构建 → 生成两档数据集 → 跑消融 → 校验闸门；
2. `benchmarks/scripts/check_expected_ar009.py` + `expected_ar009.csv`：把实测值
   固化为可机检的期望（列表驱动，缺列即失败）；
3. README 结果表 + `docs/research.md` §11 + `docs/architecture.md` + `docs/server-repro.md` §9。

## 4. 非功能需求

| 项 | 要求 |
|---|---|
| 可复现 | 数据集生成确定性；实验记录模型 SHA256 / 线程 / 预算 / 命令 |
| 无 mock | 全部质量数字来自真实 GGUF 前向 |
| 诚实 | 未达成的目标（如 hard 档未到 0.85）必须写明根因，不得粉饰 |
| 兼容 | 新审计列只在末尾追加；`coverage_weight` 缺省时行为等同开启，但显式 0 完全关闭 |
| 性能 | 覆盖结构只在多子查询时构建；单事实族零额外工作 |

## 5. 明确不做（本轮否决）

| 方向 | 理由 |
|---|---|
| 学习型覆盖打分（训练小模型） | 样本量 32 行，必然过拟合；不可解释 |
| 查询改写/扩展（把枚举展开成文档句式） | 依赖外部模型，且已在数据集侧用拼写数字消除歧义 |
| 图内（AR006）路径的覆盖改造 | 本轮只做选择级；图内改动风险高、需重新验证 patch |
| decode 期 KV 淘汰 | 不在 prefill 准入范围 |
| 跨块窗口邻近度（AR008 暂缓项） | 与短语信号重叠，先看覆盖收益再定 |

## 6. 验收标准

1. `deep/multifact8k`：q_mean ≥ 0.85（实测 0.878），`covered_queries` ≥ 2；
2. `deep/multifact8k-hard`：q_mean ≥ 0.60（实测 0.652），`covered_queries` ≥ 5；
3. 单针/长度族：选择逐位不变（质量 1.000，`selected_tokens` 不变）；
4. 闸门脚本在正常数据上 PASS，在人为劣化（如 `coverage_weight=0`）上 FAIL；
5. 干净克隆可构建、`ctest` 全通过、`./repro.sh ar009` 可复现表格。
