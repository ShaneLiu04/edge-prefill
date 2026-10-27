# [AR009] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-27 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 多事实数据集生成器 `gen_multifact.py`（拼写数字键 + 部分分） | - | passing | FR-1。`--facts/--asked/--rows-per-context`，manifest 带逐上下文 SHA256；键用 `one..twelve`（单 token、不在填充词表/密钥串） |
| T002 | 子查询提取（稀有词判据 df≤2、去重、IDF 降序、≤8）+ `block_seq` 常驻 | T001 | passing | FR-2。`block_seq` 改为总是构建（覆盖与短语共用）；不用长度启发式（否则丢 `one`/`two`） |
| T003 | 回声排除（块命中 ≥ max(2,ceil(n/2)) 视为问句回声） | T002 | passing | FR-2。chat template 末块含全部子查询；未排除时 `covered` 预满、选择与 AR008 逐位相同 |
| T004 | 每事实一锚点（`idf_p + lexical[b]`）+ 取代 entity 启发式 | T002 | passing | FR-2。叠加兄弟 IDF 会让所有子查询指向同一块；启用覆盖时 `sub_idf` 非空则跳过 entity 锚点 |
| T005 | 覆盖式填充 `gain(b)=Σ max(0, idf_p−covered_p)` + 归一化权重 | T004 | passing | FR-2。`coverage_weight` 默认 3.0；已覆盖即归零 → 单事实自动退化为 AR008 |
| T006 | `coverage_weight` 配置键 + CLI/config/bench 传播 + 0 = 关闭 | T005 | passing | FR-3。conf `coverage_weight`（0..10）；**修复** `main.cpp` 漏映射（两轮消融因此无效，见 design §3.5） |
| T007 | 审计列 `coverage_queries`（Selection→Sample→raw.csv 末尾追加） | T005 | passing | FR-3。列位置在 `specific_terms` 之后、`portfolio_choice` 之前；旧脚本按列名读取不受影响 |
| T008 | 单测：零回归开关 / 每事实入选 / 回声排除 / 单事实 no-op | T005,T006 | passing | FR-4。`selector_coverage_*` 4 例；首次运行即发现 `per_pos` 越界（见 T009） |
| T009 | 修复短语外溢 `per_pos` 越界（`flat_tok` 词→token 映射） | T008 | passing | FR-4。ASan 确认 heap-buffer-overflow：`flat_seq` 下标当 token 下标用；一个 piece 多词时越界 |
| T010 | 消融脚本 `ar009_ablation.py`（两档 × 两权重，落 raw/CSV/MD/JSON） | T007 | passing | FR-3。记录模型 SHA256/预算/线程/max_new_tokens；`covered facts` 列来自审计列 |
| T011 | 闸门 `check_expected_ar009.py` + `expected_ar009.csv` + 负向自测 | T010 | passing | FR-3。正例 2/2 PASS；把 cw=3 行改写成 cw=0 值后 FAIL（rc=1） |
| T012 | 诚实评估：两档提升、单事实族不回归、32K 不劣化 | T010 | passing | FR-1/2。multifact8k 0.488→**0.878**、hard 0.261→**0.652**；4K/16K/mn 选择逐位一致；32K 0.875→0.893 |
| T013 | 复现资产：`repro.sh ar009` + 数据集入库 | T011 | passing | FR-5。`./repro.sh ar009` = 生成两档数据 + 消融 + 闸门 |
| T014 | 文档：README、research §11、architecture、server-repro §9、组件详设 v0.7、REPORT.md | T012 | passing | FR-5。含"未达成目标"（hard 0.652 与 dense 天花板未测） |
| T015 | Review：-Werror 零警告、无模型 + 真机全量、ASan、无 mock 审计 | T009,T014 | passing | `-Wall -Wextra -Werror` 零警告；无模型 155 PASS；ASan 覆盖用例干净；全部数字来自真实 GGUF 前向 |
| T016 | ST 验收 + 干净克隆 + tag `ar009-v1.0` + Gitee 推送 | T015 | passing | 干净克隆构建 + ctest + `./repro.sh ar009` 闸门 PASS |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复
- `cancelled`：经证据否决（备注写原因）

## 进度记录

### 2026-10-27 需求阶段

- 复现 AR008 遗留盲点：多针族每行只问一个事实；新建 `gen_multifact.py` 后实测
  AR008 选择级在两档上只有 0.488 / 0.261，而预算（≈14 块）足以放下 3–6 条事实
  → 问题在准入目标而非预算。
- 依次确认四个根因（长度启发式丢短键、问句回声冒充证据、锚点打分合并子查询、
  事实键被分词器切碎），见 srs §1.1。
- 否决：学习型覆盖打分、查询改写、图内路径改造、ILP set cover、单纯加大 k。

### 2026-10-27 开发阶段（关键迭代，含两次"无效果"的诚实记录）

1. 第一版（bigram 子查询）：多事实 q 0.214→0.214。调试发现问句是枚举式
   （`vault 1, 2 and 3`），"vault 2" 根本不是相邻 bigram → 子查询为空。
2. 第二版（单词子查询 + 长度判据）：仍 0.214。发现子查询全是数字，
   而数字同时出现在密钥串里（`8491`→`849`+`1`），判别符被污染；
   且问句回声块命中全部子查询 → 覆盖恒 0。
3. 第三版（拼写数字键 + 回声排除）：0.214→0.428，但 `covered` 仍为 0：
   锚点打分叠加兄弟 IDF，所有子查询指向同一块。
4. 第四版（纯 facility location）：选择改善，但消融两组数字完全相同
   ——`main.cpp` 未把 `bench.coverage_weight` 从 `cfg` 拷进 `BenchmarkConfig`，
   两轮实验实际都在跑默认值。修复后第一组真实对比出现（0.535→0.848）。
5. 第五版（去掉长度判据，改用稀有度）：4K/16K/mn 选择逐位不变，
   两档多事实分别定稿在 0.878 / 0.652。

### 2026-10-27 验证阶段

- 无模型单测 155 PASS（新增 4 例）；`build-asan` 跑覆盖用例发现并修复
  `per_pos` 越界（AR008 遗留）。
- 真机 16 行 × 2 档 × 2 权重消融；闸门 2/2 PASS，负向自测 FAIL。
- 非回归：`coverage_weight=3` 下 4K/16K/mn 与 AR008 逐位相同；32K q 0.875→0.893。

## ST 验收记录（2026-10-27）

| # | 用例 | 结果 |
|---|---|---|
| ST-1 | 多事实主对比：同一预算（0.05）、同一机器，`coverage_weight` 0 → 3（16 行/档） | PASS（multifact8k 0.488→0.878；hard 0.261→0.652） |
| ST-2 | 覆盖审计：`coverage_queries` >= 提问数-1 | PASS（3.25/3、6.50/6） |
| ST-3 | 单事实族零回归：4K sel=322、16K sel=793、多针 sel=401，q 均 1.000 | PASS（选择逐位一致） |
| ST-4 | 32K 不劣化 | PASS（q 0.875→0.893） |
| ST-5 | 零回归开关：`coverage_weight=0` 不建覆盖结构、`coverage_queries=0` | PASS（单测 `selector_coverage_off_matches_ar008` + 消融） |
| ST-6 | 每事实入选：3 条请求事实的块全部在选集中 | PASS（`selector_coverage_admits_every_requested_fact`） |
| ST-7 | 问句回声排除：末块含全部子查询时仍能覆盖 3/3 | PASS（`selector_coverage_ignores_question_echo`） |
| ST-8 | 单事实 no-op：单键查询 token 集合与 AR008 完全一致 | PASS（`selector_coverage_is_noop_for_single_fact`） |
| ST-9 | 越界回归：一个 piece 多词时不再读写 `per_pos` 越界 | PASS（ASan 干净；修复前 heap-buffer-overflow） |
| ST-10 | 闸门正例：`check_expected_ar009.py` 2/2 PASS | PASS |
| ST-11 | 闸门负例：把 cw=3 行改写成 AR008 值必须 FAIL | PASS（rc=1，逐条列出 q/covered/delta 违规） |
| ST-12 | 全量测试：无模型 155 PASS；真机 155 PASS/0 FAIL（共享 VM 负载下约 2h15m） | PASS |
| ST-13 | `-Wall -Wextra -Werror` 全量构建（含干净构建树） | PASS（warnings: 0） |
| ST-14 | 干净克隆（本地 clone 3e0d936）：构建 + ctest 100% + 闸门 2/2 PASS | PASS |
| ST-15 | 数据集可复现：按 README 命令重新生成两档数据 | PASS（`diff -r` 逐字节一致） |
| ST-16 | 复现资产：`repro.sh ar009`、`ar009_ablation.py`、`check_expected_ar009.py`、`expected_ar009.csv`、server-repro §9 | PASS |

### 未达成 / 待办

- hard 档未达 SRS 0.85（实测 0.652）：445 token 内放不下 6 条答案句 + 格式，
  `covered=6.5/6` 证明事实已进上下文，缺的是预算而非准入；
- hard 档 dense 天花板未测（16 行 dense 前向超出会话时限），不做"相对 dense"声明；
- 覆盖增益只作用于选择级，图内（AR006）路径未消费；GPU 未验证（本机无 GPU），
  沿用 AR007 T009 待办；
- 两档数据集为合成生成（真实模型真实前向，非用户流量），外推到真实负载需另测。
