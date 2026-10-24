# [AR005] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-24 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | EmbeddingScorer 嵌入引擎（bge-small GGUF，多 seq 并行编码，池化 MEAN）+ score_group_block/score_query_blocks 接口 + build_group_spans 组跨度推导；单测（几何/校验/余弦性质/跨度一致性）+ 真实模型 needle 排名断言（S1） | - | passing | 2026-10-24 完成，FR-1/FP1/FP2 |
| T002 | build_query_gated 2D 重载：每组本组分数因果 top-k；空矩阵回退 1D 与 v1 字节等价回归；反退化断言测试；矩阵尺寸/非有限校验 | T001 | passing | FR-2/FP3 代码与用例已落地（提交 c57b58b） |
| T003 | 管线接线：Selection 加 per_group_block_scores；planner/main/benchmark/Config 键（embedding_model + 既有 query_gating 自动升 2D，无新增 gating 键）+ CLI + qg_source 审计列 + semantic_score_ms/draft_score_ms 统计 | T002 | passing | FR-2/FR-5 代码与用例已落地（提交 c57b58b） |
| T004 | 语义优先选择：EmbeddingSelectionScorer 适配器（DraftScorer 多态，选择器零改动）+ 语义优先默认权重 + needle 块进选择集真实模型守护测试（32K budget 0.15）+ 无引擎回归字节等价 | T001 | passing | FR-3/FP4 代码与真实模型用例已落地（提交 c57b58b） |
| T005 | draft 模型复现集成：download_model.sh --draft（0.5B，SHA256 校验）+ 主模型自任 draft 兜底文档化 + 净 TTFT 记账落地 | - | passing | FR-5：download_model.sh --draft/--embed/--all + README 复现条目；draft_score_ms/semantic_score_ms 净记账 |
| T006 | 任务内 head 校准：classify-heads --calibration-file；profile 格式不变；真实任务数据校准测试 | - | passing | FR-4 代码与真实模型校准用例已落地（提交 c57b58b） |
| T007 | CPU 复合突破网格：budget{0.15,0.25,0.35}×attn{12,24}×pool×qg-v2 于 16K/32K + multineedle qg-v2 + 4K/8K 源头修复验证；隔离进程 --repeats 3 全量入库 | T003,T004,T005 | pending | FR-6 |
| T008 | GPU 回归 + qg-v2 GPU 抽查：16K/32K GPU 矩阵，≥10× 回归守护 | T007 | pending | FR-6 |
| T009 | repro.sh 一键套件：环境探测/下载校验/构建(CPU/CUDA)/smoke|matrix|full/图表/期望对照/非零退出；README 第一入口改写；LTO/native 构建开关 | T005,T007,T010 | pending | FR-7 |
| T010 | 实验注册表 + 期望结果表 + 服务器规模化指南：benchmarks/matrix/*.conf 全格点；docs/expected-results.md 容差区间；7B 配置模板 | T007,T008 | pending | FR-8 |
| T011 | docs/research.md v2：Quest 精确界 vs embedding 误差分析；免训练 MoBA router 论证；MInference 静态/动态之辨 | T007,T008 | pending | FR-9 |
| T012 | Review：-Werror 双目标零警告、全量真实模型回归（116+ 用例零回归）、默认路径与 AR004 字节等价、无 mock 审计 | T007-T011 | pending | FR-9 |
| T013 | ST 验收：FR-6 验收矩阵实测核对 + 归档 + tag ar005-v1.0 + Gitee split-push 同步 | T012 | pending | FR-9 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-24 会话记录

- 完成任务：T001 EmbeddingScorer 嵌入引擎 + build_group_spans 组跨度推导
- 修改文件：include/edge_prefill/embedding_scorer.h（新增）、src/embedding_scorer.cpp（新增）、include/edge_prefill/attention_sparsity.h（build_group_spans 声明）、src/attention_sparsity.cpp（build_group_spans 实现）、tests/test_embedding_scorer.cpp（新增，6 用例）、CMakeLists.txt（源与测试注册）
- 测试：122 PASS（快速模式 116 既有零回归 + 6 新增）；真模型 S1 PASS（bge-small needle 排名前 10%）
- 实现要点：池化=GGUF 自带池化头（bge=CLS；causal 无池化→Unsupported），已同步 srs/design/组件详设三处文档；llama_tokenize 空缓冲返回负所需长度（须取反）；encoder 整批不拆分，n_batch/n_ubatch=512×8
- 覆盖率：lcov 工具链未配置，降级为用例映射自查（6.1/6.2/6.3-S1 全部用例落地；OOM 分支按 6.2 豁免声明处理）
- 备注：bge-small-en-v1.5-q8_0.gguf 已下载至 D:\edge-llm\models\，SHA256=f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804（ggml-org ModelScope 通道）

### 2026-10-24 会话记录（T002-T006 状态同步与 T005 落地）

- 完成任务：T002/T003/T004/T006 状态同步（代码与用例已在提交 c57b58b 落地，本会话按代码核验后置 passing）；T005 draft/embedding 模型下载与复现条目落地
- 修改文件：scripts/download_model.sh（--draft/--embed/--all 三模式 + SHA256 校验 + 主模型自任 draft 兜底说明）、README.md（draft/embedding 模型 SHA256 与硬错语义）、tasks.md（状态表）
- 测试：ctest build-wsl 1/1 通过（快速模式 0.38s 零回归）
- 实现要点：显式 draft/embedding 路径加载失败=硬错不降级；净 TTFT 记账（semantic_score_ms 独立于 prefill 计时）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。

### 2026-10-24 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（6 YES / 3 WARN，无 Critical/Important）
- 修复的问题：G2 条目式验收等价声明已补；G3 FR-4 阈值量化（mn 平均 q≥0.2）+ FR-3 交叉引用 FR-7→FR-6 已修正；G7 T009 依赖补 T007/T010
- 审查代理：sdd-gate-reviewer

### 2026-10-24 design 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G2/G14⑦（Important，srs FR-1/FR-3/FR-4 语义引擎表述仍为 draft 隐状态，design 已选方案 A bge-small，规格未同步修订）；G12（Important，FP7/FP9/FP10 在 §6 无对应用例）；G13（Important，"行集空→强制行 0"分支、降级链"1D 不可用→scalar""scalar 关→dense"两级、OOM 代码路径无 §6 用例）；G14（Important，FR-4 量化验收/FR-5 draft_score_ms 与主模型兜底/FR-6 数字/FR-7 matrix 模式/FR-8 注册表与期望表行格式/FR-9 溯源验收无 §6 追溯用例）
- Minor：G3（EmbeddingSelectionScorer 缺参数/错误码表）、G7（A/B/C 无权衡对比表，方案 B"可配置备选"无配置项）、G11（S5 判据双向放行，OOM"由单测覆盖"未落实）、G15（score_group_block/score_query_blocks 未加载前置与 Internal(嵌入全零) 无触发用例）
- 严重性：Important（4 项）
- 审查代理：sdd-gate-reviewer

### 2026-10-24 design 门控记录（第 2 轮）

- 门控结果：FAIL（第 2 轮）
- 上轮 4 项 Important 已确认修复到位（srs 引擎表述对齐、§1 对比表、§4.3.2 参数表、6.1 行集空/降级链/OOM、6.2 未加载前置、S5 硬判据、S10-S12）
- 失败项：G10（Important，组件_spec.md EmbeddingScorer 条目"非有限→InvalidArgument"与 design §4.3.1/srs FR-1"非有限→Internal"错误码矛盾，spec 为孤例）；G14（Important，FR-5 "--draft 干净环境拉取并校验"验收在 §6 无追溯用例；FR-7 README 第一入口、FR-6 CPU 数字"达标或根因三要素"断言缺口）
- Minor：G2（stride 应急路径未设计）、G3（build_group_spans 无错误码表）、G6（降级链无流程图）、G9（适配器实现文件未指明，T003 键名 query_gating_semantic 与设计不一致）、G11（6.1 表格竖线错位）、G12（FP6 --draft 正常拉取无用例）、G15（load_model OOM/适配器空指针前置未入 6.2）
- 严重性：Important（2 项）
- 审查代理：sdd-gate-reviewer

### 2026-10-24 design 门控记录（第 3 轮）

- 门控结果：PASS（第 3 轮）
- 审查项数：15 项（12 YES / 3 WARN，无 Critical/Important）
- 修复的问题（累计 3 轮）：第 1 轮——srs 语义引擎表述对齐（draft→bge-small 主）、§1 方案对比表、§4.3.2 参数表、§6 补 FP7/FP9/FP10 用例/行集空分支/降级链审计/OOM 落实/未加载前置/S5 硬判据/S8-S12；第 2 轮——组件_spec 错误码主语分离（引擎 Internal / 消费方 InvalidArgument）、S13 模型下载校验、S7 补 README 第一入口、S12 补 FR-6 达标断言、stride 应急路径说明、build_group_spans 错误码表、降级链流程图、适配器归属 semantic_gate.cpp、T003 键名对齐、6.2 OOM 豁免与空指针前置；第 3 轮 Minor 顺手修复——选择层 score() 失败回退显式化、降级链 reason 字符串对齐（lexical-1d-unavailable）、校准文件过短边界、build_group_spans 空指针用例
- 审查代理：sdd-gate-reviewer
