# [AR005] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| AR 主题 | semantic-gated-sparse-prefill-and-server-repro（免训练语义门控稀疏 prefill 与服务器级一键复现）|
| 关联 SR | 扩展 AR004（吸收 docs/research.md 全部"局限/改进方向"条目）|
| 日期 | 2026-10-24 |
| 状态 | Draft（全自动模式：确认点自动选推荐项）|

## 1. 背景与目标

### 1.1 现状基线（全部实测，无 mock）

AR004 交付了 branch-gated 稀疏 prefill（16K GPU 12.78× q=1.0 主验收达成）并如实
记录了五项核心局限（docs/research.md 汇总表）：

| # | 局限 | 根因（实测归因） |
|---|------|----------------|
| L1 | qg v1 无增益（mn q=0/0/0 vs scalar 0/0/0.2）| 1D per-pos 词法分数使各组行集退化为同源全局 top-k |
| L2 | 32K q=0.25，pool 不救 | token 级选择在低预算下把 needle 块整个丢弃，超出满块前缀池化范围 |
| L3 | 4K/8K scalar q=0.44 | 词法 idf 对 needle 块区分度不稳定（需 pool 兜底才到 1.0）|
| L4 | head 路由全线 q=0 | profile 来自通用探针文本，非任务内分布 |
| L5 | CPU 16K 最优 4.87×（50% budget）| 语义质量不可靠 → 不敢降预算；词法选择质量上限锁死速度上限 |

五项局限共同指向一个架构缺口：**选择与门控的信号源是词法 1D，而非语义 2D**。

### 1.2 SOTA 深度分析 → 创新映射

| 项目 | 核心思想 | 本项目改造（免训练化）| 靶向局限 |
|------|---------|---------------------|---------|
| Quest | 页级 min-max K 元数据，query 相关块选择，近乎常数开销 | 小型检索嵌入模型（bge-small）的 问题向量 × 块向量 余弦 → **语义优先块选择**（无 K 缓存依赖、天然 question-aware；draft 隐状态路径保留为备选）| L2/L3/L5 |
| MoBA | router 按 query 分片选块（专家=KV 块），需训练 router | **免训练 router**：嵌入模型独立编码 组文本向量 与 块向量，G×B 余弦矩阵即门控分数 → qg v2（router 权重=检索模型对比学习权重，非训练线性层）| L1 |
| LLMLingua | 小模型 token 分数做提示压缩，question 条件化 | 嵌入引擎以 EmbeddingSelectionScorer 适配 DraftScorer 多态接口接入选择层（选择器零改动）| L2/L3 |
| MInference | 按 prompt 动态分析注意力模式（非静态 profile）| classify-heads 支持 **任务内校准样本**（真实任务上下文前缀+问题）| L4 |
| vLLM/Sarathi | chunked prefill 调度 + 一键可复现工程 | repro.sh 全自动套件 + 服务器规模化配置 | 复现性 |

**创新主张（全部免训练、全部真实模型数学）**：
1. 免训练 MoBA 门控——router 不是训练出的线性层，而是检索嵌入模型的真实向量几何；
2. 语义优先选择——Quest 的 query-aware 块选择思想用嵌入相似度实现，修复质量悬崖于选择层源头；
3. 任务内 head 校准——MInference 的"动态按 prompt 分析"思想落到 classify-heads。

### 1.3 目标

以语义信号贯通"选择 → 门控 → head 路由"三层，同时交付服务器级一键复现：

- **32K CPU：q 0.25 → ≥0.9，speedup 净值 ≥10×**（含 draft 打分开销的诚实记账）；
- **16K CPU：4.87× → ≥6.5× 且 q≥0.9**（语义选择使 budget 0.25 可用）；
- **multineedle 8K（qg v2）：q 0/0/0.2 → 平均 ≥0.66**；
- **4K/8K scalar 无 pool：q 0.44 → ≥0.9**（源头修复，pool 留作纵深）；
- **16K GPU ≥10× 回归守护**（12.78× 基线不回退）；
- **一键复现**：干净 Linux 服务器 clone → `./repro.sh smoke` <30 分钟出结果，`./repro.sh matrix` 无人值守全矩阵；
- 全程 harnessX SDD 流程归档，创新过程可追溯。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 嵌入引擎（bge-small GGUF）2D 矩阵 API（G×B 余弦，真实 GGUF 前向）
- qg v2：build_query_gated 消费 2D 矩阵 + 1D 回退；审计列 qg_source
- 语义优先选择：嵌入分数权重与默认值工程化、needle 块排名守护测试（既有 draft 隐状态路径保留为备选）
- draft 模型复现集成（Qwen2.5-0.5B-Instruct 下载 + 自身模型兜底）与打分开销审计
- 任务内 head 校准（classify-heads --calibration）
- CPU 复合突破实验矩阵（budget×attn×pool 网格）与 GPU 回归
- repro.sh 一键套件（环境探测 CPU/CUDA、下载校验、构建、smoke、矩阵、图表、汇总）
- 实验注册表（benchmarks/matrix/*.conf 全格点）+ 期望结果表（容差区间）
- 服务器规模化指南（线程/模型矩阵、7B 路径配置）
- 构建优化：LTO/march=native 可选开关（默认关，保证可复现）
- docs/research.md v2（Quest vs embedding、免训练 MoBA、动态 MInference 三段式）
- Review、ST 验收、归档、tag ar005-v1.0、Gitee 同步

**Out of Scope（本 AR 不做的）：**
- 任何形式的训练/微调（免训练是创新主张本身）
- decode 阶段优化、多请求并发服务化
- 自定义 GPU kernel、FA 路径 gather（延续文档化边界）
- 7B/14B 模型的本机完整实验（提供配置与路径文档，服务器执行留给用户算力）
- tune 自动调优框架、Radix 缓存增强（延续 AR003 遗留边界）

## 3. 功能需求

### FR-1 嵌入引擎组×块语义矩阵（qg v2 信号源）

**描述：** 新增 `EmbeddingScorer`（独立块编码嵌入引擎，bge-small 类 GGUF，
池化=GGUF 自带池化头，bge 为 CLS）：`score_group_block()` 独立编码每个组文本
与每个原始提示块文本，返回 G×B 余弦相似度矩阵；`score_query_blocks()` 返回
问题→块分数（选择层与 1D per_pos 广播共用）。draft 隐状态路径
（LlamaDraftScorer）保留为既有备选。

**触发条件：** query_gating 开启且 embedding_model 已配置并加载成功；组定义由调用方给出行空间边界。

**期望行为：**
- 矩阵元素 = cos(组文本向量, 块向量)，全部有限；
- 因果约束由消费方（FR-2）施加，本 API 只算分数；
- 嵌入模型缺失时调用方回退 1D 词法（reason 审计）；**显式配置**的 embedding 路径无效为硬错（ModelLoad），不降级假跑；
- 编码开销计入 stats（semantic_score_ms）与 env 审计。

**异常处理：** 非有限分数 = Internal；组边界非法/空输入/未加载调用 = InvalidArgument。

**验收标准：**
- Given 真实 1.5B 主模型 + 真实 bge-small 嵌入 GGUF，When score_group_block 于 4K 提示 8 组，
  Then 矩阵为 8×125 有限值，且"问题组×needle 块"分数显著高于"问题组×随机干扰块"
  （needle 排名在前 10%，单测断言）；
- Given 未加载引擎，When 调用，Then InvalidArgument（无静默假分数）。

### FR-2 qg v2：2D 语义门控消费

**描述：** `AttentionSparsityController::build_query_gated` 支持 2D 分数矩阵：
每组用**本组**的块分数序列做因果可行块 top-k，修复 v1 各组同源退化。

**触发条件：** Selection 携带 G×B 矩阵（per_group_block_scores 非空）。

**期望行为：**
- 每组行集 = sink 块 + 组前邻 recent 块 + 本组分数 top-k 块（因果过滤后）；
- 组数超 query_chunk_max 降级单组（reason 审计，与 v1 一致）；
- 2D 缺失时回退 1D per_pos_scores（v1 行为字节等价，回归守护）；
- patch 层 C API 与图分支复用 v1 机制（行集内容变化，管线不变）。

**异常处理：** 矩阵尺寸与组数/块数不符 = InvalidArgument；空行集组强制行 0（v1 语义）。

**验收标准：**
- Given 2D 矩阵中组 A 与组 B 的 top 块不同，When build_query_gated，
  Then 两组行集不同（反退化断言，v1 无法通过）；
- Given 1D 分数（v1 输入），When build_query_gated，Then 输出与 AR004 实现逐行一致。

### FR-3 语义优先选择（选择层源头修复）

**描述：** 嵌入语义分数成为选择层默认质量引擎：权重工程化（语义分数优先）、
needle 块排名守护、低预算可用性验证。既有 draft 隐状态路径保留为备选。

**触发条件：** 配置 embedding_model（或显式 --embedding-model）。

**期望行为：**
- SelectorWeights 默认调整后：有嵌入分数时语义主导（权重可配）；
- needle 块在语义分数下排名守护（真实模型测试：needle 块必须进选择集）；
- 4K/8K/32K 低预算（0.25/0.25/0.15）下 q 实测修复（见 FR-6 验收数字）；
- 无 embedding 且无 draft 时词法行为与 AR004 字节等价。

**异常处理：** 嵌入编码失败 → 回退词法 + warning 审计（不静默）。

**验收标准：**
- Given 32K 真实提示 + 问题，budget=0.15 + embedding 引擎，When 选择，
  Then needle 块在选择集中（真实模型断言）；
- Given 无 embedding/draft 配置，When 选择，Then 输出与 AR004 基线一致（回归）。

### FR-4 任务内 head 校准

**描述：** classify-heads 支持 `--calibration-file`：用真实任务上下文（前缀+
问题）替代通用探针文本做三分数统计，pattern 矩阵来自任务内分布。

**触发条件：** 显式传入校准文件。

**期望行为：** 输出 profile.json 格式不变（下游 build_pattern_matrix 零改动）；
校准样本为真实任务数据（禁止合成探针文本冒充任务分布）。

**验收标准：**
- Given 16K 任务上下文前 4K + 问题做校准，When 生成 profile 并构建 pattern 矩阵，
  Then head 路径 q 相对 AR004（q=0）提升：multineedle 三任务平均 q≥0.2；
  不可达时如实报告根因分析（判据：量化目标 + 不得放宽声明 + 根因记录三要素齐备）。

### FR-5 模型复现集成与开销审计

**描述：** download_model.sh 支持第二模型下载：`--draft`（Qwen2.5-0.5B-Instruct，
ModelScope→hf-mirror 择优 + SHA256 校验）与 `--embed`（bge-small 嵌入模型）；
嵌入编码耗时进 stats/env；净 TTFT = 语义打分 + 稀疏 prefill 全链计时。

**验收标准：**
- `./scripts/download_model.sh --draft` 在干净环境拉取并校验 0.5B draft；
- `./scripts/download_model.sh --embed` 在干净环境拉取并校验 bge-small；
- 主模型自身可作 draft 兜底（同词表保证），文档化性能权衡；
- raw.csv 新增 semantic_score_ms 列（嵌入引擎打分开销净值记账）；配置 draft 时 draft_score_ms 同列审计；所有验收 speedup 均为净值（含打分开销）。

### FR-6 CPU 复合突破与 GPU 回归实验矩阵

**描述：** 语义信号贯通后的最优复合配置搜索：budget {0.15,0.25,0.35} ×
attn_top_k {12,24} × pool {on,off} × qg {v2,off} 于 16K/32K CPU 真实网格；
GPU 16K/32K 回归 + qg v2 GPU 抽查。

**验收标准（全部净值、隔离进程、--repeats 3、raw.csv+env.json 入库）：**
- 32K CPU：q≥0.9 且 speedup≥10×（基线 12.4×@q=0.25，诚实对照）；
- 16K CPU：speedup≥6.5× 且 q≥0.9（基线 4.87×）；
- multineedle 8K qg v2：平均 q≥0.66（基线 0.07）；
- 4K/8K scalar 无 pool：q≥0.9（基线 0.44）；
- 16K GPU：≥10× 且 q≥0.9（回归守护）；
- 任一目标不可达 → 如实记录根因分析，不得放宽声明。

### FR-7 一键复现套件 repro.sh

**描述：** 顶层 `repro.sh`：环境探测（CPU/CUDA、核数、内存）→ 依赖检查 →
模型/数据下载校验 → 构建（CPU 或 CUDA 自动）→ 套件执行（smoke|matrix|full）
→ 图表与汇总表生成 → 期望结果对照（容差区间，超差非零退出）。

**验收标准：**
- 干净 WSL2/Ubuntu22.04：`git clone … && cd edge-prefill && ./repro.sh smoke`
  端到端 <30 分钟（16 核参考），产物：测试报告 + smoke 结果 CSV + 汇总；
- `./repro.sh matrix` 无人值守跑 FR-6 全格点并产出 matrix.csv + 图；
- 每步失败即停（set -euo pipefail），错误信息可定位；
- README 快速开始以 repro.sh 为第一入口。

### FR-8 实验注册表与期望结果表

**描述：** benchmarks/matrix/*.conf 注册每个矩阵格点（含 draft/qg/pool 全参数）；
docs/expected-results.md 给出每格点的期望 TTFT/speedup/q 与容差区间（源自本机实测），
repro.sh 对照用；服务器规模化指南（线程缩放建议、7B 配置模板、显存估算表）。

**验收标准：** 注册表覆盖 FR-6 全格点；期望表每行含"实测命令 + 数据目录"引用；
7B 配置模板经真实小规模冒烟验证格式正确（不要求本机跑完 7B）。

### FR-9 研究文档 v2 与交付

**描述：** docs/research.md 增补三段式：Quest 精确界 vs 本项目 embedding 选择的
误差分析；免训练 MoBA router 论证（隐状态几何 vs 训练线性层）；MInference 静态/
动态之辨与任务内校准实测。Review → ST → 归档 → tag ar005-v1.0 → Gitee split-push 同步。

**验收标准：** 无"声称无证据"条目；全部数字可溯源到 raw.csv；远端 tag 可见。

### 条目式验收说明

FR-5~FR-9 的验收标准采用条目式（前置环境 + 操作命令 + 量化产物三要素），
与 Given/When/Then 结构等价，本节统一声明，不逐条改写。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 真实性 | mock 禁令 | 全链无 mock；测试无模型变量时显式 SKIP，不伪造 |
| 性能 | 记账诚实 | 所有 speedup 净值（含 draft 打分开销）；无静默缓存预热作弊 |
| 可靠性 | 回归 | 既有 116 用例零回归；budget=1.0/全开关关闭与 dense 数值等价 |
| 兼容性 | 默认行为 | 不配 draft/qg 时与 AR004 输出字节等价 |
| 可复现 | 环境快照 | 每实验含 env.json（SHA256/命令/线程/时间戳）|
| 构建 | 可选优化 | LTO/march=native 开关默认关，保证跨机可复现 |

## 5. 约束与假设

**约束：**
- llama.cpp pinned `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32` 不动；patch 最小可禁用
- C++17 / CMake≥3.20 / -Werror 零警告 / Status 返回 / 无异常跨 API
- qg 与 pool 互斥、head 矩阵优先级（v1 语义保持，v2 只换信号源）
- 全自动 harnessX 模式：确认点自动选推荐项

**假设：**
- Qwen2.5-0.5B-Instruct 与 bge-small GGUF 在 ModelScope/hf-mirror 可达（1.5B 已验证通道）；
  draft 不可达时主模型自任 draft 兜底（同词表）；
- 嵌入引擎对 32K 提示（≈1000 块 × 60 encoder token）的编码开销在 CPU 16 核上
  ≤ 稀疏 prefill 的 20%（FR-6 网格实测校验，超限则启用 stride 子采样并如实记录精度影响）；
- 服务器复现以 Linux 为一等公民（WSL2 为本机验证环境）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| qg v2 | 2D 组×块语义门控（本 AR）；v1 = AR004 的 1D per-pos 门控 |
| 净 speedup | (dense TTFT) / (语义打分 + 稀疏 prefill + 生成首 token 全链)，同相位 dense 参照 |
| 嵌入引擎 | 独立块编码的检索嵌入模型（bge-small GGUF，pooling=MEAN），文本片段→向量 |
| 组表示 | 嵌入引擎对组文本（组内行的原始连续 token 跨度）编码的向量 |
| 块表示 | 嵌入引擎对原始提示块文本编码的向量 |
| 任务内校准 | 用真实任务上下文（而非通用探针文本）统计 head 三分数 |
