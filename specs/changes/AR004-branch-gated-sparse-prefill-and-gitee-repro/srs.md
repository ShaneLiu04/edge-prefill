# [AR004] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| AR 主题 | branch-gated-sparse-prefill-and-gitee-repro（NSA/MoBA 式分支稀疏架构创新与 Gitee 开放复现）|
| 关联 SR | 扩展 AR003（吸收其未完成任务 T015–T018 中与文档/交付相关的部分；tune 框架与 Radix 快照增强不在本 AR 范围）|
| 日期 | 2026-10-22 |
| 状态 | Draft（待用户确认后走 req 门控）|

## 1. 背景与目标

### 1.1 现状基线（全部实测，无 mock）

AR001–AR003 已交付完整链路：draft 语义打分 → 块级选择 → attention
全局/分层/head 级 top-k gather（patch 层最小改动）→ FFN skip → chunk 调度。
实测基线（qwen2.5-1.5b-instruct q4_k_m，16 核 CPU，llama.cpp b11748 pinned）：

- **16K prompt**：TTFT 25.9s vs dense 160+s，**6.2x–7.0x**，且已打赢 GPU-FA
  （41.6s）与 GPU-非FA（276.2s）；
- **质量悬崖**：激进稀疏（attn-24 @8K）q=0.44/0.44/0.50——被丢弃块是硬删除，
  针落在被丢块时必然失分；
- **调度浪费 2.5x**：本轮修复 scalar gather 崩溃（sparse_prefill.cpp:259-270
  传 chunk_tokens.size() 对齐内外层拆分）后实测：同一 4K k62 配置，单 ubatch
  21.2s vs 512-chunk 52.5s——**现有全部头条数字都基于 512 分块，有 2.5x 免费
  提升空间未收割**；
- **PPL 指标损坏**：ppl 任务 quality=0（即使 dense），综合质量指标
  （0.4×NIAH+0.4×QA+0.2×PPL）不可用；
- **工程债**：`docs/` 为空、无 git 仓库、gitee 未同步、`third_party/` 空、
  scripts/ 混有大量诊断脚本。

### 1.2 SOTA 深度分析结论

| 项目 | 核心思想 | 对本项目参考价值 | 采纳决策 |
|------|---------|----------------|---------|
| NSA（DeepSeek）| 三分支：压缩上下文+选中 token+滑窗；GQA 组共享选择 | 被丢块池化为 1 行压缩 KV，免训练救回质量悬崖；纯 ggml 算子可实现于 patch 层 | **采纳为核心创新 FR-2** |
| MoBA（Moonshot）| router 按 query 选块 + chunk-gating（查询分片共享决策）| 全局单一行集合 → 按 query-chunk 分组选择；组件详设已预留"逐 query 动态 top-k"扩展点；head 路径分组-gather-concat 机制可复用 | **采纳为核心创新 FR-3** |
| MInference（Microsoft）| prefill 注意力三模式：vertical/slash/block-sparse，按 head 分类 | 验证 sink+recent 必留块设计；用已有 HeadProfile 分数做按 head 模式路由 | **采纳为 FR-4** |
| SnapKV/PyramidKV | 观察窗选 KV；按层金字塔预算 | 无 profile 时的默认 head 矩阵加金字塔先验（中间层更稀疏）| **采纳进 FR-4** |
| Quest | 页级 minmax 元数据按 query 选页 | draft 打分已承担该角色；块选择器加"分段最低保留"防整区域全丢 | **采纳进 FR-2**（锚完整性）|
| LLMLingua-2 | token 分类压缩，分段保留预算 | 同上——分段预算思想 | 同上 |
| FlashAttention-3/FlexAttention | kernel 级优化 | 我们不写 kernel，patch 层保持 FA 兼容约束并文档化 | 不采纳（记录边界）|

### 1.3 目标

交付 **NSA 式压缩上下文分支 + MoBA 式 query-chunk 门控 + 大 chunk 默认调度 +
MInference 式 head 模式路由** 四项创新，达成：

- **16K TTFT ≥10× dense 且综合质量 q ≥0.9**（大 chunk 2.5x 收割 + 分支保质量）；
- 8K 激进稀疏（attn-24）质量从 q≈0.44 提升至 ≥0.75（压缩分支效果的可测证明）；
- **Gitee 开放复现**：干净克隆 → README → 下模型 → 构建 → 一键实验，
  结果 CSV+图入库可对照核验；
- 全程 harnessX SDD 流程归档，创新过程可追溯。

## 2. 需求范围

**In Scope：**
- 大 chunk 默认调度（chunk 对齐 n_ubatch）+ 崩溃修复回归测试 + 全长度基准刷新
- NSA 式压缩上下文分支（patch 扩展：被丢块 → 池化压缩 KV 行）
- MoBA 式 query-chunk 门控（draft 分数矩阵 → 每查询分片独立选块）
- MInference 式 head 模式路由 + PyramidKV 金字塔先验默认矩阵
- PPL 质量指标修复与综合质量指标落地
- GPU sparse 路径（phase5 CUDA 构建与 CPU/GPU 对照实验）
- Gitee 仓库工程化：git 初始化、sync 脚本、README 复现文档、docs/ 填充、
  模型下载脚本（ModelScope→hf-mirror 自动择优 + SHA256 校验）、
  结果 CSV+图入库、诊断脚本清理
- 主实验矩阵（4K/8K/16K/32K × {dense, global, head, branch, gated} × CPU/GPU）
  与可解释性图表
- 研究文档 docs/research.md（SOTA 对比三段式）
- Review、ST 验收、归档、release tag

**Out of Scope：**
- 模型训练/微调（压缩分支免训练，用固定权重注意力天然加权）
- decode 阶段算法优化
- 多请求并发服务化
- flash attention 路径的 gather（延续限制，文档记录；FA 下自动回退）
- 自定义 GPU kernel
- `tune` 自动调优框架、Radix 缓存/按层 KV 增强（AR003 遗留，后续 AR）

## 3. 功能需求

### 3.1 FR-1 大 chunk 默认调度与基准刷新

**描述：** chunk 调度默认对齐 n_ubatch（单次 llama_decode 完成整个 chunk），
取代 512 固定分块；保证 scalar gather 路径在任意 chunk 配置下不再出现
内外层拆分不一致（本轮崩溃根因的回归防线）。

**触发条件：** 默认配置即生效；`--chunk-size` 显式覆盖。

**期望行为：**
- `Config.chunk_size` 语义变更：`0`=自动（对齐 n_ubatch），`>0`=显式值；
  默认 0
- `ChunkScheduler` 传给 backend 的 chunk 必须单次 llama_decode 完成
  （backend 不得二次拆分；在 sparse_prefill.cpp 修复处加断言/回归测试）
- 等价性：大 chunk 下 budget=1.0 且稀疏全关 → 与 dense logits diff <1e-5

**验收标准：**
- Given 4K k62 配置, When 默认运行, Then TTFT 相比 512 分块降低 ≥2x
  （实测 21.2s vs 52.5s 的回归化）
- Given 任一 chunk 配置（含 0/512/n_ubatch/超大值）, When 稀疏 prefill,
  Then 无崩溃且输出正确（原崩溃配置进入回归测试）
- Given 大 chunk + 全稀疏关闭, When prefill, Then 与 dense 等价 <1e-5

### 3.2 FR-2 NSA 式压缩上下文分支

**描述：** 被丢弃块不再硬删除：每块池化（mean-pool）为 1 行压缩 KV，
与已选行拼接进注意力集。查询对压缩行正常算注意力分（softmax 天然加权），
实现免训练的"粗粒度上下文记忆"。同时块选择器增加**分段最低保留**
（每原始文本段至少保留 1 块，防整段全丢，Quest/LLMLingua 分段预算思想）。

**触发条件：** `--compressed-branch on`（默认 on）且稀疏激活且有被丢块。

**期望行为：**
- patch 扩展：新增 C API
  `llama_set_prefill_attn_compressed(ctx, rows, n)`（被丢块的池化行号集合，
  语义=这些块的行参与池化）或等价最小接口；池化用 ggml 现有算子
  （reshape/mean/concat），不写自定义 kernel
- K 与 V 分别池化；压缩行参与 KQ mask 时视为"可见"，位置用块内最大位置
  （保证因果性：压缩行仅对块之后查询可见）
- 关闭时完全恢复现行为（被丢块=硬删除）；`budget=1.0` 时无被丢块、分支空转
- 质量目标：attn-24 @8K q 从 0.44 → ≥0.75；预算稀疏 16K q 不低于现值
- 开销：压缩行数 = 被丢块数 ≤ 总行数/block_size，注意力规模增幅 ≤10%
  （32 行块池化为 1 行，压缩率 32:1）

**异常处理：** patch 缺失 → `Unsupported` → 回退硬删除并记录
`compressed-branch-unsupported`；池化后行数溢出 n_kv 上限 → 该 chunk 回退
dense 并审计。

**验收标准：**
- Given attn-24 @8K 真实模型, When 压缩分支开启, Then 综合质量 ≥0.75 且
  TTFT 相比关闭分支增幅 ≤15%
- Given 全稀疏关闭或 budget=1.0, When 运行, Then 分支不产生任何额外计算，
  logits 与 dense 等价 <1e-5
- Given 分段最低保留启用, When 任意激进配置, Then 每个原始文本段至少 1 块入选，
  NIAH 针在段内时不再因整段丢失而失分
- Given 压缩分支关闭, When 运行, Then 行为与 AR003 逐字段一致（回归零破坏）

### 3.3 FR-3 MoBA 式 query-chunk 门控

**描述：** 现有全局行集合对所有查询一视同仁；扩展为**查询分片（query
chunk）级独立选块**：draft 打分阶段保留 per-position 分数 → 按 query-chunk
（默认 128 token）聚合出 分片×块 得分矩阵 → 每分片独立 top-k → graph 按
分片循环 gather-concat（复用 head 路径的分组机制）。

**触发条件：** `--query-gating on` 且稀疏激活；head 矩阵路径与 query-gating
二选一（head 矩阵优先时记录 `query-gating-skipped`），避免图复杂度相乘。

**期望行为：**
- draft scorer 输出扩展：per-position 分数矩阵（位置×块），仅深层信号
  （现有 draft 已算，落盘复用，不重复前向）
- planner 生成分片级行集合（每分片必含 sink+recent+该分片前文锚点）
- patch 扩展：graph 按分片循环——每分片 gather 自己的行集合、注意力、
  concat 输出；分片数上限（默认 16）防图爆炸，超限安全降级全局集合
- 关闭时完全恢复全局单一集合行为（AR003 一致）
- 质量与性能：中段查询（远离问题文本的查询）获得更贴合的块集合，
  8K+ 长度质量提升可测；gather 总规模不增（每分片 k 更小，总行数不变）

**异常处理：** per-position 分数缺失（draft 禁用）→ 回退全局集合并记录
`query-gating-no-scores`；分片行集合为空 → 该分片强制 sink+recent。

**验收标准：**
- Given 8K/16K 真实模型多针数据, When query-gating 开启 vs 关闭, Then
  综合质量不降且至少一个长度档质量提升 ≥5 个百分点（或报告不可达分析）
- Given query-gating 关闭, When 运行, Then Plan 与 AR003 全局路径逐字段一致
- Given head 矩阵与 query-gating 同时启用, When 规划, Then head 矩阵优先且
  reasons 记录跳过原因

### 3.4 FR-4 MInference 式 head 模式路由与金字塔先验

**描述：** 用已有 `HeadProfile`（echo/induction/retrieval 分数）把 head 分为
三模式：**锚点型**（vertical：强 sink 依赖 → 必留首块+小 k）、**局部型**
（slash：强对角依赖 → 必留 recent+小 k）、**检索型**（retrieval：稠密或大 k）；
无 profile 时启用 **PyramidKV 金字塔先验**默认矩阵（浅层/深层偏密、中层偏疏）。

**触发条件：** `--head-pattern-routing on` 且 head profile 存在（路由模式）；
profile 缺失且 `--pyramid-prior on`（默认矩阵模式）。

**期望行为：**
- 路由模式：profile 分数 → 三类模式 → 模式各自必留模式与 k 值模板；
  阈值可配置、默认值文档化
- 金字塔模式：k(l) = 按层位置的凸函数（中层最小），上下限可配置；
  与 AR003 per-head 矩阵格式兼容（生成默认矩阵走既有路径）
- 两模式均不引入新 patch（复用 AR003 head 矩阵机制），纯 planner 层实现
- 关闭时与 AR003 行为逐字段一致

**验收标准：**
- Given 无 profile + 金字塔模式, When 16K 稀疏 prefill, Then 质量不低于
  同 k 预算的均匀矩阵且 TTFT 不高于均匀矩阵的 105%
- Given 有 profile + 路由模式, When 运行, Then 三类 head 的 top-k 与必留块
  符合模板映射，审计字段记录每 head 分类结果
- Given 两开关全关, When 规划, Then 与 AR003 逐字段一致

### 3.5 FR-5 质量指标修复与综合质量落地

**描述：** 修复 ppl 任务 quality=0（当前 dense 也是 0，metric 通路损坏）；
落地综合质量指标 q = 0.4×NIAH + 0.4×QA + 0.2×PPL，全部主实验配置输出。

**验收标准：**
- Given dense 配置, When 跑 ppl 任务, Then quality>0 且与 llama.cpp 官方
  ppl 量级一致（可交叉验证）
- Given 任一主实验行, When 汇总, Then CSV 含综合 q 列且分项可追溯
- Given 修复前后的 dense ppl, When 对照, Then 修复不改变其他任务的
  quality 语义（回归）

### 3.6 FR-6 GPU sparse 路径与对照实验

**描述：** phase5 CUDA 构建（`/root/llama-src-cuda` 本地源避免 9P 慢），
验证 scalar gather 与 head 矩阵路径在 CUDA 后端的正确性与收益；
输出 CPU vs GPU sparse 对照（含 GPU dense 基线已就绪数据）。

**验收标准：**
- Given CUDA 构建, When 稀疏 prefill, Then 输出与 CPU 同配置语义一致
  （综合质量 q 差异 ≤0.02，NIAH/QA 逐任务一致）且无图构建崩溃
- Given 16K GPU sparse vs GPU dense-FA, When 对照, Then 加速比 ≥1.3x
  （FA 基线 41.6s；若 gather 在 GPU 无收益则诚实报告并分析原因）
- 实验数据含 raw.csv + env.json（GPU 型号/driver/toolkit 记录）

### 3.7 FR-7 Gitee 仓库工程化与开放复现

**描述：** 仓库达到"陌生用户 30 分钟内复现头条实验"标准：
git 初始化 → 结构清理 → README/docs → 模型/数据下载 → 一键实验 →
结果对照。

**期望行为：**
- **结构清理**：诊断脚本（diag_*/equiv_*/recheck_*）移入
  `scripts/diagnostics/` 或删除；`third_party/` 写 .gitignore（llama 源
  构建期拉取）；构建产物/大文件不入库
- **`scripts/download_model.sh`**：依次尝试 ModelScope → hf-mirror.com，
  成功即校验 SHA256 与 README 记录一致；失败给出明确手动指引
- **`scripts/sync_gitee.sh`**：适配本仓库的增量同步（尊重 gitee 限流；
  凭据不落库）
- **README.md**：快速开始（3 条命令跑通）、实验配置表（模型/量化/SHA256/
  线程/n_ctx/预算矩阵）、预期结果表（对照仓库内 CSV）、硬件要求、
  故障排查（CUDA/网络）
- **docs/**：`architecture.md`（模块图）、`experiments.md`（实验方法学）、
  `research.md`（见 FR-9）
- **结果入库**：`benchmarks/results/` 全部 CSV + 图 + env.json 入库；
  头条图表由脚本可重生成
- 远端：gitee.com/liu-xingyan04/edge-prefill，master 分支 + release tag

**验收标准：**
- Given 干净克隆 + README 步骤, When 新环境执行, Then 下载模型（校验通过）
  → 构建（ctest 全过）→ `run_experiments.sh` 一键复现 ≥1 个头条配置
  且复现 TTFT 落在仓库 CSV 值的 0.5x–2x 区间内（超出则输出机器差异分析）
- Given `git status`, When 检查, Then 无构建产物/模型/凭据泄漏
- Given README 审阅, When 检查, Then 含全部大模型实验配置项与预期结果表

### 3.8 FR-8 主实验矩阵与可解释性图表

**描述：** 系统对照矩阵 + 便于深度解释的图表集，全部由真实 CSV 生成
（matplotlib 脚本入库）。

**期望行为：**
- 矩阵：{4K, 8K, 16K, 32K} × {dense, global-topk, head-adaptive,
  +compressed-branch, +query-gating, 全开} × {CPU, GPU}（32K/GPU 视时间裁剪）
- 图表（每张可独立解释一个现象）：
  1. TTFT-长度曲线（各配置，对数轴）——**头条图**
  2. 质量-加速比 Pareto 前沿（AR002/003/004 三代对比）
  3. 压缩分支消融：attn-24 各长度 q 值开/关对比——**质量悬崖救回图**
  4. query-gating 消融：各长度 q 提升量
  5. chunk 大小- TTFT 曲线（512 → n_ubatch 收割图）
  6. head 模式分布（路由分类后的层×头热力图）
  7. CPU vs GPU sparse 对照柱状图
- 每张图的数据来源 CSV 路径在图注/文档中注明

**验收标准：**
- Given 全矩阵数据, When 生成图表, Then **16K 全开配置 TTFT ≥10× dense
  且 q ≥0.9**（主验收线；未达标给出 Pareto 前沿与不可达分析）
- Given 任一图表, When 审阅, Then 背后 CSV 在仓库且命令可重生成该图
- Given 32K 档, When 运行, Then 至少 CPU 路径完整（数据集已在
  /tmp/deep/len32k）

### 3.9 FR-9 研究文档 docs/research.md

**描述：** 本 AR 创新过程的结构化沉淀：每个创新点含
"SOTA 核心思想 → 本项目映射与改造 → 实测证据（CSV/图引用）→ 局限"。

**验收标准：**
- Given docs/research.md 审阅, When 检查, Then 覆盖 ≥6 个 SOTA 项目
  （NSA/MoBA/MInference/Quest/SnapKV/LLMLingua-2）三段式对比
- Given 每个创新点声明, When 对照实验报告, Then 有 raw.csv + 图表数字佐证
  （无"声称无证据"条目）

### 3.10 FR-10 Review、ST 验收与交付

**描述：** 代码规范审查（Status/RAII/命名/-Werror/覆盖）、ST 端到端验收、
harnessX 归档、Gitee 最终同步 + release tag。

**验收标准：**
- Given 全部开发任务完成, When review, Then 规范复核全过才进 ST
- Given 真实模型环境, When ST, Then 主验收线（≥10x/≥0.9）实测通过并归档
- Given ST 通过 commit, When tag, Then 远端可见且指向该 commit

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 16K TTFT | **≥10× dense 且 q ≥0.9**（主验收线）|
| 性能 | 4K/8K | ≥3×/≥5×（大 chunk 刷新后）|
| 质量 | attn-24 @8K | 压缩分支使其 ≥0.75（悬崖救回证明）|
| 质量 | 综合指标 | q=0.4×NIAH+0.4×QA+0.2×PPL 全配置输出 |
| 等价性 | 全开关关闭 | logits 与 dense diff <1e-5 |
| 内存 | 额外开销 | ≤20%（不含可选 draft/profile）|
| 可靠性 | patch 缺失/非法输入 | 明确错误码+安全回退，绝不半写状态 |
| 复现 | 干净克隆 | gitee 克隆按 README 复现（模型/数据/构建/实验）|
| 可追溯 | 图表/声明 | 每个数字可追溯 raw.csv + env.json + 命令 |

## 5. 约束与假设

**约束：**
- 无 mock：全部延迟/质量数据来自真实 llama.cpp + GGUF 推理；测试无模型时显式 SKIP
- llama.cpp 固定 commit `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32`；patch
  最小、可审查、可禁用、关闭时完全恢复原路径
- C++17 / CMake ≥3.20 / `-Wall -Wextra -Werror`；Status 返回、无跨边界异常
- GPU 实验顺序化执行（防 CPU 饿死 GPU 驱动线程，本轮已证）
- flash attention 下 gather 路径禁用（自动回退，文档记录）
- 实验与开发并行：phase4/5 后台继续跑，创新开发不阻塞在实验上

**假设：**
- 池化压缩 KV 的质量收益在免训练设定下显著（NSA 结论 + 我们 NIAH 结构，
  若实测不显著则按验收标准诚实报告并保留机制为可选开关）
- gitee 网络可达且支持 40KB+ 推送批次；ModelScope/hf-mirror 至少一个可达
- 高算力服务器可用（GPU phase5）或本机 RTX 5000 路径即可完成 GPU 对照

## 6. 与历史 AR 的关系

- **AR001–AR003**：全部能力保留为基线；AR004 所有新开关关闭时行为与 AR003
  逐字段一致（回归零破坏是硬门禁）
- **AR003 遗留**：tune 框架、Radix 快照增强、按层 KV 增强不在本 AR；
  文档/交付/实验类任务由本 AR 对应 FR 承接
