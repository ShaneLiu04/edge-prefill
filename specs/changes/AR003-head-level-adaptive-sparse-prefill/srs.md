# [AR003] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | head-level-adaptive-sparse-prefill（Head 级自适应稀疏 Prefill 与 SOTA 融合创新）|
| 关联 SR | supersede AR002（T005–T017 未完成任务并入本 AR）|
| 日期 | 2026-10-21 |
| 状态 | Reviewed（req 门控 PASS，2026-10-21）|

## 1. 背景与目标

AR002 已交付 draft 语义打分 + 块级选择 + attention 全局 top-k gather 执行路径
（T001–T004，8K TTFT 2.33–2.43x、质量 100%）。SOTA 深度分析（MInference、
RazorAttention、H2O、StreamingLLM、PagedAttention/Radix、Twig/FastV）表明：

1. **全局 top-k 是粗粒度的**——不同 head 的注意力模式差异极大（retrieval/induction
   head 必须稠密，非检索 head 可激进稀疏），per-head 稀疏是 2–4x 的未开发收益空间；
2. **head 功能可离线探测**（RazorAttention：echo/induction 检测，无需训练）；
3. **深层注意力信号优于浅层**（Twig 实证），印证 draft 打分融合深层信号的方向；
4. AR002 的 Radix 缓存、按层 KV 精度、自动调优、复现框架尚未交付，与创新同属
   一个高质量交付闭环。

本 AR 目标：交付 **head 级自适应稀疏 prefill**（离线 head 分类 + per-head gather
patch + 层-头二维自适应 + 在线质量风险控制），吸收 AR002 全部遗留任务，达成
**8K TTFT ≥3.5x、4K ≥2.5x、质量 ≥97%**，并以 CPU+CUDA 双路径复现框架支撑
高速算力服务器上的高效复现实验。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- llama.cpp per-head attention gather patch（最小、可审查、可禁用）
- 离线 head 分类工具（真实模型 echo/induction 探测 → JSON profile）
- 层-头二维自适应稀疏规划（planner/selector 扩展，draft 深层信号融合）
- 在线质量风险控制器（逐请求自动加密预算 / dense 回退，全程审计）
- Radix 前缀缓存与 KV 分页快照（跨请求/多轮真实 state 复用）
- 按层 KV 精度 patch 与相位感知调度
- 自动调优 Pareto 框架（基于真实 BenchmarkRunner）
- CPU + CUDA 实验环境、复现脚本、README/docs
- 质量回归、主实验矩阵（≥3.5x/≥2.5x/≥97%）、性能迭代
- 架构与研究分析文档（SOTA 对比、创新总结）
- Gitee 同步工具沉淀（`scripts/sync_gitee.sh`，≤40KB 批次）与 release tag

**Out of Scope（本 AR 不做的）：**
- 模型训练/微调、蒸馏 head 分类器
- decode 阶段算法优化（spec 明确边界外）
- 多请求并发服务化调度
- flash attention 路径的 per-head gather（延续 AR002 首版限制，文档记录）
- 自定义 GPU kernel（依赖 llama.cpp 上游算子）

## 3. 功能需求

### 3.1 FR-1 Head 级稀疏 gather patch

**描述：** 扩展 `patches/llama-prefill-attn-top-k.patch`，支持 per-head
attention gather indices，取代仅全局 top-k 的单张量 gather。

**触发条件：** `llama_set_prefill_attn_head_top_k` 矩阵已设置且
`llama_set_prefill_attn_ranked_rows` 池非空、`prefill_mode` 激活、非 flash
attention。

**期望行为：**
- C API（int32 错误码 0=Ok, 2=InvalidArgument, 9=Unsupported）：
  `llama_set_prefill_attn_ranked_rows(ctx, rows, n)` 设置按分数排名的候选 KV
  行池（不要求升序，排名序）；`llama_set_prefill_attn_head_top_k(ctx, top_k, n)`
  设置层×头 top-k 矩阵（n 必须 == n_layer×n_head；元素 ==-1 表示该 head 稠密
  直通，≥1 表示 gather 排名前 k 行）
- graph 构建：每层按 distinct top-k 值分组（≤4 组，超限该层安全降级 dense）；
  组内取池前 k、过滤未写行（row < live n_kv）、内部升序后 gather
  K/V/KQ mask；dense 组（-1）直通原路径零 gather 开销
- 关闭（矩阵清空或未设置）时完全恢复 AR002 标量路径/原路径，logits 与 dense
  max diff < 1e-5

**异常处理：** patch 缺失 → `Unsupported` → 回退 AR002 全局 top-k 路径，
`Plan.reasons` 记录 `per-head-unsupported`；其余功能不受影响。

**验收标准：**
- Given 真实模型与合法 ranked 池 + 矩阵, When 启用后 prefill, Then dense head
  （-1）所在层的 attention 输出与 dense 该层一致（首 token logits 整体
  <1e-5），稀疏 head 仅对其排名前 k 的已写行计算
- Given 矩阵长度 ≠ n_layer×n_head、任一元素为 0/<-1/越界、或矩阵非空而池为空,
  When 调用 API, Then 返回 2 且模型状态不变
- Given patch 未应用, When 启用 per-head, Then 回退全局 top-k 且审计记录降级原因
- Given 矩阵清空, When prefill, Then 与 AR002 标量路径/dense 等价（<1e-5）

### 3.2 FR-2 离线 head 分类器

**描述：** 新增 CLI 子命令 `classify-heads`：在真实 GGUF 模型上运行重复
token 探测序列，统计每层每 head 的 echo/induction 行为，输出 JSON head profile。

**触发条件：** 用户显式执行 `edge-prefill classify-heads --model M --output P`。

**期望行为：**
- 探测序列：固定种子的随机词表 token 序列 `S = [A₁..A_m] × r`（重复块），
  参数（m、r、种子）可配置、默认值文档化；全程真实模型前向，无任何合成分数
- echo 检测：head h 在重复位置 i 对 i−m 的注意力质量分数
- induction 检测：追加 `A₁A₂..` 前缀的 recall 探测，head 对前次出现位置的注意力
- 输出 `head_profile.json`：`{model_sha, n_layer, n_head, probe_params,
  heads:[{layer, head, echo_score, induction_score, class}]}`，
  class = `retrieval | normal`，阈值可配置
- 确定性：相同模型 + 相同参数 → 输出逐字节一致（供复现校验）

**异常处理：** 模型加载失败返回 `ModelLoad`；输出路径不可写返回 `Io`；
序列长度超过 n_ctx 返回 `InvalidArgument` 并给出建议参数。

**验收标准：**
- Given 真实模型与默认参数, When classify-heads, Then JSON 输出含全部层×head
  记录且 sha256 可复现
- Given 探测长度 > n_ctx, When 运行, Then 返回 2 与明确错误信息
- Given 无模型环境, When 运行, Then 测试显式 SKIP（不伪造）

### 3.3 FR-3 层-头二维自适应稀疏规划

**描述：** planner 加载 head profile，结合 draft 打分与 budget 生成
层-头二维 top-k 矩阵：retrieval head 全量、非检索 head 按 budget 与 draft
信号差异化分配，深层信号权重提升（MInference/Twig 思想）。

**触发条件：** `Config.head_profile` 非空且 sparse 模式启用。

**期望行为：**
- `Plan` 新增 `per_head_top_k`（层×头矩阵或空=dense/全局回退）
- budget 语义保持：`budget=1.0` 或 profile 缺失 → AR002 行为完全不变
- 非 retrieval head 的 top-k 分配单调于 draft 分数；mandatory 锚
  （sink/recent/coverage/best-lex）逐 head 保留
- 消融开关：`--head-adaptive on|off`，off 时退回全局 top-k

**异常处理：** profile 与模型结构不匹配（n_layer/n_head 不符）→ `InvalidArgument`，
不静默使用错误 profile。

**验收标准：**
- Given 合法 profile + budget<1, When 规划, Then 每个 retrieval head top_k=n_kv、
  非 retrieval head top_k ≤ 全局 top-k 上限且按 draft 分数单调
- Given profile 层数与模型不符, When 加载, Then 返回 2
- Given `--head-adaptive off`, When 规划, Then Plan 与 AR002 全局路径逐字段一致

### 3.4 FR-4 在线质量风险控制器

**描述：** 逐请求评估 draft 打分的区分度与覆盖率；风险高时自动加密
top-k 预算，仍不足时回退 dense；全程审计可追溯。

**触发条件：** sparse 模式 + draft 打分完成后的规划阶段。

**期望行为：**
- 风险信号：draft 分数分布（top/bottom 分离度）、selected 覆盖率、
  mandatory 锚完整率
- 三级动作：`ok`（原预算）→ `tighten`（预算提升，幅度可配置）→
  `dense-fallback`（该请求禁用稀疏）
- 审计：`Plan.reasons` 记录 `risk-tighten`/`risk-dense` 与触发数值；
  metrics CSV 新增 `risk_level` 列

**异常处理：** 风险计算失败按 `tighten` 处理（保守侧），绝不静默放行低质量稀疏。

**验收标准：**
- Given 构造低区分度 draft 场景（真实模型、退化输入）, When 规划, Then
  触发 tighten 或 dense-fallback 且 reasons 含具体数值
- Given 高区分度场景, When 规划, Then risk_level=ok 且预算不变
- Given 关闭开关 `--risk-guard off`, When 规划, Then 行为与无控制器逐字段一致

### 3.5 FR-5 Radix 前缀缓存与 KV 分页快照（吸收 AR002 T005/T006）

**描述：** 跨请求/多轮前缀 KV 复用：Radix 树最长前缀命中、真实 state
snapshot 序列化、LRU 淘汰、签名校验、恢复失败回退重算。

**验收标准：**
- Given 共享 system prompt 的二次请求（8K 上下文场景）, When 命中前缀, Then
  复用 token 数 = 共享前缀长且 TTFT 加速 ≥5x
- Given 签名不匹配（模型/精度/上下文变化）, When find, Then 不命中且不崩溃
- Given 恢复失败, When 应用 snapshot, Then 丢弃缓存并重算，结果正确
- 缓存指标（hit_rate/reused_tokens/saved_ttft_ms/saved_kv_bytes）入 CSV，
  禁用开关行为与 AR001 一致

### 3.6 FR-6 按层 KV 精度（吸收 AR002 T007）

**描述：** `llama_set_kv_type_per_layer` 最小 patch；首尾层保守 F16 策略；
prefill/decode 类型不一致返回 `Unsupported` 并统一回退（`kv-phase-fallback`）。

**验收标准：**
- Given 合法 per-layer 类型数组, When 启用, Then KV 内存较全 F16 降 ≥30%
  且质量保持 ≥97%
- Given 数组长度 ≠ n_layer, When 调用, Then 返回 2
- Given 类型不一致的相位配置, When 规划, Then `Unsupported` + 统一类型回退 + 审计

### 3.7 FR-7 自动调优 Pareto 框架（吸收 AR002 T008）

**描述：** `tune` 子命令基于真实 `BenchmarkRunner` 搜索
budget/block/top-k/head-adaptive/KV/chunk 配置，输出 Pareto 前沿与
`recommended.conf`；单候选失败记录原因不终止；推荐配置必须过质量门禁。

**验收标准：**
- Given 真实模型 + 搜索空间, When tune, Then 前沿上每点可追溯到真实
  raw.csv 样本（无代理指标）
- Given 全部候选未过质量门禁, When tune, Then 明确报告不可达原因而非输出空推荐

### 3.8 FR-8 复现框架：脚本、README、双路径（吸收 AR002 T009/T010）

**描述：** `scripts/setup.sh`（拉取并 patch 固定 commit llama.cpp、构建）、
`scripts/download_data.sh`、`scripts/run_experiments.sh`（CPU/CUDA 矩阵）、
`scripts/sync_gitee.sh`（≤40KB 批次同步工具）；README 与
`docs/experiments.md` 写清模型下载、SHA256、构建、实验命令、CUDA 路径。

**验收标准：**
- Given 干净克隆 + README 步骤, When 在 WSL CPU 与服务器 CUDA 各执行一次,
  Then 构建通过（ctest 全过）且 `run_experiments.sh` 退出码 0 并产出 raw.csv/report.md
- Given 任一增量变更, When 执行 sync_gitee.sh, Then 同步成功且远端与本地
  master sha 一致（沿用 raw receive-pack 方案，禁用 `Expect: 100-continue`）
- Given README 审阅, When 检查实验配置表, Then 含模型、量化、SHA256、线程、
  n_ctx、预算矩阵等全部大模型实验配置项

### 3.9 FR-9 质量保障与回归（吸收 AR002 T011）

**描述：** dense 等价、无 mock 硬门禁、真实模型集成测试、PPL 综合指标、
全部新能力的消融与回退用例。

**验收标准：**
- Given 全部新开关关闭, When 运行全量测试, Then 与 AR002 行为一致（回归零破坏）
- Given 无模型环境, When ctest, Then 真实模型用例显式 SKIP 且无伪造通过
- 质量综合指标 = 0.4×NIAH + 0.4×QA + 0.2×PPL，每个主实验配置均输出

### 3.10 FR-10 主实验矩阵与性能迭代（吸收 AR002 T012/T013）

**描述：** 8K/4K × CPU/CUDA × {dense, AR002 全局 top-k, AR003 head 自适应}
对照矩阵；未达标项迭代稀疏率/精度/复用策略直至达标或给出不可达分析。

**验收标准：**
- Given qwen2.5-1.5b-instruct q4_k_m 真实模型, When 8K 主实验, Then
  head 自适应 TTFT ≥3.5× dense 且质量保持 ≥97%
- Given 同上 4K, When 实验, Then TTFT ≥2.5×
- 实验结果含 raw.csv + report.md + env.json（模型/数据 SHA、线程、命令、环境）

### 3.11 FR-11 架构与研究文档（吸收 AR002 T014）

**描述：** `docs/research.md`：head 级稀疏、离线 head 分类、二维自适应、
风险控制的设计依据；与 MInference/RazorAttention/H2O/StreamingLLM/PagedAttention
的核心思想对比与本项目取舍；创新点总结。

**验收标准：**
- Given docs/research.md 审阅, When 检查对比条目, Then 覆盖 ≥5 个 SOTA 项目
  （MInference/RazorAttention/H2O/StreamingLLM/PagedAttention 等），每项含
  "核心思想 → 对本项目参考价值 → 采纳/不采纳理由"三段式
- Given 文档中的创新点声明, When 对照实验报告, Then 每个创新点有对应
  raw.csv + report.md + env.json 数字佐证

### 3.12 FR-12 Review、ST 验收与 Gitee 交付（吸收 AR002 T015–T017）

**描述：** 代码规范审查（Status/RAII/命名/警告/覆盖）、ST 端到端验收、
阶段性 Gitee 同步（创新核心完成后一次、复现框架后一次、实验完成后一次）、
release tag。

**验收标准：**
- Given 全部开发任务完成, When 执行 review, Then Status/RAII/命名/警告/
  测试覆盖复核全部通过后才进入 ST
- Given 真实模型环境, When 执行 ST 端到端验收, Then 主验收指标（≥3.5x/
  ≥2.5x/≥97%）实测通过并归档验收记录
- Given 任一阶段性同步点（T006/T012/T014 后）, When 同步完成, Then 远端
  master 与本地 sha 一致、临时分支零残留
- Given ST 通过的 commit, When 打 release tag, Then tag 指向该 commit 且
  远端可见

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 8K TTFT 加速 | ≥3.5× dense（qwen2.5-1.5b q4_k_m，CPU 路径为主）|
| 性能 | 4K TTFT 加速 | ≥2.5× dense |
| 性能 | 共享前缀二次请求 | ≥5× |
| 内存 | KV 占用 | 较全 F16 dense 降 ≥30% |
| 质量 | 综合保持率 | ≥97%（0.4×NIAH + 0.4×QA + 0.2×PPL）|
| 等价性 | 全开关关闭 | logits 与 dense max abs diff < 1e-5 |
| 可靠性 | patch 缺失/非法输入 | 明确错误码 + 安全回退，绝不半写状态 |
| 内存 | 额外开销 | ≤20%（不含可选 draft/profile）|
| 复现 | 干净克隆 | CPU（WSL）与 CUDA（服务器）均可按 README 复现 |

## 5. 约束与假设

**约束：**
- 无 mock：全部延迟/质量数据来自真实 llama.cpp + GGUF 推理；测试无模型时
  显式 SKIP
- llama.cpp 固定 commit `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32`；上游 patch
  最小、可审查、可禁用、关闭时完全恢复原路径
- C++17 / CMake ≥3.20 / `-Wall -Wextra -Werror`；Status 返回、无跨边界异常
- 公共接口与 CLI 源码级向后兼容；新能力 opt-in
- Gitee 同步受企业代理 BDWAF 限制：单 POST ≤40KB，同步走
  carrier 批次 + raw receive-pack（禁用 `Expect: 100-continue`）
- flash attention 下不支持 per-head gather（首版限制，文档记录）

**假设：**
- 测试模型 qwen2.5-1.5b-instruct q4_k_m GGUF 可获取（SHA256 记录于 README）
- 服务器具备 CUDA 工具链与 ≥16GB 显存（7B 级模型 8K 实验可选项）
- head profile 的探测参数对同族模型有效；跨模型需重新探测（工具支持）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| retrieval head | echo/induction 探测得分高于阈值的注意力头，需全量 KV |
| head profile | 离线探测输出的 JSON 层×头分类与得分 |
| per-head top-k | 每个注意力头独立的 KV gather 行集合 |
| risk tighten | 质量风险触发的预算加密动作 |
| carrier 批次 | Gitee 同步中承载 blob 子集的临时分支推送单元（≤40KB）|
