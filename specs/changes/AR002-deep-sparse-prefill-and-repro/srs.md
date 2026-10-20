# [AR002] 需求设计说明书：深度稀疏 Prefill 与自适应复用创新

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| AR 主题 | deep-sparse-prefill-and-repro |
| 关联 SR | 无（仓库级增量 AR，基于 AR001 架构基线） |
| 日期 | 2026-10-20 |
| 状态 | Draft |
| 仓库 | https://gitee.com/liu-xingyan04/edge-prefill |

## 1. 背景与目标

AR001 已建立 EdgePrefill 的真实可运行基线：llama.cpp/GGUF 后端、块级稀疏选择、
前缀 KV 复用、分块 prefill、prefill-only FFN 跳过、KV 精度配置与真实基准。当前
8K NIAH 实验已达到约 2.25x TTFT 加速，但整体质量与加速余量仍未达到 AR002 的更
高目标，且跨请求复用、attention 级稀疏、动态精度调度和自动调优能力尚未形成体系。

AR002 的目标是在 AR001 基线上继续做**深度架构创新**，而不是简单参数调优：

1. 将稀疏能力从“块级选择”深化为“块级 + attention 级 top-k”的分层稀疏 Prefill。
2. 将前缀复用从单请求扩展到**多轮对话与跨请求共享前缀**的真实 KV 复用。
3. 将 KV 精度与 FFN 层跳过从静态配置升级为**运行时自适应调度**。
4. 将基准从固定实验升级为**自动参数搜索 + 质量-延迟 Pareto 前沿**。
5. 提供 CPU 与 CUDA/服务器双路径的一键复现能力，并同步 Gitee。

### 1.1 核心目标

1. 8K prompt 上，TTFT 相对 dense baseline **≥ 2.5x**。
2. 质量保持率 **≥ 97%**，口径延续 AR001：`0.4*NIAH + 0.4*QA + 0.2*PPL`。
3. 所有性能、质量、内存数据均来自真实 llama.cpp/GGUF 推理，**无任何 mock 或伪造结果**。
4. 公共接口与 CLI 保持向后兼容，新能力以 opt-in 方式提供。
5. 交付可从干净 Gitee 克隆复现的实验工程，覆盖 CPU 必测与 CUDA 服务器复现。

### 1.2 与先进项目的参考关系

AR002 不照搬现有项目，而是吸收其核心思想并适配端侧单请求、CPU/轻量 CUDA 场景。

| 项目/方向 | 核心思想 | 对本项目的参考价值 | AR002 的适配方式 |
|-----------|----------|--------------------|------------------|
| MInference / FlashPrefill | 长上下文 prefill 中存在可预测的动态稀疏 attention 模式，可用模式识别 + 稀疏计算降低延迟 | 证明 prefill 阶段存在可利用的稀疏性，但其 GPU kernel 与在线索引成本不适配端侧 | 在 llama.cpp 层引入可审查的 attention top-k patch，结合块级选择，不实现自研 GPU kernel |
| CLAA / FastKV | 跨层 attention 聚合可用于识别重要 token，减少深层冗余计算 | 强调“层间信号”而非单层启发式，可提升选择稳定性 | 将多信号选择扩展为跨层/多信号融合，并支持 attention 级稀疏调度 |
| Quest / SnapKV / H2O | 查询感知的 KV 选择、重要 token 保留和淘汰能降低长上下文成本 | 适合作为块内/块间重要性判据，但主要面向 decode 或在线 KV 压缩 | 用于 prefill 阶段的查询感知 top-k 与必选块策略，保持位置语义 |
| vLLM PagedAttention / Prefix Cache | 分页 KV 与前缀复用显著降低重复 prefill | 证明共享前缀复用是高价值优化点 | 引入跨请求/多轮的 Radix + 分页 KV 复用，适配单机端侧 |
| Sarathi-Serve | 分块 prefill 可改善长 prompt 的调度与内存压力 | 支持稳定 prefill 与缓存复用 | 继续作为基础机制，与稀疏和复用联合调度 |
| llama.cpp | 端侧真实推理事实标准 | 保持真实可运行与可复现的唯一后端 | 通过最小 patch 扩展运行时稀疏能力，保持可禁用 |

### 1.3 差异化创新点

1. **分层稀疏 Prefill（Hierarchical Sparse Prefill）**
   - 第一层：块级选择，继承 AR001 的多信号策略。
   - 第二层：attention 级 top-k / 动态稀疏，在保留因果与位置语义的前提下进一步减少无效 attention 计算。
   - 目标是让稀疏策略同时具备工程可运行性与研究创新性。

2. **跨请求与多轮前缀 KV 复用**
   - 从单条请求的前缀命中扩展到多轮对话、多请求共享 system prompt 与公共上下文。
   - 通过 Radix + 分页 KV 与真实 state 快照实现复用，避免重复 prefill。

3. **自适应精度与层调度**
   - 根据层位置、上下文长度、缓存命中和当前负载动态选择 KV 精度与 FFN 跳过策略。
   - 在质量门禁约束下自动权衡延迟、内存与质量。

4. **自动调优与 Pareto 基准**
   - 自动搜索 budget、block size、top-k、KV 精度、FFN mask 等参数组合。
   - 输出质量-延迟 Pareto 前沿和推荐配置，而不是仅报告单一最优结果。

---

## 2. 需求范围

**In Scope：**

- 分层稀疏 Prefill：
  - 块级多信号选择。
  - Attention 级 top-k / 动态稀疏注意力，基于最小可审查 llama.cpp patch。
  - 位置保持与 dense 回退。
- 跨请求与多轮前缀 KV 复用：
  - Radix/分页缓存。
  - 多请求共享前缀。
  - 真实 KV state 快照与恢复。
- 自适应 KV 精度与层调度：
  - 按层/相位/负载动态选择 KV 类型。
  - 动态 FFN 层跳过策略。
- 自动调优与 Pareto 基准：
  - 参数搜索。
  - 质量-延迟 Pareto 输出。
  - 推荐配置生成。
- 服务器级复现与交付：
  - CPU 必测。
  - CUDA/GPU 服务器复现路径。
  - README、实验文档、复现脚本、结果归档。
  - Gitee 同步与 tag。
- 质量保障与回归：
  - 无 mock 硬门禁。
  - budget=1.0 / 稀疏关闭路径与 dense 数值等价。
  - 关键功能单测与真实模型集成测试。

**Out of Scope：**

- 模型训练、微调、蒸馏。
- 自研 GPU/NPU attention kernel。
- 多请求服务化调度系统。
- Decode 阶段深度优化（仅报告端到端总延迟）。
- 移动端 NPU 适配（Qualcomm/Apple/MTK）。
- 对外提供稳定 ABI 承诺；公共 API 可演进但需保持源码级向后兼容。

---

## 3. 功能需求

### FR-1 分层稀疏 Prefill

**描述：** 在块级选择的基础上，引入 attention 级 top-k / 动态稀疏注意力，进一步减少无效 prefill 计算。

**触发条件：**
- prompt 长度超过 `min_dense_tokens`。
- `budget < 1.0`。
- 用户显式启用 attention 级稀疏或由 planner 自动选择。

**期望行为：**
1. 第一层仍执行 AR001 的块级多信号选择。
2. 第二层对参与计算的块启用 attention 级 top-k 或动态稀疏策略。
3. 稀疏策略必须保持位置语义与因果性。
4. Attention 级稀疏必须可通过配置完全关闭，关闭后行为回到块级稀疏或 dense。
5. Planner 必须根据上下文长度、质量要求和硬件条件决定是否启用该层。
6. 输出可审计的稀疏计划与每层稀疏统计。

**异常处理：**
- 不支持的模型/架构或 patch 缺失时，禁用 attention 级稀疏并回退到块级或 dense，同时在 `Plan.reasons` 中记录。
- 质量门禁失败时，自动回退到更保守策略。

**验收标准：**
- Given 8K prompt 与 attention 级稀疏启用，When 运行真实模型基准，Then TTFT 相对 dense ≥ 2.5x，质量保持率 ≥ 97%。
- Given attention 级稀疏关闭，When 与**同一配置下未启用 attention patch 的执行路径**对比，Then 贪心解码逐 token 相同，且 logits max abs diff < 1e-5。
- Given budget=1.0 且所有稀疏关闭，When 与 dense 对比 logits，Then max abs diff < 1e-5。

### FR-2 跨请求与多轮前缀 KV 复用

**描述：** 支持多轮对话与多个请求之间共享相同前缀的真实 KV 复用，避免重复 prefill。

**触发条件：**
- 新请求与缓存中的历史请求存在公共 token 前缀。
- 前缀长度超过可配置阈值。
- KV 签名兼容（模型、上下文长度、KV 类型一致）。

**期望行为：**
1. 以 Radix/分页方式管理公共前缀。
2. 支持同一会话内多轮追加，以及不同请求间共享 system prompt / 长上下文。
3. 命中前缀时只 prefill 后缀，不重复计算已缓存 KV。
4. 支持 LRU 淘汰、容量限制与同进程内线程安全的缓存访问；不提供跨请求并行调度系统。
5. 输出命中率、复用 token 数、节省时间与内存占用。
6. 复用后的输出必须与无缓存路径一致或在容差内等价。

**异常处理：**
- KV 签名不匹配、缓存损坏或上下文不足时，安全失效并回退全量 prefill。
- 不允许静默使用错误 KV state。

**验收标准：**
- Given 同一 4K system prompt 的两个不同请求，When 第二个请求执行，Then prefill token 数减少 ≥ 90%，TTFT 相对无缓存基线下降 ≥ 5x。
- Given 多轮对话共享前缀，When 追加新轮次，Then 输出与无缓存全量 prefill 一致（贪心解码逐 token 相同）。
- Given 不同模型或 KV 类型，When 查询缓存，Then 不得复用并记录 miss。

### FR-3 自适应 KV 精度与层调度

**描述：** 根据上下文长度、层位置、缓存命中与质量要求，动态选择 KV 精度与 FFN 跳过策略。

**触发条件：**
- 启用 `adaptive` 模式或 planner 判断可收益。

**期望行为：**
1. 支持按层配置 KV 精度，而不仅是全局 `type_k/type_v`。
2. 支持根据层深/位置动态决定是否跳过 FFN。
3. 在质量保持率约束下优化延迟与内存。
4. 所有自适应决策必须可审计，并记录在 `Plan.reasons` 或 JSON 计划中。
5. 必须提供保守/激进/质量优先等策略档位。

**异常处理：**
- 不支持的 KV 类型或架构时回退到保守配置。
- 质量门禁失败时自动降级到更安全策略。

**验收标准：**
- Given 8K prompt 与自适应精度策略，When 与 dense f16 对比，Then KV 内存下降 ≥ 30%，质量保持率 ≥ 97%。
- Given 质量优先模式，When 运行，Then 不得启用会导致质量保持率 < 97% 的激进度或层跳过策略。
- Given 关闭自适应调度，When 运行，Then 行为与 AR001 静态配置一致。

### FR-4 自动调优与 Pareto 基准

**描述：** 提供自动参数搜索与质量-延迟 Pareto 前沿分析，替代单一固定配置实验。

**触发条件：**
- `edge-prefill tune` 或 `bench --sweep`。

**期望行为：**
1. 支持搜索：
   - budget
   - block size
   - attention top-k / 稀疏率
   - KV 精度
   - FFN 层跳过 mask
   - chunk size
2. 每组配置在真实模型上执行完整基准，不使用代理指标替代真实推理。
3. 输出每个配置的 TTFT、E2E、KV 内存、质量与质量保持率。
4. 生成质量-延迟 Pareto 前沿与推荐配置。
5. 支持将推荐配置导出为可复用 conf 文件。
6. 记录完整环境、命令与原始结果，保证可追溯。

**异常处理：**
- 搜索过程中单个配置失败时记录失败原因，不终止整体搜索。
- 不得用失败或缺失数据填充结果。

**验收标准：**
- Given 8K 数据集与允许的搜索空间，When 执行自动调优，Then 输出至少一个满足 TTFT ≥ 2.5x 且质量保持率 ≥ 97% 的配置，或明确报告不可达原因。
- Given 自动调优结果，When 导出推荐配置并重新运行，Then 结果可复现且与搜索阶段一致。

### FR-5 服务器级复现与交付

**描述：** 从干净 Gitee 克隆出发，可在 CPU 与 CUDA/服务器环境中便捷复现主要实验。

**触发条件：**
- 用户按 README 或脚本执行。

**期望行为：**
1. `README.md` 包含：
   - 项目原理与创新点。
   - 架构说明。
   - CPU 与 CUDA 构建方法。
   - 模型下载与校验。
   - 快速开始。
   - 完整实验复现步骤。
   - 结果解读与限制。
2. 提供：
   - `scripts/setup.sh`
   - `scripts/download_models.sh`
   - `scripts/run_bench.sh`
   - `scripts/run_tune.sh`
3. 支持记录 CPU/GPU、线程数、模型与数据 SHA256、命令、环境。
4. 支持一键复现 8K 主实验。
5. 支持 CUDA offload 配置，但不强制依赖 GPU。
6. 结果归档包含原始 CSV、报告、环境信息与推荐配置。

**异常处理：**
- 依赖缺失时给出明确安装指引。
- 模型或数据缺失时提示下载命令，不得用假数据代替。

**验收标准：**
- Given 干净 Ubuntu + 依赖，When 按 README 执行，Then CPU 构建与 smoke test 成功。
- Given 支持 CUDA 的服务器，When 按 README 启用 GPU，Then 可运行主实验并生成报告。
- Given 干净克隆，When 执行一键脚本，Then 能复现主实验结果：环境与命令可追溯，关键指标在相同 seed 与重复次数下落在已记录结果的波动范围内，且贪心输出逐 token 相同。

### FR-6 质量保障与回归门禁

**描述：** 建立无 mock、真实推理、dense 等价和质量门禁的硬性保障。

**触发条件：**
- 任何代码合入、基准发布或结果归档。

**期望行为：**
1. 禁止 mock 模型、伪造计时、硬编码质量分数或合成结果冒充真实推理。
2. `budget=1.0` 且关闭所有稀疏/跳过时，必须与 dense baseline 数值等价。
3. 关键模块必须有单测与真实模型集成测试。
4. 每次发布结果必须附带原始 CSV、环境信息与命令。
5. 质量保持率 < 97% 的配置不得被推荐为默认配置。
6. 所有新增接口必须保持 AR001 兼容或有明确迁移说明。

**异常处理：**
- 回归失败时阻止推荐配置或发布结果。
- 测试缺模型时可 skip 真实模型用例，但不得伪造通过。

**验收标准：**
- Given 无模型环境，When 运行测试，Then 真实模型用例明确 skip，其余测试通过。
- Given 真实模型环境，When 运行完整测试，Then 所有测试通过，无 mock 路径。
- Given budget=1.0 与全关闭优化，When 与 dense 对比，Then logits max abs diff < 1e-5。

### FR-7 架构与接口演进

**描述：** 为分层稀疏、跨请求复用、自适应调度和自动调优提供可扩展的架构支撑，同时保持向后兼容。

**触发条件：**
- 新能力需要新增公共接口、CLI 参数或配置项。

**期望行为：**
1. 新增接口遵循现有 `Status` / RAII / 无异常公共边界约定。
2. AR001 已有 CLI 参数和配置文件必须继续可用。
3. 新能力默认不改变既有行为，必须显式 opt-in 或由 planner 在安全条件下启用。
4. 关键决策必须可审计，输出到 plan/selection JSON。
5. 代码结构保持模块化，避免把稀疏、缓存、精度和调优逻辑耦合在单一模块中。

**异常处理：**
- 不兼容配置组合必须返回 `InvalidArgument` 或给出明确警告，不得静默选择未定义行为。

**验收标准：**
- Given AR001 的配置文件，When 在 AR002 代码上运行，Then 语义不变；允许新增 warning，但该 warning 不得改变退出码、计划语义或默认配置。
- Given 新增 attention 稀疏或自适应精度参数，When 未启用，Then 不影响既有路径。
- Given 非法参数组合，When 运行，Then 返回明确错误码和可诊断信息。

---

## 4. 非功能需求与验收口径

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 8K TTFT 相对 dense | ≥ 2.5x |
| 质量 | 8K 综合保持率 | ≥ 97%（`0.4*NIAH + 0.4*QA + 0.2*PPL`） |
| 性能 | 4K TTFT 相对 dense | ≥ 1.8x（目标） |
| 复用 | 4K 共享前缀二次请求 TTFT | ≥ 5x |
| 内存 | KV 内存相对 dense f16 | ≥ 30% 下降（在质量 ≥97% 前提下） |
| 可靠性 | 基准与测试崩溃/未捕获异常 | 0 |
| 确定性 | 相同配置重复运行 | 选择、计划与贪心输出一致 |
| 可维护性 | 编译警告 | `-Wall -Wextra -Werror` 通过 |
| 可测试性 | 核心模块覆盖 | 关键路径均有 CTest 用例 |
| 真实性 | mock/伪造结果 | 严禁 |
| 可移植性 | 平台 | Linux x86_64 CPU 必测；CUDA 服务器可复现 |
| 兼容性 | AR001 配置/CLI | 源码级向后兼容 |

**质量口径：** 质量保持率 = 优化组指标 / dense 基线指标。NIAH 使用准确率，QA 使用
EM/F1，PPL 使用 `exp(-(logp_opt - logp_base)/tokens)` 归一化；最终按预注册权重
`0.4*NIAH + 0.4*QA + 0.2*PPL` 聚合，禁止事后挑选有利指标。

**测量口径：** TTFT 从 tokenize 完成后开始计时，到首 token logits 可用；端到端延迟
包含固定 32 步 decode。每组实验 warmup ≥ 1 次、重复 ≥ 5 次，报告均值与中位数，
记录线程数、CPU/GPU 型号与频率策略。

---

## 5. 约束与假设

**约束：**
- C++17；CMake ≥ 3.20；llama.cpp commit 固定，patch 必须最小且可审查。
- 所有实验必须使用真实 GGUF 模型与真实数据集；不得用 mock、代理指标或伪造结果替代。
- llama.cpp patch 必须可禁用、可复现，并在 CI 中验证可干净应用。
- 公共 API 保持源码级向后兼容，新增能力必须 opt-in。
- 首选模型仍为 Qwen2.5-1.5B-Instruct Q4_K_M；服务器复现可扩展到更大模型。
- CUDA 支持不得破坏 CPU-only 构建。
- Gitee 为唯一主发布源。

**假设：**
- 端侧目标为 4K–8K prompt、单请求、CPU/轻量 GPU 推理。
- 服务器具备足够算力执行更大规模模型与自动调优实验。
- 公开数据集可下载并校验；NIAH/QA/PPL 数据生成脚本保持确定性。
- Attention 级稀疏属于有损近似，必须与质量指标一起报告，不得单独宣称加速。

---

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 分层稀疏 Prefill | 先做块级选择，再做 attention 级 top-k/动态稀疏的两层优化 |
| Attention 级 top-k | 在 attention 计算中仅保留与当前查询最相关的 top-k token/块参与计算 |
| 跨请求复用 | 不同请求共享相同前缀时复用已计算 KV state |
| 多轮复用 | 同一会话内后续轮次复用之前轮次的 KV state |
| 自适应精度 | 根据层、相位、负载或上下文动态选择 KV 精度 |
| Pareto 前沿 | 在质量与延迟二维空间中不被其他配置同时支配的配置集合 |
| 质量保持率 | 优化组质量指标相对 dense baseline 的比例 |
| Dense baseline | 不启用任何稀疏、跳过或复用优化的全量 prefill |

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| Attention 级稀疏导致质量下降 | 无法达到 97% | 引入可配置稀疏率、top-k 下限、质量优先回退与自动调优 |
| llama.cpp patch 复杂度上升 | 维护困难 | 保持最小 patch、固定 commit、CI 校验、提供关闭开关 |
| 跨请求 KV 复用引入状态错误 | 输出错误或崩溃 | KV 签名校验、等价性测试、真实 state 快照对比 |
| 自适应策略搜索空间过大 | 实验耗时过长 | 分阶段搜索、缓存中间结果、服务器并行执行 |
| CUDA 与 CPU 行为不一致 | 结论不可信 | 双路径测试、环境记录、关键用例交叉验证 |
| 质量与延迟冲突 | 无法同时达标 | Pareto 分析、多目标优化、默认推荐仅选择满足质量门禁的配置 |

---

## 8. 交付物

- 源码：`include/`、`src/`、`tests/`、`benchmarks/`。
- 构建与复现：`CMakeLists.txt`、`cmake/`、`scripts/`、`patches/`。
- 文档：`README.md`、`AGENTS.md`、`docs/architecture.md`、`docs/experiments.md`、`docs/research.md`。
- 规格：本 `srs.md`、后续 `design.md`、`tasks.md`。
- 实验：`benchmarks/results/` 下的原始 CSV、报告、环境信息、推荐配置。
- Gitee 仓库同步与 release tag。
