# [AR001] 需求设计说明书：端侧 LLM Prefill 架构创新

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| AR 主题 | edge-prefill-arch |
| 关联 SR | 无（新增仓库级 AR） |
| 日期 | 2026-10-20 |
| 状态 | Draft |
| 仓库 | https://gitee.com/liu-xingyan04/edge-prefill |

## 1. 背景与目标

端侧大模型的首字延迟（TTFT）主要由 prefill 阶段决定。与云端不同，端侧场景具有
batch=1、prompt 中等长度（2K–8K）、CPU/NPU 算力与内存带宽受限、功耗敏感等特点。
现有服务端稀疏注意力方案在短中上下文往往因索引开销失效，直接移植无法满足端侧需求。

本 AR 在空仓库上建立一个**真实可运行、可复现、可扩展**的 C++17 prefill 优化框架，
以 llama.cpp/GGUF 为唯一推理后端，不做任何 mock，不用伪造数据或计时。通过组合
多种互补机制，在保持任务质量的前提下显著降低 8K prompt 的 TTFT，并提供从干净
克隆到完整实验复现的一键流程。

**核心目标：**

1. 真实加载 GGUF 模型，真实执行 prefill/decode，所有指标来自真实推理。
2. 组合五类机制：块稀疏上下文选择、查询感知 token 削减、分块 prefill + 前缀 KV 复用、
   相位感知 KV 量化、prefill-only FFN 跳过。
3. 8K prompt 上 TTFT 相对 dense baseline **≥ 2.0x**，质量保持 **≥ 97%**（按第 4 节口径）。
4. 同步至 Gitee，README/脚本/配置完整，可从干净环境复现实验。

## 2. 需求范围

**In Scope：**
- C++17/CMake 工程，核心库 + CLI + 测试 + 基准。
- 固定版本 llama.cpp 集成（FetchContent 或子模块），允许通过小型 patch 增加
  prefill-only 运行时能力；patch 必须可复现、可审查。
- GGUF 模型加载、分词、batch 构造、KV 缓存、prefill、decode、性能计时。
- 多信号块级上下文选择（查询相关性、块多样性、位置先验、可选 draft 语义信号）。
- 位置保持的稀疏 prefill：保留原始 position，未选 token 不进入计算与 KV。
- 分块 prefill、前缀 KV 复用（Radix/Trie）。
- 可配置 KV 精度（f16/q8_0/q4_0）与相位感知策略。
- prefill-only FFN 跳过（运行时 mask，decode 全量）。
- 真实基准：NIAH 检索、抽取式 QA、语言建模困惑度；dense baseline 对照。
- 指标：TTFT、端到端总延迟（固定 decode token 数）、prefill tokens/s、KV 内存、
  质量保持率、Pareto 表。
- 复现脚本、模型/数据校验和、环境说明、结果归档。
- Gitee 仓库 README、LICENSE、AGENTS.md、CI（至少构建 + 单测）。

**Out of Scope：**
- 自研 GPU/NPU 稀疏注意力 kernel（首版不做；块稀疏通过不计算未选块实现）。
- 模型训练、微调、蒸馏。
- 移动端 NPU 后端（Qualcomm/Apple/MTK）适配；首版 Linux x86_64 CPU 为主，CUDA 可选。
- Decode 阶段算法优化（仅报告端到端总延迟）。
- 多请求连续批处理/服务化调度（保留接口，首版聚焦单请求低延迟）。

## 3. 功能需求

### FR-1 真实 llama.cpp/GGUF 后端

**描述：** 框架必须通过 llama.cpp 加载真实 GGUF 并执行推理。

**触发条件：** CLI 指定 `--model path.gguf`。

**期望行为：** 初始化 backend、model、context、vocab；支持 tokenize、batch decode、
logits 采样、KV 状态读写；输出模型元信息。

**异常处理：** 模型不存在/格式不匹配/上下文不足/OOM 时给出明确错误码与非零退出，
不得静默降级或返回伪造结果。

**验收标准：**
- Given 真实 Qwen2.5-1.5B-Instruct Q4_K_M GGUF 与固定 prompt，When 以贪心解码运行
  `edge-prefill run --max-new-tokens 32`，Then 退出码为 0、输出非空且 token 数达到
  上限，并且 FR-9 的任务级判分可执行。
- Given 非法路径，When 运行，Then 返回非零退出码和可诊断错误。

### FR-2 多信号块级上下文选择

**描述：** 将 prompt token 划分为固定大小块（默认 32），为每块计算重要性并选择预算内块。

**触发条件：** `--budget <0..1>` 且 prompt 长度超过阈值。

**期望行为：**
- 信号至少包括：查询词法相关性（IDF/BM25 类）、块内信息多样性/新颖度、
  位置先验（attention sink + 最近窗口）、可选 draft 模型语义相似度。
- 必选块：首个 sink 块、最后窗口块、与查询高重叠块；避免全选或全弃。
- 输出确定：相同输入、相同种子、相同配置得到相同选择。
- 选择结果可导出 JSON（块索引、分数、token 数）用于审计。

**异常处理：** budget 越界、块大小非法返回 `InvalidArgument`；prompt 过短时回退 dense 并记录原因。

**验收标准：**
- Given 8K prompt 与 budget=0.5，When 选择，Then 保留 token 数在目标 ±5% 内，
  且必选块全部命中。
- Given 相同输入重复运行 10 次，Then 选择集合完全一致。

### FR-3 位置保持的稀疏 Prefill

**描述：** 仅将选中 token 送入 llama_decode，但保留其原始位置编号。

**触发条件：** 启用选择器且 budget < 1.0。

**期望行为：**
- 使用原始 `pos` 构造 batch；未选 token 不产生 KV。
- 后续 decode 能对稀疏 KV 正常注意力；位置语义与 dense 一致。
- budget=1.0 时与 dense 路径数值等价（允许浮点容差）。
- 记录实际参与计算的 token 数、KV 占用。

**异常处理：** 位置越界、重复位置、空选择返回 `InvalidArgument`；调用方可选择回退 dense 重算。

**验收标准：**
- Given budget=1.0，When 与 dense baseline 比较 logits，Then max abs diff < 1e-5。
- Given budget=0.5，When 执行 prefill + 8 token decode，Then 不崩溃且输出可读。

### FR-4 前缀 KV 复用

**描述：** 对重复的 system prompt/多轮前缀复用已计算 KV。

**触发条件：** 新请求与缓存条目存在最长公共 token 前缀。

**期望行为：**
- Radix/Trie 管理前缀；命中的 token 不重复 prefill。
- 支持多条目、LRU 淘汰、最大缓存 token 数配置。
- 命中率、复用 token 数、节省时间写入报告。

**异常处理：** 缓存损坏/模型变更/上下文不足时安全失效并重算。

**验收标准：**
- Given 同一 4K 前缀请求两次，When 第二次运行，Then prefill token 数减少 ≥ 95%，
  TTFT 相对无缓存基线下降 ≥ 5x，输出与无缓存一致（贪心解码逐 token 相同）。

### FR-5 自适应分块 Prefill

**描述：** 将选中 token 切分为可配置 chunk，逐步构建 KV。

**触发条件：** `--chunk-size > 0`。

**期望行为：**
- chunk 大小自适应（基于 n_ubatch、内存、prompt 长度）。
- 支持 chunk 间计时与累计 TTFT。
- 不改变最终 KV 语义（与一次性 prefill 在容差内一致）。

**异常处理：** chunk 大小大于上下文/小于 1 时归一化或报错。

**验收标准：**
- Given 8K prompt，When chunk-size=512，Then 完成 prefill 且 logits 与一次性 prefill
  在容差内一致。
- 报告输出每 chunk 耗时与 token 数。

### FR-6 Prefill-only FFN 跳过

**描述：** 在 prefill 阶段按 mask 跳过指定层的 FFN 计算，decode 阶段恢复全量。

**触发条件：** `--skip-ffn-layers <mask/list>` 且当前为 prefill。

**期望行为：**
- 通过维护的 llama.cpp 小型 patch 提供运行时开关，不修改模型文件。
- 跳过层仅影响 prefill 的 hidden state，仍生成该层 attention/KV；decode 全量。
- mask=0 时行为与上游完全一致。
- 记录跳过层数与理论 FLOPs 节省。

**异常处理：** 不支持的架构/模型时报错或显式禁用；不得静默产生错误结果。

**验收标准：**
- Given mask=0，When 对比上游，Then logits 完全一致（或 < 1e-6）。
- Given 跳过 25% 层且其余配置相同，When 运行 8K 基准，Then prefill tokens/s 提升
  ≥ 10%，且质量按第 4 节口径评估并报告。

### FR-7 相位感知 KV 量化

**描述：** 允许为 prefill/decode 配置 KV cache 类型，并报告质量/延迟权衡。

**触发条件：** `--kv-type f16|q8_0|q4_0`（K/V 可分别配置）。

**期望行为：**
- 通过 llama_context_params.type_k/type_v 设置。
- 规划器可根据相位选择保守/激进配置；首版支持统一配置，并接受显式相位配置：
  当 prefill/decode 类型不同时安全拒绝（Unsupported）并回退统一类型，记录
  `kv-phase-fallback`；真正的双 context handoff 列为后续 AR 扩展点。
- 报告 KV 内存占用与质量变化。

**异常处理：** 后端不支持该类型时回退 f16 并警告。

**验收标准：**
- Given q8_0，When 8K 推理，Then KV 内存相对 f16 下降 ≥ 40%，且质量保持 ≥ 97%。
- Given 非法 `--kv-type`，When 运行，Then 返回非零退出码与可诊断错误信息。

### FR-8 策略规划器

**描述：** 根据 prompt 长度、预算、硬件、模型与质量模式组合上述机制。

**触发条件：** 每次请求/基准运行。

**期望行为：**
- 模式：`latency`、`balanced`、`quality`、`reuse`。
- 输出实际启用的 pass、预算、chunk、skip mask、KV 类型及理由。
- 配置可序列化为 JSON，支持命令行覆盖。

**异常处理：** 冲突配置时给出明确优先级与警告。

**验收标准：**
- Given 8K prompt + balanced，Then 自动选择非 dense 组合并生成可审计计划。
- Given 2K prompt，Then 若预估收益为负则回退 dense。

### FR-9 真实基准与报告

**描述：** 提供可复现的真实模型基准，覆盖质量与延迟。

**触发条件：** `edge-prefill bench --config ...`。

**期望行为：**
- 数据集：NIAH 检索（多 needle/深度）、抽取式 QA（真实文档）、WikiText 困惑度。
- 基线：同模型、同线程、同 KV 类型的 dense llama.cpp prefill。
- 指标：TTFT、端到端总延迟（固定 decode 步数）、prefill tokens/s、KV 内存、
  exact match/F1/perplexity、质量保持率。
- 输出 CSV + Markdown；包含 warmup、重复次数、均值/中位数/标准差、环境信息。
- 支持服务器大规模复现（可配置模型/数据集路径、GPU offload、线程数）。

**异常处理：** 数据集缺失时给出下载命令；不得用随机分数填充。

**验收标准：**
- Given 干净环境与模型，When 执行一键脚本，Then 生成包含 baseline 与优化组的报告。
- 报告中的每个数字都能追溯到原始 CSV 与命令。

### FR-10 可复现与交付

**描述：** 从 Gitee 克隆后可便捷复刻。

**期望行为：**
- `README.md`：原理、架构、构建、模型下载、快速开始、完整实验、结果解读、限制。
- `scripts/setup.sh`、`scripts/download_models.sh`、`scripts/run_bench.sh`。
- 固定 llama.cpp commit、模型 SHA256、数据集 SHA256、编译器/系统信息。
- `AGENTS.md` 记录开发/测试/风格约定。
- CI 至少完成 Linux CPU 构建与单测。

**验收标准：**
- Given 干净 Ubuntu + 依赖，When 按 README 执行，Then 构建成功并跑通 smoke test。
- Given 模型路径，When 运行基准脚本，Then 产出报告。

## 4. 非功能需求与验收口径

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 8K TTFT 相对 dense | ≥ 2.0x |
| 质量 | 8K 综合保持率 | ≥ 97%（NIAH + QA + PPL 归一化聚合） |
| 性能 | 4K TTFT 相对 dense | ≥ 1.5x（目标） |
| 复用 | 4K 前缀二次请求 TTFT | ≥ 5x（目标） |
| 内存 | 峰值 RSS 额外开销 | ≤ 20%（不含可选 draft 模型） |
| 可靠性 | 崩溃/未捕获异常 | 0（基准与测试集） |
| 确定性 | 相同配置重复运行 | 选择与贪心输出一致 |
| 可维护性 | 编译警告 | `-Wall -Wextra -Werror` 通过 |
| 可测试性 | 核心模块单测覆盖 | 关键路径均有 CTest 用例 |
| 可移植性 | 平台 | Linux x86_64 CPU 必测；CUDA 可选 |

**质量口径：** 质量保持率 = 优化组指标 / dense 基线指标。NIAH 用准确率，QA 用
EM/F1，PPL 用 `exp(-(logp_opt - logp_base)/tokens)` 归一化；最终按预注册权重聚合，
权重与原始结果一并写入报告，禁止事后挑选有利指标。

**测量口径：** TTFT 从 tokenize 完成后开始计时到首 token logits 可用；端到端总延迟
包含固定 32 步 decode；每组 warmup ≥ 1 次、重复 ≥ 5 次，报告均值与中位数；
CPU 固定线程数，记录 CPU 型号与频率策略。

## 5. 约束与假设

**约束：**
- C++17；CMake ≥ 3.20；llama.cpp 版本固定，patch 必须最小且可审查。
- 不引入闭源依赖；不依赖云服务；运行时不联网。
- 不使用 mock 模型/伪造计时/硬编码结果。
- 首版以 Qwen2.5-1.5B-Instruct Q4_K_M 为主模型；Qwen2.5-0.5B 可选作 draft；
  Llama-3.1-8B / Qwen3-4B 用于泛化验证。
- Gitee 仓库为主发布源；GitHub 仅作为上游依赖来源。

**假设：**
- 端侧目标为 4K–8K prompt、单请求、CPU/NPU 推理；首版在 x86_64 CPU 上验证。
- 真实模型与数据集可由脚本下载；服务器复现时可使用更大模型/GPU。
- 质量评估使用公开可复现任务；NIAH 为确定性生成，但由真实模型回答与判分。
- FFN 跳过属于有损近似，必须与质量指标一起报告，不得单独宣称加速。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| Prefill | 对完整 prompt 并行计算并构建 KV cache 的阶段 |
| TTFT | Time To First Token，首 token 可用时间 |
| Dense baseline | 不启用任何优化的 llama.cpp 全量 prefill |
| 块稀疏选择 | 以 token block 为单位选择参与 prefill 的子集，未选块不计算、不建 KV |
| 位置保持 | 选中 token 仍使用原始 position id，保证 RoPE 与因果语义 |
| 相位感知 | 对 prefill 与 decode 采用不同的稀疏/精度/层策略 |
| FFN 跳过 | prefill 阶段跳过指定层的前馈网络，decode 恢复 |
| 质量保持率 | 优化组质量指标相对 dense baseline 的比例 |

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 50% 块选择导致 QA 质量下降 | 达不到 97% | 多信号 + 必选块 + 预算自适应；按任务校准；提供 quality 模式 |
| FFN 跳过损伤质量 | 质量不达标 | 限制跳过比例/层选择；与稀疏预算联合调优；默认关闭 |
| llama.cpp patch 随上游变化失效 | 构建失败 | 固定 commit + 独立 patch + CI 校验；提供上游回退开关 |
| CPU 上 8K 实验耗时 | 迭代慢 | 支持 CUDA 构建；服务器复现脚本；缓存/并行实验 |
| 度量偏差 | 结论不可信 | 预注册指标、固定种子、重复运行、原始 CSV 归档 |
| 位置保持稀疏 KV 与实现细节冲突 | 结果错误 | dense 等价性测试 + 逐 token 对齐测试 + 断言 |

## 8. 交付物

- 源码：`include/`、`src/`、`tests/`、`benchmarks/`。
- 构建与复现：`CMakeLists.txt`、`cmake/`、`scripts/`、`patches/`。
- 文档：`README.md`、`AGENTS.md`、`docs/architecture.md`、`docs/experiments.md`、
  `docs/research.md`。
- 规格：本 `srs.md`、`design.md`、`tasks.md`。
- 实验：`benchmarks/results/` 下的 CSV/Markdown（含环境与命令）。
- Gitee 仓库同步与 release tag。
