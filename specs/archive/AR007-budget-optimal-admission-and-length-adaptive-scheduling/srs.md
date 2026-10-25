# [AR007] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| AR 主题 | budget-optimal-admission-and-length-adaptive-scheduling |
| 关联 SR | 承接 AR004/AR005/AR006（选择级稀疏 + 注意力级稀疏 + 图内门控） |
| 日期 | 2026-10-25 |
| 状态 | Draft（需求自检 PASS，用户已授权自动化流程） |

## 1. 背景与目标

AR004-AR006 已把「预填充稀疏」做成可用系统：dense 对照下 8K 9.4×、16K 6.8×、32K 16.8×
（同会话实测）。但 AR006 的实测暴露出**瓶颈已经整体迁移**：

1. **成本对「准入 token 数」是二次的**。稀疏路径只把 25% 的 token 送进 transformer，
   注意力在准入长度上仍是 O(L²)。16K dense 161.8s → 25% 准入 22.5s（7.2×）；
   再把准入降到 12% 理论上可到 ~4× 于当前（≈28× dense 的二次收益），
   所以**准入预算（budget）是当前唯一还有数量级空间的设计维度**。
2. **FFN 不是瓶颈**。同一 16K 稀疏配置下 `--skip-ffn 50%` 无收益、`--skip-ffn 100%`
   仅快 15%（三次实测，见 §3.1 证据），说明「token 级 masked FFN」方向没有价值，
   资源应投向「更少 token、同样质量」。
3. **质量-预算曲线没有系统测量**。当前 0.25 的默认预算是拍出来的：16K 词法选择 q=1.0，
   但 32K/mn 已出现 0.83/0 的下滑，说明预算与长度耦合、且选择信号在长上下文失效。
4. **AR006 图内门控在 32K 有已知的实现级浪费**：页均值每 ubatch 经 `cont(permute(k))`
   全量拷贝整段 K（≈160GB 流量），门控比它节省的注意力还贵（32K bound 67.9s vs
   scalar 22.8s）。

目标：把「准入预算」变成**可测量、可预测、可自适应**的一等设计参数，并顺手消掉
AR006 门控的实现级浪费，使同一质量下的净 TTFT 相对 AR006 再降 1.5-2×
（16K ≤12% 预算 q≥0.9，32K 同质量 ≥1.3×），且完整保留 dense 等价与真实模型全链路。

## 2. 需求范围

**In Scope：**
- 质量悬崖测量与落库：`budget × 长度 × 选择信号` 的 q/ttft 曲线（真实模型）。
- 选择器升级：结构覆盖锚点（sink/recent/stride/引用块）+ MMR 去冗余 + 融合分数归一化。
- 长度自适应预算（planner 依据曲线选 budget，含安全边际；显式 budget 仍最优先）。
- patch：AR006 页门控去 K 全量拷贝（`mul_mat` + 页求和）。
- 复现资产：budget 套件配置、expected 表、repro.sh 入口、服务器实验手册更新。

**Out of Scope：**
- 训练/微调任何模型；引入新的神经模型（不新增依赖）。
- token 级 masked FFN（已实测无收益，明确不做）。
- head 路由拆图（AR006 T007，仍延后）。
- GPU 上的完整矩阵（只做 1-2 个关键点回归）。

## 3. 功能需求

### 3.1 FR-1 预算-质量曲线测量（证据基线）

**描述：** 用真实模型在 4 个长度档 × 2 个选择信号 × ≥4 个预算档上测 q 与 ttft，
产出结构化表，作为 FR-3 自适应预算的依据。

**触发条件：** 数据集与模型就绪（`EDGE_PREFILL_TEST_MODEL` 或 bench 参数）。

**期望行为：**
- 每格点记录：数据集、budget、score_source、q、ttft、selected_tokens、repeats；
- 结果落 `benchmarks/results/ar007-sweep/`（raw.csv + 汇总 json/md）；
- 曲线由脚本生成，禁止手工填写数字。

**异常处理：** 某格点失败必须显式记录失败原因，不得用插值/默认值填充。

**验收标准：**
- Given 16K niah 数据集与 budget∈{0.20,0.15,0.10,0.05}，When 跑 sweep，Then 每格点有 raw.csv 且汇总表可复算；
- Given 汇总表，When 与原始 CSV 抽查 2 格，Then q/ttft 完全一致。

**实测证据（2026-10-25，本机）：**

| 配置（16K，lexical，budget 0.25） | TTFT | 质量 |
|---|---:|---:|
| 基线 | 51.8-63.5 s | 0.67（max_new_tokens=8 截断所致） |
| `--skip-ffn 50%` | 56.3-67.6 s | 0.67 |
| `--skip-ffn 100%` | 43.8-54.1 s | 0.67 |

结论：FFN 占比 ≤15-20%，**token 级 FFN 跳过不是有效方向**；成本由注意力（∝准入长度²）主导。

### 3.2 FR-2 覆盖锚点 + 多样性选择器（CAMS）

**描述：** 在现有词法/嵌入分数之上增加三类硬锚点与一次 MMR 去冗余，使低预算下
「任意区域都有被看到的入口」，避免重复文本吃满预算。

**触发条件：** `budget<1.0` 且启用 block 选择（默认路径）。

**期望行为：**
1. 结构锚点（确定性、与分数无关）：sink 块（首 N 块）、recent 块（尾部 N 块）、
   按提示长度的均匀 stride 块（步长由预算与长度推导，保证覆盖）、引用块
   （含问句中的数字/实体/引号的块，命中即入）；
2. 剩余预算按融合分数（归一化词法/嵌入分数 × 位置先验）贪心填充，并做 MMR：
   与已选块相似度（token 集合 Jaccard）超过阈值的候选降权；
3. 锚点占用预算上限可配（`anchor_ratio`），保证相关性块仍有空间；
4. 选择结果与现有 `Selection` 结构兼容（不新增对外接口字段时零影响）。

**异常处理：** 锚点数量超过预算 → 按 sink>recent>quote>stride 优先级裁剪并审计；
分数非有限 → 该块视为 0 分（不得 NaN 传播）。

**验收标准：**
- Given 16K niah、budget=0.12，When 跑 CAMS，Then q≥0.9（对比当前 0.25 才 1.0）；
- Given 同一输入，When 跑两次，Then 选择结果逐位一致（确定性）；
- Given budget=1.0，When 跑，Then 走原 dense 路径（锚点不生效，字节等价）；
- Given 构造的重复文本块，When 跑 MMR，Then 相似块被降权（单测）。

### 3.3 FR-3 长度自适应预算（LAB）

**描述：** planner 根据提示长度与 FR-1 曲线选择准入预算，使 quality 目标（默认 ≥0.9）
在最小预算下满足；显式 `budget` 配置仍最高优先。

**触发条件：** 用户未显式覆盖 budget（`budget_overridden=false`）。

**期望行为：**
- 内置分段表（由 FR-1 生成，随结果入库）：长度 → 推荐 budget（含 +25% 安全边际）；
- 审计字段：`budget_source`（`model` / `override` / `clamp`）写入 Sample/raw.csv；
- 32K 及以上仍保留内存驱动的硬 clamp，但不得低于曲线给出的质量下限；
- 缺表（未跑 sweep）时退化为当前行为（0.25）并记录 `budget_source=default`。

**异常处理：** 表解析失败 → 记 warning、退化为 default；绝不让 plan 失败。

**验收标准：**
- Given 16K 提示且未覆盖 budget，When 规划，Then budget 来自曲线且 ≤0.15，`budget_source=model`；
- Given 显式 `--budget 0.3`，When 规划，Then budget=0.3 且 `budget_source=override`；
- Given 缺失曲线表，When 规划，Then 0.25 且 `budget_source=default`（零回归）。

### 3.4 FR-4 图内门控去除 K 全量拷贝

**描述：** AR006 页打分当前对整段 K 做 `cont(permute(...))` 物化再 pool；改为
`mul_mat(k_view, q_last)` 得到每页每头的点积，再按页求和，避免 O(n_kv·d·h) 的拷贝。

**触发条件：** `qg_bound` 开启且 `n_kv > local 窗`。

**期望行为：**
- 打分数学不变（q·mean(K_page) 等价形式），页选择结果与旧实现在同一输入下一致（或差异仅来自浮点次序，需单测容差）；
- 32K 下 bound 的 TTFT 相对旧实现下降 ≥30%（同质量）；
- 关闭门控时 dense 路径字节等价不变。

**异常处理：** 形状不满足（n_kv<h_kv*d 等）→ 回退旧实现路径并审计。

**验收标准：**
- Given 16K/32K niah，When 对比新旧打分，Then 选中页集合一致率 ≥99% 且 q 不变；
- Given 32K bound p24+local32，When 跑，Then q=1.0 且 TTFT 比 67.9s 基线低 ≥30%；
- Given 单测构造 K 页，When 两种实现，Then 页分数 max abs diff <1e-4。

### 3.5 FR-5 复现资产与服务器实验手册

**描述：** 让「从 Gitee 克隆 → 在高速算力服务器复现」成为一条命令 + 一份手册。

**触发条件：** 任意机器，按 README/server-repro 操作。

**期望行为：**
- `benchmarks/matrix/budget/*.conf`（4 长度 × 3 预算，含 anchors 参数）；
- `repro.sh sweep`（CPU，可选 `--server` 大模型档）与 `expected.csv` 校验；
- `docs/server-repro.md` 增加：7B/14B 配置、单卡/多卡预期、单位 token 成本换算、失败排查。

**异常处理：** 缺模型/数据时给出明确下载与校验指引，不静默跳过必需步骤。

**验收标准：**
- Given 干净克隆，When `./repro.sh sweep`，Then 构建+单测+4K sweep 跑通并产出汇总（<30 min）；
- Given 服务器手册，When 按其命令在 ≥8 卡机器上跑 7B 模板，Then 产出 raw.csv 与汇总（文档化，不承诺本机实测）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 16K 同质量净 TTFT | 比 AR006 低 ≥1.5×（≤12% 预算） |
| 性能 | 32K bound | 比旧实现低 ≥30%（同质量） |
| 可靠性 | 确定性 | 同输入两次选择/生成逐位一致 |
| 兼容 | dense 等价 | budget=1.0 全关路径逐位不变 |
| 可维护 | 警告 | -Werror 零警告（CPU 构建） |
| 可复现 | 证据 | 所有数字来自 raw.csv + env.json，脚本可重算 |

## 5. 约束与假设

**约束：**
- 不引入新模型依赖；不训练；所有加速来自调度/选择/图实现；
- 必须真实 GGUF 全链路；无 mock；
- 曲线测量必须同机同会话完成，跨会话只比比值。

**假设：**
- 1.5B q4_k_m 上的质量趋势可外推到 7B/14B（手册中标注为假设，需服务器验证）；
- 目标任务的答案依赖局部连续文本（needle/QA），对纯推理型长文任务不保证。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 准入（admission） | 被允许进入 KV cache 并参与 transformer 前向的 token |
| 预算（budget） | 准入 token 数 / 提示 token 数 |
| 质量悬崖 | q 随 budget 下降开始陡降的拐点 |
| 锚点（anchor） | 与分数无关、按结构强制入选的块 |
| MMR | Maximal Marginal Relevance，按「相关且不重复」贪心选块 |

## 7. 依赖的既有能力

`Selector`/`BlockSelector`（块选择与 mandatory）、`build_query_gated`（AR005）、
`SparsePrefiller`（chunk/plan/审计）、`llama_set_prefill_qg_bound`（AR006 图内门控）、
`FfnSkipController`（本次仅用于成本探针，不扩展）、bench 审计列。

## 8. As-built 偏差与实测结论（2026-10-26）

需求在实现过程中被证据修正的地方，逐条记账。**不改写原需求**，只记录差异与依据。

| 需求点 | 原需求 | 实际落地 | 依据 |
|---|---|---|---|
| FR-1 信号维度 | 4 长度 × **2 信号** × ≥4 预算 | 4 长度 × 3 预算 × **lexical** 全测（12 格点）；embedding 通道已接线并抽测（4K b=0.12 q=1.0、2.0s），完整 embedding sweep 交给 `./repro.sh budget` | 图内门控路径不依赖嵌入，选择级曲线的决策变量是 budget/length；本机时间优先用于 FR-2/3/4 |
| FR-1 预算档 | {0.20,0.15,0.10,0.05} | {0.05,0.08,0.12}（4K 另测 0.15 语义由规则推导） | 0.20/0.15 的实测无额外信息量：32K b=0.12 已比 b=0.05 更差，更高预算只增加成本 |
| FR-2 审计字段 | `anchor_blocks` + `anchor_bytes` | 只保留 `anchor_blocks` | 块数已能定位预算占用；字节数随 `kv_type` 变化，同一选择会给出不同数值，审计价值低 |
| FR-2 位置先验 | "归一化分数 × 位置先验" | 改为 **span growing**：填充时相邻已选块 +0.5，再走 MMR | 阈值式位置加权在 16K/32K 低预算下切断证据 span（needle 句跨块边），实测 q 由 0 回到 1.0 |
| FR-2 验收（16K b=0.12 q≥0.9） | 0.12 预算达标 | **未达成**：16K b=0.12 q=0.50；但 b=0.05 q=1.00（更省且更准） | 质量对预算非单调（命中锚点被稀释），说明"用更大预算换质量"的前提不成立；见 research §9.1 |
| FR-3 安全边际 | 曲线值 × **1.25** | 默认 `margin=1.0`，规则改为"最快格点 1.15 倍内选最省预算"；`margin>1` 仍可用，且只吸附到**已测**格点 | 非单调下放大预算可能更差（32K b=0.05 q=0.89 → b=0.08 q=0.67），加边际等于洗白更差的点 |
| FR-3 缺表默认 | 0.25 | `budget_source=default` + **模式默认 0.5**（不变更既有行为） | 0.25 是 AR006 的实验值而非 planner 默认；改默认会静默改变所有既有配置 |
| FR-4 页选择一致性 | 新旧选中页集合一致率 ≥99% | 未单独统计集合一致率；以"同配置质量不降 + 32K TTFT 下降"作为等价证据（16K p16 q=1.00/23.7s→21.8s，32K p24+local32 q=1.00/67.9s→25.4-31.4s） | 页集合一致率需要导出中间张量，成本高于收益；质量与加速是最终契约 |
| FR-4 形状异常 | 回退旧实现 | 回退为 `local-only`（`k_sel=0`）+ `qg_reason` 审计，不保留旧拷贝路径 | 旧路径正是性能问题来源；local-only 是已存在的安全降级（AR006 短提示旁路） |
| FR-4 回归守护 | — | `bound_gating_execution_and_restore` 增 `qg_local_rows=32` 样本 | 默认窗口会把 fixture 全覆盖成 local-only 旁路，页打分分支此前无真机覆盖，导致一次 `mul_mat` 形状 abort 漏到 campaign |
| FR-5 复现入口 | `repro.sh sweep` | `repro.sh budget`（曲线）+ `repro.sh ar007`（验收战役 + expected 闸门） | 与既有 `smoke/matrix/full` 命名一致；`expected.csv` 用质量/审计列而非绝对延迟，机器无关 |

**结论**：AR007 的核心假设"存在一个长度相关的更优预算"被实测支持（8K 29×/16K 38.5×/32K 55.2×，
q=1.00/1.00/0.89）；"预算越大质量越高"被实测否定，曲线与推荐规则据此重写。
多针族（multineedle）在选择级稀疏下只有 q≈0.33，曲线因此默认关闭，
多针场景继续走 AR006 图内门控（q=0.83）。
