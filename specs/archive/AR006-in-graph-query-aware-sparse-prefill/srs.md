# [AR006] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| AR 主题 | in-graph-query-aware-sparse-prefill（图内精确界查询感知稀疏 prefill） |
| 关联 SR | 延续 AR004/AR005 的 prefill 稀疏加速主线 |
| 日期 | 2026-10-25 |
| 状态 | Draft（自动化 harnessX，用户已豁免逐步确认） |

## 1. 背景与目标

### 1.1 已实测的痛点（全部来自真实运行，无 mock）

1. **qg 行集预算与 gather 成本矛盾**：AR005 修复后，每个查询组必须包含自身查询跨度行（局部上下文）
   才有稳定质量；4K needle 在 `query_top_k=384` 时 niah-01 仍缺针，`512` 才 4/4。
   每 ubatch 内 4 个组各 gather 一次（每行被重复搬运），实测 qg ttft ≈ scalar qgoff 的 2 倍。
2. **语义打分开销在关键路径**：bge-small 嵌入编码 4K 约 1.1s（块缓存后）、32K 更高；
   `semantic_score_ms` 直接计入净 TTFT，且需要额外模型（部署体积/加载时间）。
3. **缺 per-position 分数时退化为前缀行集**：无嵌入模型的 lexical-1d 路径下所有块分数相等，
   排序退化为块序，行集退化成因果前缀——质量靠运气（实测同配置 q 在 0.44~1.0 间波动）。
4. **行集是 plan 级静态的**：同一提示的所有层共用一份分组行集；而真实注意力分布在层间、
   查询间差异很大（MInference/Quest 的核心观察），静态行集在部分层是浪费、在部分层是缺口。

### 1.2 目标

在**不引入训练、不增加外部模型**的前提下，把查询感知的块级选择搬到**注意力图内部、逐层、逐查询分片**
执行，用**精确上界**（Quest 式 K 块 min/max 元数据）替代嵌入打分；同时用
**单次并集 gather** 消除重复搬运、用**三分支预算**（local window + selected pages + compressed）
把质量底座固定下来。

**北极星指标（净 TTFT，含一切开销）**：
- 32K：q≥0.9，净加速 ≥10×（不低于 AR004/AR005 基线），且 qg 路径净加速 ≥ 同配置 scalar 的 1.0×（当前 0.5×）；
- 16K：q≥0.9，净加速 ≥8×（基线 4.87×/AR005 目标 6.5×）；
- 4K/8K：q≥0.9，净加速不劣于 AR005 验收配置；
- mn8K（12 用例）：平均 q≥0.66 且不低于 AR005；
- `semantic_score_ms == 0`（attention 门控不再依赖嵌入；嵌入仅在选择级可选）。
- budget=1.0 / 全开关关闭时与 dense 字节等价（max abs logit diff < 1e-5）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- patch 侧新增图内精确界门控：K 块 min/max 元数据 + 分片上界打分 + top-k 页选择 + gather/掩码；
- 单次并集 gather（每 ubatch 一次），替代当前 per-group 重复 gather；
- 三分支预算一等公民化：`local window`（局部必留，等价当前 own-span 的推广）+ `selected pages` + `compressed/pool`；
- 规划侧适配：预算配置、审计列（`qg_source=attention-bound`、`sel_pages_mean` 等）、降级链；
- 可选 head 路由（retrieval/streaming，由真实 K 统计导出，非合成探针）；
- 文档与复现：README 一键复现、7B/14B 配置模板、服务器（高算力）复现指南、expected 表；
- 真实实验矩阵（1.5B CPU 全量 + GPU 关键格点守护 + 7B 配置模板验证性小跑）。

**Out of Scope（本 AR 不做的）：**
- 训练/微调 router 或任何模型参数（全部免训练）；
- 分布式/多卡推理、服务化调度（服务端多请求另立 AR）；
- decode 阶段稀疏（保持 dense decode）；
- 更换 llama.cpp pinned commit（继续 3bb63cbcbfb62056c84fe9e521b0f3d756d29c32）。

## 3. 功能需求

### 3.1 图内 K 页元数据（MoBA 门控信号）

**描述：** 在 attention 图内按页（page = 32 行 KV）计算每头 K 的页均值向量，作为查询-页打分信号，
不引入额外模型或 host 同步。Quest 式 min/max 精确界已实现并实测否决（见 FR-2）。

**触发条件：** prefill 图构建且稀疏开关开启（`qg_bound` 且 `qg_sel_pages>0`）。

**期望行为：**
- 元数据从本层实际写入/可见的 K 缓存视图导出（`ggml_pool_1d(AVG)`，页大小 32，对齐 AR004/AR005 块）；
- 元数据只用于打分，不改变 K/V 数值与位置；尾部不足一页的行固定留在 local 窗口；
- 关闭开关时图与 dense 路径逐节点一致（字节等价）。

**异常处理：** KV 长度不足一页 → 不建选择节点（local-only/dense 等价）；
页尺寸不匹配（非 32）→ 规划层 `InvalidArgument`。

**验收标准：**
- Given 开关开启、16K 提示，When 构建 prefill 图，Then 页数 = floor(n_kv/32) 且门控参与选择（审计可见）；
- Given 开关关闭，When 运行，Then logits 与 AR005 dense 路径 max abs diff < 1e-5（既有回归）。

### 3.2 FR-2 分片页打分与 top-k 页选择（含精确界评估结论）

**描述：** 对每个查询分片（默认 = 预填充 ubatch），对全部非 local 页打分并选出预算内 top-k 页。

**打分函数（经实测决策）：** 默认 `score(q,p) = q · mean(K_page)`（MoBA 式门控），
其中 q 取分片末 token（Quest 式当前查询）在每组第一个 query head 上的投影（GQA 对齐）。
**精确上界方案已实现并实测否决**：`u(q,p)=Σ_d max(q_d·kmax_pd, q_d·kmin_pd)`（Quest 式）
作为排序信号时，16K 真实提示 retrieval 失败（quality 0）——上界对 key 范数大的页系统性高估，
与真实注意力显著性不一致。该结论作为 research v3 的证据保留（实现删除，仅留文档与设计记录）。

**触发条件：** FR-1 页均值元数据可用（`qg_bound` 开、页数 ≥ 1）。

**期望行为：**
- 页索引 → 行索引通过预计算查表张量 `row_table[32, n_pages]`（元素 `(r,p)=p·32+r`）
  经 `get_rows` 展开（ggml 无 F32→I32 cast 路径，整数指数运算不可用）；
- 选择结果为页索引集合；页内行全部进入该分片注意力（块粒度），配合因果掩码；
- top-k 与预算 `qg_sel_pages` 一致；`qg_sel_pages=0` 时关闭 selected 分支；
- 各头共享同一页集（ggml KQ mask 只广播到 head 维，逐头掩码不可行）。
- 预算上限：`n_local + 32·k_sel ≤ 4096`（超出自动裁剪并审计）。

**异常处理：** 上界/打分出现非有限值 → 该分片回退 dense（reason 审计），不得静默产出错误行集。

**验收标准：**
- Given 16K 真实提示、`qg_sel_pages=16`，When 运行，Then needle 质量 = 1.0（实测已达成）；
- Given `qg_sel_pages=8` 同提示，Then 记录质量下降并作为预算下限证据（实测 q=0）；
- Given 构造的 K 页与查询（单测），When 计算页均值门控，Then 与 host 参考实现逐元素一致；
- Given 任一配置，When 两次运行，Then 生成逐位一致（确定性）。

### 3.3 FR-3 单次并集 gather + 可见性掩码

**描述：** 每个 ubatch 只 gather 一次（local window ∪ selected pages ∪ compressed 行的并集），
用掩码表达各查询分片的可见性，消除当前 per-group 重复 gather。

**触发条件：** 稀疏路径开启。

**期望行为：**
- gather 行数为并集大小；同一行不重复搬运；
- 掩码同时表达因果性、local window 边界与 selected 页集合；
- 关闭时回退 dense；
- 与 FR-1/FR-2 组合时总 gather 次数：每 ubatch 每层 1 次。

**异常处理：** 并集超过实现上限（如 4096 行）→ 按预算裁剪 selected 且审计；仍超限 → dense 回退。

**验收标准：**
- Given qg 32K 配置，When 运行，Then qg ttft ≤ scalar qgoff ttft × 1.15（同预算、同数据集、3 次重复均值）；
- Given 单测构造的并集/掩码，When 与逐组独立 gather 的参考实现对比，Then attention 输出 max abs diff < 1e-5（等价性）。

### 3.4 FR-4 三分支预算一等公民

**描述：** 显式配置 local window 大小、selected 页预算、compressed 行预算三部分，取代隐式组合。

**触发条件：** 稀疏路径开启。

**期望行为：**
- 配置键：`qg_local_rows`（默认 = query_chunk_size）、`qg_sel_blocks`、`qg_pool_blocks`；
- local 分支永远包含（含 sink 行 0）；selected 与 pool 按预算；
- 三者为 0 时该分支关闭；全部关闭且 local=0 → dense 等价；
- 审计列输出三分支实际行数。

**异常处理：** 非法组合（如 local=0 且 selected=0 且 pool=0 但稀疏开关开）→ `InvalidArgument`。

**验收标准：**
- Given 三组不同预算配置，When 运行 4K，Then 审计列三分支行数与配置一致；
- Given local-only（selected=0,pool=0），When 与 AR005 own-span qg 对照，Then 质量在 ±0.02 内。

### 3.5 FR-5 免嵌入门控与嵌入降级为可选

**描述：** attention 门控默认不再需要嵌入模型；`embedding_model` 仅在显式开启选择级语义筛选时使用，
且计时只进 `semantic_score_ms`。

**触发条件：** 默认 `qg_source=attention-bound`；`--selector-embedding on` 时选择级用嵌入。

**期望行为：**
- 默认路径 `semantic_score_ms == 0`；
- 显式配置嵌入而加载失败 → 硬错（`ModelLoad`），不静默降级（沿用 AR005 契约）；
- 保留 AR005 的 2D 路径作为可选项（`qg_score_source=embedding-2d|attention-bound|lexical-1d`）。

**异常处理：** 未知取值 → `InvalidArgument`。

**验收标准：**
- Given 默认配置，When 跑 32K，Then `semantic_score_ms==0` 且 `qg_source=attention-bound`；
- Given `qg_score_source=embedding-2d`，When 跑 4K，Then 与 AR005 修复后结果一致（回归）。

### 3.6 FR-6 head 路由（retrieval/streaming，真实统计）

**描述：** 由真实 K 元数据统计导出每头“是否检索头”，streaming 头只跑 local 分支，retrieval 头跑三分支。

**触发条件：** `--head-route auto|off`（默认 off，保持可禁用）。

**期望行为：**
- 统计量：上界分数的熵/覆盖率或页内注意力质量的代理（由元数据导出，无需额外前向）；
- 输出确定：同模型同校准文本 → 同分类；结果可写入 profile（复用 AR003 格式，扩展字段 mean-bound-spread）；
- off 时逐字段与 FR-4 全头一致。

**异常处理：** 校准文本过短/页数不足 → 回退全头路径并审计。

**验收标准：**
- Given 同一校准文本两次运行，When 分类，Then 输出逐字节一致；
- Given auto 与 off，When 16K 实测，Then auto 质量 ≥ off − 0.02 且 ttft ≤ off。

### 3.7 FR-7 复现与服务器规模化（产品化）

**描述：** 从 Gitee 干净克隆可一键复现：构建、下载校验、smoke/matrix/full 三套件、期望表对照、
7B/14B 配置模板与服务器指南。

**触发条件：** `repro.sh` / README。

**期望行为：**
- `repro.sh smoke` < 30min（含模型下载校验）；`matrix` 产出 matrix.csv/md；`full` 产出全部验收表；
- README 首屏给出一行命令与预期输出；模型/数据 SHA256、线程数、llama commit 全记录；
- `benchmarks/matrix/*.conf` 为机器无关注册表；7B/14B 模板在 `benchmarks/matrix/models/`（或 conf 注释）中给出显存/线程建议；
- 服务器指南：`docs/server-repro.md`（大内存/多核参数、GPU offload、批大小、预期时长）。

**异常处理：** 任一校验失败（SHA256/依赖/模型缺失）→ 非零退出并打印修复建议。

**验收标准：**
- Given 干净克隆（仅 Gitee + 网络），When 执行 README 两行命令，Then 测试通过且 smoke 在 30min 内完成；
- Given 7B 模板，When 在 ≥32GB 服务器执行，Then 无需改代码即可运行（记录预期数值区间）。

### 3.8 FR-8 质量守护与如实报告

**描述：** 保留“稀疏即损失”的诚实原则：每个配置报告质量与净加速；`budget=1.0`/全关与 dense 等价。

**触发条件：** 所有测量。

**期望行为：**
- raw.csv 追加审计列：`qg_source`、`sel_pages_mean`、`local_rows_mean`、`pool_rows_mean`、`head_route`；
- 期望表 `benchmarks/matrix/expected.csv` 含容差区间与机器指纹（CPU/线程/commit）；
- 不可达指标必须记录根因三要素（现象/证据/替代），不得用合成数据填补。

**异常处理：** 质量低于门槛 → 矩阵中标注 FAIL，但不删除数据。

**验收标准：**
- Given matrix，When 与 expected.csv 对照，Then check_expected.py 以非零码报出越界项；
- Given 任一结果行，When 追溯，Then 可由 conf + env.json + 命令重现。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | qg 净 TTFT | ≤ scalar 同预算 ×1.15（32K） |
| 性能 | 图构建开销 | 元数据+选择节点 < 注意力节点的 15%（节点数/实测时间） |
| 质量 | 32K needle | q ≥ 0.9（4 用例均值与最小值都记录） |
| 质量 | 16K needle | q ≥ 0.9 |
| 可靠性 | dense 等价 | budget=1.0/全关 max abs logit diff < 1e-5 |
| 可移植 | 后端 | CPU 必过；CUDA 关键格点守护（top_k/get_rows 后端可用性显式校验） |
| 可维护 | 最小 patch | 新逻辑全部可在 C API 关闭，关闭即原路径 |

## 5. 约束与假设

**约束：**
- 免训练：不得引入可训练参数/新模型（嵌入仅选择级可选）；
- 固定 llama.cpp commit，patch 必须可审查、可禁用；
- 端侧优先：单请求 batch=1，CPU 可跑，额外内存 ≤10%；
- 无 mock：所有质量/延迟数字来自真实 GGUF 推理；
- 自动化 harnessX：每步记录工作与收获反馈，交付文档同步 specs/docs/README。

**假设：**
- 页粒度 32 行与现有块粒度一致（AR004/AR005 已验证）；
- `ggml_pool_1d(MAX)`、`ggml_top_k`/`ggml_argsort`、动态索引 `get_rows` 在 CPU 后端可用（设计阶段需实测确认，不可用则退化为阈值法 + host 端预选）；
- 目标模型为 Qwen2.5 系列（1.5B 实测、7B 模板），GQA 非均匀时回退 dense（沿用现有约束）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| page | 32 行 KV 的块（与块选择粒度一致） |
| 精确界 | `u(q,p)=Σ_d max(q_d·kmax_pd, q_d·kmin_pd)`，页内任意 key 的 logit 上界 |
| local window | 查询分片自身及前 W 行的必留窗口（含 sink 行 0） |
| selected pages | 上界打分 top-k 页 |
| compressed | 池化压缩行分支（AR004 pool 的延续） |
| 并集 gather | 每 ubatch 一次 gather 三分支并集，用掩码表达分片可见性 |
| streaming/retrieval head | 只需局部窗口即可 / 需要全局检索的头 |
