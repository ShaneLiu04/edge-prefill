# [AR007] AR 详细设计

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.5） |
| 日期 | 2026-10-25 |
| 状态 | Draft |

## 1. 方案总览与备选对比

### 1.1 问题定位（实测驱动）

| 事实 | 证据 | 推论 |
|---|---|---|
| 成本 ∝ 准入长度² | 16K dense 161.8s（15.5K 准入）vs 25% 准入 22.5s；8K 118.9s vs 11.2s | 预算（准入比例）是唯一还有数量级空间的设计维度 |
| FFN 不是瓶颈 | 16K 稀疏：`--skip-ffn 50%` 无收益、`100%` 仅快 15-20%（3 次实测） | 不做 token 级 FFN 跳过 |
| 默认预算 = 0.5（未覆盖时） | `src/planner.cpp` balanced 分支 | 用户默认只拿到 ~4-5×，而 0.15-0.25 已实测 q=1.0 |
| 32K 被硬 clamp 到 4096 token | `budget = min(budget, 4096/n_tokens)` | 长上下文的预算策略是「内存驱动的粗暴截断」，非质量驱动 |
| AR006 门控 32K 比标量还慢 | 67.9s vs 22.8s | 页均值打分物化了整段 K 的拷贝 |

### 1.2 备选方案

| 方案 | 核心 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A. Token 级 masked FFN | 深层对低重要 token 跳过 FFN | 直击 FLOPs | 实测 FFN 仅 15-20%，且需 gather/scatter 改造 | 否决（证据在 SRS §3.1） |
| B. 更激进的注意力行裁剪（AR006 加深） | 更小 page 预算 | 改动小 | 已实测：行裁剪省的时间远小于准入；32K 覆盖不足 | 否决为主方向 |
| C. 预算中心化：曲线 + 锚点/MMR + 自适应 | 用更少 token 达同质量 | 二次收益、无新依赖、可测量 | 需要真实曲线与质量门 | **采纳** |
| D. 引入第二模型做打分/压缩 | 语义压缩器 | 质量上限高 | 违背「无额外模型/关键路径」原则（AR006 已把嵌入移出） | 否决 |
| E. 页打分去拷贝 | `mul_mat` + 页求和 | 纯收益、风险低 | 只影响 32K 门控路径 | **采纳（FR-4）** |

SOTA 映射（详见 research 增补）：Quest 的页粒度与精确性思想 → 我们保留页粒度但去掉物化拷贝；
MoBA 的「当前块必留」→ 我们的 sink/recent 锚点；LongLLMLingua 的「问题感知粗到细压缩」→
我们的 quote/entity 锚点与 MMR；SnapKV 的 observation window → 已有 recent 锚点；
PyramidKV/MInference 的层/头预算 → 不在本 AR（AR006 T007 仍 pending）。

### 1.3 总体流程

```
prompt + question
   │
   ├─ [Planner] 长度 → 曲线表查预算（含边际）→ budget / budget_source=model
   │             （显式 budget 优先 → override；无表 → default 0.25）
   ▼
[SAMSelector] 词法/嵌入分数 → 归一化 → 融合
   │  ├─ 硬锚点: sink / recent / 预算感知 stride / quote+entity 块
   │  ├─ MMR 贪心: score × (1 - λ·max_jaccard(已选))
   │  └─ 预算裁剪: 非锚点按 (score) 升序移除
   ▼
Selection（token_indices/block_indices/mandatory_blocks + 审计）
   ▼
SparsePrefiller → patch（AR005/AR006 图内路径）
   ▼
bench 审计: budget_source / anchor_blocks / mmr_penalized / q / ttft
```

## 2. 功能点（FP）分解

| FP | 内容 | 影响文件 | 依赖 |
|---|---|---|---|
| FP1 | 曲线测量与落库：`sweep_budget.py` + `results/ar007-sweep/{raw,summary.json,summary.md}` | `benchmarks/scripts/`, `benchmarks/results/` | - |
| FP2 | 结构锚点：sink/recent（既有）+ 预算感知 stride + quote/entity 块；锚点预算上限 `anchor_ratio` | `src/selector.cpp`, `include/edge_prefill/selector.h`, `src/config.cpp` | - |
| FP3 | MMR 去冗余：块 token 集合 Jaccard，阈值 `mmr_threshold`、强度 `mmr_lambda` | `src/selector.cpp`, config | FP2 |
| FP4 | 长度自适应预算：曲线表读取（JSON）+ 边际 + `budget_source` 审计 | `src/planner.cpp`, `src/benchmark.cpp`, `src/main.cpp`, config | FP1 |
| FP5 | 页打分去拷贝（patch） | `llama-graph.cpp`（patch 生成管线）、`patches/*.patch` | - |
| FP6 | 复现资产：`benchmarks/matrix/budget/*.conf`、expected、`repro.sh sweep`、server-repro 更新 | `benchmarks/`, `docs/`, `repro.sh` | FP1 |

## 3. 接口与数据结构

### 3.1 配置（`Config`）

| 键 | 类型 | 默认 | 语义 |
|---|---|---|---|
| `anchor_ratio` | double | 0.35 | 锚点最多占预算比例（超出按 sink>recent>quote>stride 裁剪） |
| `anchor_stride_scale` | double | 1.0 | stride 密度系数（1.0 = 每预算 1/4 放一个覆盖采样点，至少 4 个） |
| `mmr_threshold` | double | 0.60 | Jaccard 超过该值开始惩罚 |
| `mmr_lambda` | double | 0.50 | 惩罚强度 `score*(1-λ*sim)` |
| `budget_curve` | string | `benchmarks/matrix/budget/curve.json` | 曲线表路径（空=禁用自适应） |

### 3.2 审计字段（`Sample` / raw.csv 新增列）

| 列 | 类型 | 语义 |
|---|---|---|
| `budget_source` | string | `override` / `model` / `default` / `clamp` |
| `anchor_blocks` | uint32 | 锚点块数 |
| `mmr_penalized` | uint32 | 被 MMR 降权过的候选块数 |

> 兼容性：新增列追加在 raw.csv 末尾，旧脚本按列名读取不受影响（AR005/AR006 已确立该约定）。

### 3.3 曲线表格式（as-built：纯文本 key = value）

设计初稿用 JSON；实现改为**行式 `key = value`**（与 `config.cpp` 同解析风格，零依赖、
错误行号可定位），脚本同时产出机器可读的 `curve.json` 供分析：

```
# benchmarks/results/ar007-sweep/budget_curve.txt（planner 输入）
model_sha256 = 6a1a2eb6...
quality_floor = 0.95
4000 = 0.15
8000 = 0.12
16000 = 0.12
32000 = 0.13
```

语义（与 sweep 脚本、planner 实现一致，as-built）：
- 数字键 = 该数据集**实测到的最大提示 token 数**（`max(prompt_tokens)`，不是名义长度），
  值 = 该长度下"达标且足够快"的最省 budget：先取 quality 与 min_quality 均 ≥ floor 的
  格点，再保留 TTFT 在最快格点 `speed_tolerance`(1.15) 倍以内的那些，取其中 budget 最小者。
  这样把"计时噪声导致的假优势"（0.08 vs 0.12 差 1.7%）用预算而不是计时器决胜；
- 没有格点达标 → 记 `floor-missed`，取质量最高的实测格点，且**不加边际**：
  曲线非单调时放大预算可能更差（32K：b0.05 q=0.89、b0.08 q=0.67），
  加边际等于把一个更差的点洗白；
- `recommend(n)` 取**不小于 n 的最小长度点**（ceiling；短于所有点用最小点，长于所有点用
  最大点再走 planner 的 8192 token ceiling）——曲线是按数据集长度实测的稀疏点，不是密集网格，
  提示落在两点之间时借用"更长的那次测量"，因为长提示的上下文包含了短提示的全部内容；
- `model_sha256` 与当前模型不符 → 调用方（runner）清空 `budget_curve` 并记 warning，
  绝不带着别的模型的数字跑；
- 解析严格：非法行 → `InvalidArgument`（带行号）；无点 → `InvalidArgument`；
  文件缺失 → 调用方降级为 default（`budget-curve-unavailable` reason）；
- bench 网格的 `budget=0` 是"让曲线决定"哨兵：`main.cpp` 只在
  `!budget_overridden && budget_curve 非空` 时设置，`benchmark.cpp` 见到 0 就不强制
  `budget_overridden`，每样本的 `budget_source` 如实记录 `model|default|clamp`。

### 3.4 选择器实现细节（as-built）

- 锚点上限 `anchor_ratio`（默认 0.35）× 目标 token 数；裁剪顺序 stride → entity，
  永不删 sink 与末块（末块承载首 token logits，sink 承载注意力汇）；
- stride 采样数 `n_samples = max(4, round(0.25*target_blocks*stride_scale))`，
  显式 `coverage_stride>1` 时取两者较小值；`coverage_stride=0` 表示自动；
- entity 锚点：问句词中 df ≤ max(2, n_blocks/50) 且（长度≥6 或含数字或 CJK）
  的词，各取该词得分最高的上下文块（多事实问句因此每个事实至少有一个锚点）；
- MMR 增量实现：`max_sim[b]` 仅在新增选中块时刷新，填充复杂度 O(n_blocks²) 次
  Jaccard（32K ≈ 1e6 次集合操作，实测 <0.5s），避免 O(n³) 超时；
- 审计：`Selection.anchor_blocks` / `Selection.mmr_penalized` → raw.csv 两列。

## 4. 关键流程（时序）

1. `main/bench` 解析配置 → `Planner::make_plan`；若 `budget_curve` 非空且模型指纹匹配且
   `!budget_overridden` → `budget = curve.recommendation[nearest_leq(length)]`，`budget_source=model`；
   随后仍执行内存 clamp（记录 `budget_source=clamp`）。
2. `BlockSelector::select` 计算分数 → 归一化（min-max 到 [0,1]，非有限值置 0）→ 融合；
   生成锚点集合（含 stride 数量 = `max(4, round(budget_blocks*0.25*anchor_stride_scale))`）；
   贪心填充时对每个候选计算 `sim = max_jaccard(候选, 已选)`，填充分数
   `adj = fused + 0.5·[候选与已选块相邻]`，再按 `adj·(1 - λ·max(0, sim-thr)/(1-thr))` 降权；
   相邻块偏好（span growing）负责把证据 span 连成片，λ 只惩罚近重复文本。
3. `SparsePrefiller` 不变（消费 Selection）。
4. patch 页打分（as-built，去 K 拷贝）：`q_heads = cont(view_3d(q, d, 1, h_kv, ...))`
   （只物化 `h_kv·d` 个数）→ `row_dot = mul_mat(k_p, q_heads)`，其中
   `k_p = permute(k,0,2,1,3)` 把缓存张量 `[d, h_kv, n_kv]` 变成 `[d, n_kv, h_kv]` 的 view
   → `reshape_3d(row_dot, bs, n_pages, h_kv)` → `sum_rows` → `[1, n_pages, h_kv]`
   → `permute+cont` → `[h_kv, n_pages]` → `pool_1d(MAX, h_kv)` → `[n_pages]`
   → `+bias` → `top_k`（其余流程不变）。页内求和等价于"页均值 × q"（差一个 1/bs 常数，
   排序不变）。
   > 教训：`ggml_mul_mat(a,b)` 只允许 `b->ne[2] % a->ne[2] == 0`（a 是广播方），
   > 直接用裸 `k`（`[d,h_kv,n_kv]`）会让 `n_kv % h_kv` 判定失败并在 32K 直接 abort。
   > 现在 `bound_gating_execution_and_restore` 额外跑一个 `qg_local_rows=32` 的样本，
   > 强制进入页打分分支（默认窗口会把整段 fixture 覆盖成 local-only 旁路），防止回归。

## 5. 错误处理与降级链

| 场景 | 行为 |
|---|---|
| 曲线表缺失/解析失败/指纹不符 | warning + `budget_source=default`，预算沿用 0.5（balanced）或显式值；不失败 |
| `anchor_ratio` 越界 | `InvalidArgument`（配置层） |
| MMR 参数越界 | `InvalidArgument` |
| 锚点超预算 | 按优先级裁剪并计数，不失败 |
| 页打分形状不满足（`n_kv % bs != 0` 或 `q->ne[1] % h_kv != 0`） | 回退到 `local-only`（`k_sel=0`）并审计 `qg_reason`；不 abort |
| 非有限分数 | 视为 0，MmrPenalty 不计 |

## 6. 测试设计

| ID | 类型 | 内容 |
|---|---|---|
| UT-1 | 无模型 | stride 锚点数量随预算单调不减且有下限 4；锚点裁剪优先级正确 |
| UT-2 | 无模型 | MMR：构造 10 个重复块 + 1 个独特相关块，独特块必须入选中（λ=0.5） |
| UT-3 | 无模型 | 分数归一化对 NaN/Inf 的处理（置 0，不传播） |
| UT-4 | 无模型 | 曲线表解析：合法/缺字段/指纹不符/长度不命中（取 ≤length 最近点） |
| UT-5 | 无模型 | `budget_source` 三态与显式覆盖优先级 |
| UT-6 | 真机 | 16K budget=0.12 + 锚点/MMR 后 q≥0.9（4 case） |
| UT-7 | 真机 | 页分数新旧实现 max abs diff < 1e-4（构造小 KV） |
| UT-8 | 真机 | dense 等价回归（budget=1.0 全关）逐位一致 |
| IT-1 | 真机 | 32K bound p24+local32：q=1.0 且 ttft 降幅 ≥30% |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 低预算下质量悬崖位置随任务漂移 | 自适应预算选得过于乐观 | 曲线含 0.25 边际；`quality_floor=0.95`；文档标注任务适用范围 |
| MMR 误伤「多点相同事实」任务（mn） | 质量下降 | 阈值 0.6 较高（仅近重复文本触发）；mn 数据集实测守护 |
| 曲线表与模型绑定 | 换模型误用 | 指纹校验（model_sha256），不符即弃用 |
| patch 改动引入数值差异 | 质量/等价回归 | UT-7 容差测试 + IT-1 + dense 等价回归 |
| 主机噪声导致跨会话数字不可比 | 误判收益 | 只与同会话 dense/基线比值对比，raw.csv + env.json 入库 |

## 8. 交付物

- 代码：selector/planner/config/benchmark 改动 + patch（页打分去拷贝）。
- 数据：`benchmarks/results/ar007-sweep/`（曲线）、`ar007-campaign/`（对比验收）、`benchmarks/matrix/budget/*.conf`、`curve.json`、`expected.csv`。
- 文档：srs/design/tasks、README 结果表、research 增补、server-repro 服务器实验章节、组件详设变更记录。
- 复现：`repro.sh sweep`（<30min CPU）、Gitee 同步（split-push）。
