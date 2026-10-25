# EdgePrefill AR004 研究笔记：SOTA 思想映射与实测证据

每个创新点按"SOTA 核心思想 → 本项目映射与改造 → 实测证据（CSV/图引用）→ 局限"三段式
沉淀。全部数字来自 `benchmarks/results/` 入库的 raw.csv（含 env.json 环境快照），
可由 `benchmarks/scripts/assemble_matrix.py` 与 `ar004_figures.py` 重新汇总与绘图。
复现命令见各 results 目录 `configs/` 与 `docs/experiments.md`。

模型：Qwen2.5-1.5B-Instruct Q4_K_M（SHA256 `6a1a2eb6...`），
llama.cpp pinned `3bb63cb`，CPU=Xeon Gold 6234×16T，GPU=Quadro RTX 5000。
quality q = needle 命中率（niah 任务）或 PPL 保持率 `min(1, ppl_dense/ppl_sparse)`。

---

## 1. NSA（Native Sparse Attention）→ 压缩上下文分支（T004/T005/T006）

**SOTA 核心思想**：NSA 用三分支注意力（压缩粗粒度 / 选中细粒度 / 滑窗）替代全注意力，
让"被跳过"的上下文仍以压缩形态参与注意力，训练感知稀疏。

**本项目映射与改造**：免训练版压缩分支——被丢弃的 32-token 块在图内 mean-pool 成
1 行压缩 KV（`ggml_pool_1d`），与已选行 concat 进注意力集；因果门用块内最后 token
的 mask 行。与 NSA 不同点：无需重训练，压缩行由现有权重直接消费（softmax 天然加权）；
限制池化集为"满块前缀"（部分选择块的行号对齐会永久丢失）。

**实测证据**：
- `ar004-t006-t009/len8k-attn24-pool/`：attn-24 @8K q=0.44 → **q=1.0**（SRS FR-2
  验收线 ≥0.75 超额），TTFT 23.5s vs dense 52.0s；
- `ar004-t015-matrix/len4k-attn24-pool/`：4K q=0.44 → 1.0；
- `ar004-t015-matrix/len16k-attn24-pool-gpu/`：**16K 主验收 12.78× 且 q=1.0**；
- 图：`figures/ar004_fig3_cliff_rescue.png`（悬崖救回柱状图）。

**局限**：32K q=0.25 不救——token 级选择在 13% budget 下把 needle 整块丢弃，
超出"满块前缀"池化范围（`ar004-t015-matrix/len32k-attn24-pool/`，诚实记录）。

---

## 2. MoBA（Mixture of Block Attention）→ query-chunk 门控（T007/T008/T009）

**SOTA 核心思想**：MoBA 把 query 按块分片，每片独立做块级 top-k 门控
（专家=KV 块），不同 query 看不同上下文子集。

**本项目映射与改造**：图内分片循环 gather-concat——planner 按行空间分片
（每组必含 sink 块/组前邻 recent 块/因果可行块 top-k），patch 层 4 个 C API +
状态机（append-only 组配置），graph 每组独立 build_attn_mha 后沿 token 轴 concat；
per-chunk 与行窗求交适配 chunk 调度。

**实测证据**：
- 正确性/审计：`ar004-t006-t009/mn-qg/raw.csv` 的 `query_gating/qg_groups` 列
  证实 8 组激活、有限 logits（单测 `qg_execution_engages` 116/116 全绿）；
- 消融：多针 mn-v1/v2/v3 @8K，scalar attn-24 q=0/0/0.2 vs qg q=0/0/0
  （`ar004-t006-t009/mn-*`，图 `ar004_fig4_gating_ablation.png`）；
- **不可达分析（T009）**：1D per-position 词法分数使各组行集退化为同源因果全局
  top-k（组间仅 sink/recent/因果窗差异）；答案 token 不在问题中，干扰块词法命中
  稀释 needle 块均分。

**局限**：无语义 draft 分数时门控退化为全局 top-k——需 DraftScorer 位置×块分数
矩阵（v1 记录限制，接口已预留 `Selection.per_pos_scores`）。

---

## 3. MInference → head 模式路由与金字塔先验（T010/T011）

**SOTA 核心思想**：MInference 预分析每层每头注意力模式（streaming/sink/
sparse-pattern），推理时按头套不同稀疏模板，长上下文 prefill 提速。

**本项目映射与改造**：`classify-heads` 离线探针（echo/induction/retrieval 三分数）
产出 profile.json；`build_pattern_matrix` 按分类套模板
（retrieval=-1/landmark k=32/local k=48/normal k=head_top_k）；
无 profile 时 `build_pyramid_matrix` 凸函数 k(l) 先验兜底。

**实测证据**：
- profile：`exp-heads/profile.json`（336 头×3 分数），图
  `ar004_fig6_head_heatmap.png`；
- head-adaptive 执行：`ar004-t002-final/len16k-head/` 3.83×、GPU 11.56×
  （`ar004-t012-gpu/len16k-head/`）；
- 单测：`build_ranked_pool` 回归守护 + pyramid 冒烟（pool=2009, k=48/128）。

**局限**：head 路由在本 niah 数据上 q=0（与历史 `exp-head-16k` 一致）——profile
来自通用探针而非任务内校准，模式路由模板未在此任务分布上命中 needle
（诚实记录，改进方向=任务内 head 校准）。

---

## 4. Quest → 块级选择 + 标量 gather（T002 主线）

**SOTA 核心思想**：Quest 用每块的最小/最大 K 投影做 query 相关的块选择，
prefill/decode 均只算相关块，长上下文近乎常数开销。

**本项目映射与改造**：词法 idf + 多样性 + draft 分数的块选择器产出全局
top-k 行集，llama.cpp patch 的 `llama_set_prefill_attn_indices` 按行 gather
（decode 顺序行空间，位置≠行号需映射）；per-chunk 与行窗求交保因果。

**实测证据**：
- `ar004-t002-final/`：4K/8K/16K/32K attn-24 = 2.46×/2.61×/3.92×/12.41×（CPU）；
- GPU：8K 4.92×、16K 14.42×、32K 60.96×（`ar004-t012-gpu/`、
  `ar004-t015-matrix/len32k-attn24-gpu/`）；
- 图：`ar004_fig1_ttft_curves.png`、`ar004_fig2_pareto.png`、
  `ar004_fig7_cpu_gpu.png`。

**局限**：q 随长度两极（16K q=1，4K/8K q=0.44，32K q=0.25）——词法分数对
needle 位置的区分度不稳定；4K/8K 的悬崖由压缩分支救回（见 §1）。

---

## 5. SnapKV → head-adaptive top-k 矩阵（AR003 基座）

**SOTA 核心思想**：SnapKV 观察每头的"观测窗口"注意力分布，压缩 KV 到各头
各自的 top-k 位置，按头差异化保留。

**本项目映射与改造**：AR003 交付 per-head gather 矩阵 patch
（`llama_set_prefill_attn_head_top_k` + ranked rows），AR004 在其上叠
profile-free 池与金字塔先验；GQA 头共享行集由 set_input 跨头复制。

**实测证据**：
- `ar004-t002-final/len*-head/`：4K/8K/16K/32K = 2.23×/1.27×/3.83×/13.10×；
- GPU head：16K 11.56×（`ar004-t012-gpu/len16k-head/`）。

**局限**：与压缩分支 v1 互斥（head 矩阵优先时 pool 关闭）；8K head 1.27×
为矩阵构建开销占比过高（诚实记录）。

---

## 6. LLMLingua-2 → 分段最低保留（T005）

**SOTA 核心思想**：LLMLingua-2 用 token 分类器打分做提示压缩，按段保底
避免整段语义丢失。

**本项目映射与改造**：块选择器加"每原始文本段至少保留 1 块"的分段最低保留，
防整段全丢（Quest/LLMLingua 分段预算思想）；planner 输出池化集合时同样
保持段语义连续。

**实测证据**：
- 单测：`pool_planner_*` 与 `qg_planner_*` 系列锚点断言（116/116 全绿）；
- 8K/4K 池化分支 q=1.0 的段保底贡献由 T006 消融背书
  （`ar004-t006-t009/len8k-attn24-pool/`）。

**局限**：段保底不覆盖 32K 的 13% budget 场景（needle 段整体低于词法阈值时
仅保 1 块仍不够还原 passcode）。

---

## 7. vLLM/Sarathi chunked prefill → 大 chunk 默认调度（T001）

**SOTA 核心思想**：chunked prefill 把长提示切成 ubatch 尺寸的块逐次 decode，
平衡首 token 延迟与调度公平；块尺寸对齐计算核最优粒度。

**本项目映射与改造**：`chunk_size=0` 语义=自动对齐 n_ubatch（单次 llama_decode
完成一个 chunk），backend 不二次拆分（崩溃根因的断言防线）；任一 chunk 配置
（0/128/512/超大）下 gather 几何一致。

**实测证据**：
- `ar004-t015-matrix/chunk-{128,512,0}/`：4K attn24 TTFT 9.6/10.0/9.9s
  （±5% 等价），图 `ar004_fig5_chunk_harvest.png`；
- 诚实发现：SRS FR-1 的"2.5× 收割"实为对双重拆分病理的修复
  （修复前 52.5s → 修复后 ~10s），而非持续可调优项。

**局限**：本任务负载下块尺寸不再是敏感旋钮；更优收益来自 attention 级稀疏
（§1/§4）而非调度尺寸。

---

## 汇总：主验收与诚实结论

| 结论 | 证据 |
|---|---|
| **16K ≥10× 且 q≥0.9 达成**（GPU，attn24+压缩分支） | 12.78×，q=1.0，`ar004-t015-matrix/len16k-attn24-pool-gpu/` |
| 32K GPU 61× | `ar004-t015-matrix/len32k-attn24-gpu/`（dense 二次方恶化 787s） |
| 压缩分支救回质量悬崖 | 4K/8K q=0.44→1.0，fig3 |
| query 门控 v1 无增益 | mn 消融 0/0/0 vs 0/0/0.2，fig4，1D 分数局限分析 |
| CPU 16K 最优 4.87×（50% budget） | `matrix.md`，CPU 达不到 10× 诚实记录 |
| 环境非受控（宿主负载） | T002 首轮污染套件保留 `ar004-t002/`，隔离重跑 `ar004-t002-final/` |

---

# AR006 增补（v3）：图内查询感知门控 —— Quest × MoBA × NSA 在预填充内的合流

AR004/AR005 的门控都在 **host 侧**：先选 token/块，再用补丁 gather。AR006 把门控搬进
**prefill 图内部**（每 ubatch、每层），从而同时干掉三样成本：嵌入打分（1.1s/次）、
逐组多次 gather、以及 host↔device 同步。全部数字来自 enchmarks/results/ar006-campaign/
同一台机器同一会话（dense/scalar/qg/bound 交错测量，比值可比）。

## 8.1 SOTA 核心思想

| 来源 | 机制 | 我们的取用 |
|---|---|---|
| Quest (arXiv 2406.10774) | decode 期用页级 **精确界** u=Σ_d max(q·kmax, q·kmin) 选页 | 页粒度、当前查询打分；**上界打分经实测否决**（见 8.3） |
| MoBA (arXiv 2402.19473) | 块级 gate，块内 **K 均值** 作为专家表示，当前块必留 | 打分函数 q·mean(K_page)；local 窗必留 |
| NSA (arXiv 2502.11089) | 压缩 / 选中 / 滑窗 三分支 | 滑窗≈local+sink；选中≈top-k 页；压缩池化沿用 AR004 |
| DuoAttention / MInference | head 分流式/检索两类 | 作为后续 head 路由依据（ggml 掩码只广播 head 维，需拆图，暂缓） |

## 8.2 本项目映射与改造

- **位置**：llama-graph.cpp::build_attn（非 FlashAttn、非 SWA 层）新增分支；K 页均值
  ggml_pool_1d(AVG, 32) 由本层真实 K 视图导出，**无任何额外模型**；
- **分片**：以 ubatch 为上界（MoBA 的 chunk 门控语义），q 取 ubatch 末 token 在每组
  第一个 query head 上的投影（GQA 对齐，group 内并集 → 全头掩码可表达）；
- **选择**：	op_k 在图内完成（ggml 原生 kernel）；**页→行** 用预计算
  
ow_table[32,n_pages] + get_rows 查表展开——ggml 没有 F32→I32 cast 路径，
  range*32+add 这类整数指数运算不可用（T001 探针结论）；
- **单次 gather**：local（sink 页 + 尾部窗口）∪ selected 一次 get_rows；
  可见性掩码 = 每图构建时预转置一次的因果 mask 行 gather（逐层转置会多搬 2×n_tok×n_kv 字节）；
- **互斥与恢复**：C API llama_set_prefill_qg_bound arming 时清理其余稀疏路径，
  关闭后 dense 路径逐位不变（单测 ound_gating_execution_and_restore）。

## 8.3 实测证据（同一会话、同机、交错测量）

| 观察 | 数据 | 结论 |
|---|---|---|
| 精确界排序失败 | 16K niah-01 p16：quality 0（生成 1234567890） | 上界对 key 范数大的页系统性高估，与 softmax 质量错位 → 否决 |
| 门控开销本身不是瓶颈 | 同上、sel_pages=0（只留 local）ttft 290s ≈ dense 312s | 失败在**信号**不在机制；页均值元数据几乎免费 |
| MoBA 页均值可用 | campaign 16K：p16 6.83x q=1.0、p24 6.85x q=1.0 | 页均值是免训练可用的门控信号 |
| 8K 最优区间 | campaign 8K：p16 9.38x q=1.0（scalar 9.73x，qg-2d 8.77x） | 门控≈标量 gather，但 sem 1.1s→0 |
| 多针任务门控胜出 | mn：p16 3.77x q=0.83 vs scalar 2.83x q=0.67 | local 窗保住针的近邻，纯 top-k 行表会丢 |
| 32K 默认窗覆盖不足 | 32K p16 q=0.0、p24 q=0.25；local_rows=32 时 5.64x q=1.0 | 页预算买不回覆盖；32K 需加宽 local，加速来自 local 不是选择 |
| 32K 成本=页均值重扫 | ttft 随 ubatch 数增长；门控把整段 K 经 cont(permute) 复制（95MB×28×61≈160GB 流量） | 下一步：view+mul_mat+page-sum 去掉拷贝、按层间隔门控 |

（上表数字来自 enchmarks/results/ar006-campaign/matrix.md（同会话 dense 参照）。
本机绝对延迟在会话间波动可达 3×，故只引用同会话比值；32K 与 16K 的 SRS 目标未达成，
根因与下一步见 REPORT.md。）

## 8.4 局限

- 每头掩码在 ggml 中不可表达（KQ mask 广播 head 维）→ 我们做**全头并集**选择，
  比逐头最优更保守（多选）；
- head 路由（检索/流式两分支）需要两次 attention 子图，收益待 32K+ 实测后评估；
- 页尺寸固定 32（对齐 AR004/AR005 块），更细的页会增加 row_table 与 top-k 宽度。
