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

---

# AR007 增补（v4）：预算不是越大越好 —— 预算-质量曲线、结构锚点与去拷贝页打分

AR006 把门控搬进图内后，剩下的问题是**选谁**和**选多少**。AR007 的出发点是两条实测异常：
预算从 0.05 提到 0.25，质量反而下降；32K 的门控开销随 ubatch 数线性增长（页均值把整段 K
复制一遍）。前者催生预算-质量曲线与结构锚点，后者催生 view+mul_mat 的页打分。

## 9.1 预算-质量曲线是可测的，而且不单调

`benchmarks/results/ar007-sweep/`（1.5B q4_k_m，16 线程，同一会话，lexical 信号）：

| 长度 | b=0.05 | b=0.08 | b=0.12 |
|---|---|---|---|
| 3947 | q=0.50（210 tok） | **q=1.00**（322 tok, 3.70s） | q=1.00（482 tok, 3.63s） |
| 7809 | **q=1.00**（385 tok, 3.35s） | q=1.00（641 tok, 5.88s） | q=1.00（957 tok, 7.67s） |
| 15576 | **q=1.00**（793 tok, 6.99s） | q=1.00（1256 tok, 11.0s） | q=0.50（1881 tok, 15.7s） |
| 31080 | q=0.89（1565 tok, 13.7s） | q=0.67（2510 tok, 14.8s） | q=0.75（3752 tok, 26.9s） |

三条结论：

1. **质量对预算非单调**。16K 在 b=0.12 掉到 0.5、32K 在 b=0.08 掉到 0.67，
   因为预算变大时锚点（按比例扩）与均匀采样块一起变多，把真正的证据块挤出预算；
   这不是"预算不够"，而是"预算分配变了"。
2. **不同长度的最优点不同且差很多**：4K 需要 8%，8K/16K 只需 5%。
   一个全局常数预算在任何方向上都会浪费或掉质，这就是 FR-3 存在的理由。
3. **速度排序也不能只看预算**：4K 的 0.08 与 0.12 只差 1.7% 时间（噪声量级），
   所以脚本的推荐规则是"最快格点 1.15 倍以内的最省预算"，而不是"最快"或"最小"。

曲线落成 `budget_curve.txt`（行式 `key = value`，带 `model_sha256` 指纹），
planner 用 **ceiling 语义**（提示落在两点之间时借用更长的那次测量），
`budget_source` 审计列区分 `model|override|default|clamp`。

## 9.2 结构锚点：把"预算"从数量问题变成覆盖问题

- **锚点集合**：sink（块 0，注意力汇）、末块（承载最后一个 token 的 logits 依赖）、
  问句词锚点（df 低、长度≥6/含数字/CJK 的词各取 lexical 最高块）、
  位置采样（`max(4, round(0.25·target_blocks·stride_scale))` 个块）。
- **裁剪优先级** sink > recent > quote > stride：永不删 sink 与末块，
  stride 先让位给问句锚点。
- **span growing**：贪心填充时给"与已选块相邻"的候选 +0.5 分。证据句常常跨块边界
  （needle 句起于块中段），纯 top-k 会选到离散点、切断答案 span——实测低预算下
  q 从 0 回到 1.0 的关键一步。
- **MMR**：候选与已选块的最大 Jaccard 相似度超过 `mmr_threshold` 时按 λ 降权，
  抑制重复段落（模板化日志、重复标题）刷满预算。实现在 `max_sim[]` 上增量更新，
  32K（≈1e6 次集合操作）< 0.5s。

## 9.3 同会话验收（dense vs 曲线 vs AR006 固定 0.25）

`benchmarks/results/ar007-campaign/REPORT.md`（dense 每数据集实测一次，交错测量）：

| 数据集 | dense | 曲线（自动预算） | AR006 b=0.25 |
|---|---|---|---|
| 8K | 48.8s | **1.67s, 29.1×, q=1.00** | 9.1s, 5.4×, q=0.875 |
| 16K | 199.5s | **5.18s, 38.5×, q=1.00** | 19.9s, 10.0×, q=0.356 |
| 32K | 384.1s | **6.96s, 55.2×, q=0.89** | 21.9s（clamp 到 4096 tok）, 17.5×, q=0.750 |
| multi-needle | 53.4s | 1.75s, 30.5×, **q=0.33** | q=0.39（b=0.25）；AR006 门控 q=0.83 |

**诚实结论**：曲线选择器在单针检索族（explen 4K–32K）上全面超过固定预算，
同预算下更快、同速度下更准；但在多针族（`deep/multineedle`）上掉到 0.33，
且这一族的质量对预算不敏感（0.08/0.12/0.25 分别 0.33/0.50/0.39）。
根因：多针文档里问句词在大量相似句子上都有高词法分，选择级稀疏的块粒度
（32 token）无法在 5% 预算下同时覆盖 6 个事实的邻域；AR006 的图内页门控靠
local 窗 + 页均值反而能保到 0.83。因此 **`budget_curve` 默认不启用**（只有
conf 显式给出路径才生效），多针场景应使用 AR006 门控（`qg_bound=on`）。

## 9.4 页打分去拷贝：32K 门控从 67.9s 降到 25.4s

AR006 的页均值是 `cont(permute(k))` + `pool_1d`：每个 ubatch、每层都把整段 K
复制一遍（32K：95MB × 28 层 × 61 ubatch ≈ 160GB 流量）。AR007 换成
`q_heads = cont(view_3d(q))` + `mul_mat(k_p, q_heads)` + 页内 `sum_rows`，
语义等价（页内求和 = 页均值 × q，差常数 1/bs），只物化 `h_kv·d` 个数：

| 32K bound p24+local32 | TTFT | q |
|---|---|---|
| AR006（cont+pool） | 67.9s | 1.00 |
| AR007（view+mul_mat+page-sum） | **25.4s** | 1.00 |

降幅 63%（FR-4 目标 ≥30%）。16K p16 也从 23.7s 降到 21.8s，q 保持 1.00。

**回归教训**：`ggml_mul_mat(a,b)` 只保证 `b->ne[2] % a->ne[2] == 0`（a 为广播方）。
缓存张量是 `[d, h_kv, n_kv]`，裸 `k` 直接参与乘法会在 32K 触发
`GGML_ASSERT(ggml_can_mul_mat)` abort；必须用 `permute(k,0,2,1,3)` 的 `[d, n_kv, h_kv]`
view。默认 local 窗会把短提示整段覆盖成 `local-only` 旁路，所以单测里必须显式
`qg_local_rows=32` 才能压进页打分分支——已加入 `bound_gating_execution_and_restore`。

## 9.5 局限

- 曲线是**任务族专用**的：单针 NIAH 测出来的 budget 不迁移到多针/摘要类任务，
  目前靠"默认不启用 + 文档标注适用域"管理，不做自动任务识别；
- 锚点预算按比例扩，在高预算下会稀释证据（9.1 的非单调性），
  下一步是把锚点数量的绝对上限（例如 ≤16 块）与预算解耦；
- 页打分的 `sum_rows` 仍是 O(n_kv) 的 page 求和，但没有整段拷贝，CPU 上已够快；
  GPU 上还需要一次 per-head 规约的等价实现验证（T009）。

---

# AR008 增补（v5）：判别符、短语与组合 —— 从"更多预算"转向"正确证据"

AR007 的结论是"预算要按长度调"，但它同时留下一个刺眼的失败：多针集 q=0.33。
本轮先做根因复核，再谈优化，结论推翻了 AR007 的假设。

## 10.1 多针失败的真正原因：判别符被过滤掉了

```
问句：What is the secret passcode for vault 5?
文档：... The secret passcode for vault 5 is CRIMSON-LYNX-7624. ...
      （vault 4/5/6 各出现一次，只有数字不同）
```

`selector.cpp` 的查询词过滤是 `term.size() >= 2 || CJK`，**单个数字被静默丢弃**。
于是问句词退化成 {secret, passcode, vault}，三个兄弟块的词法分完全相同，
选中哪个只由块序号决定 → 正确率 1/3 = 实测 0.333。AR007 把它归因为"块粒度覆盖不足"，
但同一份数据里 `vault 5` 唯一出现（mn-02.txt 位置 50.1%），覆盖从来不是问题。

修复分三步，每一步都被实测逼出来：

| 步骤 | 做法 | 结果 |
|---|---|---|
| 1 | 查询 bigram 作独立词项，保留单字符数字，块级短语 IDF | q 0.33 → 0.67 |
| 2 | 跨块 bigram（块 b 末词 + 块 b+1 首词，记给两个块） | 0.67 → 0.83 |
| 3 | 以**最高 IDF 短语**（`vault 5` 而非模板 `for vault`）锚定其全部命中块 | 0.83 → **1.00** |

第 3 步的教训最贵：按"块短语总分"取 arg-max 时，模板短语 `for vault`
在三个兄弟块里都有，分数被拉平；而第 2 步引入的跨块命中又可能落在相邻块上，
于是锚点落在错误的事实上。只有 IFF 最高的短语才是判别单元。

代价（同会话、同预算 0.05、repeats=3）：

| 数据集 | `phrase_weight=0`（AR007） | `phrase_weight=0.5` | 质量 |
|---|---:|---:|---|
| 8K | 2271 ms | 2229 ms | 1.000 → 1.000 |
| 16K | 4826 ms | 4907 ms | 1.000 → 1.000 |

即 ±2% 以内，满足需求上限；短语信号只在"有判别短语"时存在，否则完全不构建。

## 10.2 组合：同一份 prompt，不同路径差 3 倍

仓库现在有 4 条完整执行路径，实测（同会话，dense 每数据集一次）：

| 数据集 | selection | gate-p16 | gate-p24+local32 | dense |
|---|---:|---:|---:|---:|
| 4K | **5.5 s / q=1.00** | 18.5 s / 1.00 | 14.5 s / 1.00 | 66.3 s |
| 8K | **5.7 s / q=1.00** | 25.7 s / 1.00 | 25.1 s / 1.00 | 141.9 s |
| 16K | **15.9 s / q=1.00** | 57.3 s / 1.00 | 60.1 s / 1.00 | 302.7 s |
| 32K | 23.1 s / **q=0.875** | 63.8 s / q=0.25 | **66.8 s / q=1.00** | 852.1 s |
| 多针 | **8.2 s / q=1.00** | 38.2 s / 1.00 | 16.7 s / 1.00 | 92.0 s |

三点观察：

1. **选择级稀疏在大多数格点同时更快更准**（4K/8K/16K/多针），因为它把 FFN 与
   attention 的工作量都按准入 token 数降下来了；门控只降 attention 行数，
   全量 token 仍要过 28 层 FFN；
2. **32K 是唯一需要升级的格点**：选择级 q=0.875（floor-missed），门控 p24+local32
   q=1.00，代价 2.9× TTFT。这是一个可度量的质量-延迟折中，而不是"门控更好"；
3. 因此"固定用某条路径"必然在某个格点上浪费或掉质 —— 这正是组合路由的存在理由。

## 10.3 组合路由与 Pareto 闸门

路由表由实测生成（`portfolio_matrix.py`），规则：**达到质量底线（均值与最差例都 ≥0.95）
的组合里取最快；若都达不到，取质量最高**。结果是 selection 在 4 个格点胜出、
32K 升级到门控。表带 `model_sha256` 指纹，缺失/不符即回退 selection。

更关键的是**回退机制**：`check_portfolio.py` 以"同会话最佳固定组合"为基线，
要求路由组合 Pareto 不劣（质量 ≥ 基线 -0.01 且 TTFT ≤ 基线 ×1.05）。
注入一个更慢的臂（16K → gate-p16）时闸门必须 FAIL，实测确实 FAIL。
这把用户要求的"如果创新不能提升组合性能就回退"变成了可执行的检查，而不是口头承诺。

## 10.4 矩阵本身抓到的缺陷

路由初版从 per-cell `Config` 读 `qg_bound/qg_sel_pages`，但这些字段属于
`BenchmarkConfig`，于是**门控臂实际没开门控**（`qg_source` 为空、16K 质量 0.356
等于纯选择级），却按门控的耗时记账。矩阵记录 `qg_source` 才暴露出来。
修复后 4K/8K/16K/32K 门控全部 q=1.00。教训：组合实验必须记录"这条路径到底有没有生效"
的审计列，否则测的是另一条路径。

## 10.5 局限

- 组合表是模型/数据集绑定的，换模型必须重测（指纹已强制）；
- 32K 的升级是用延迟换质量（36.8× → 12.8×），偏好速度的用户应保留 selection 并接受 q=0.875；
- 无判别短语（纯语义/改写问句）时短语信号为空，表退化为 `default`，
  仓库没有该 regime 的数据集，因此不宣称提升；
- 多事实"一问多跳"覆盖（submodular facility location）仍未做：仓库缺对应数据集，
  无法诚实评估，留作后继 AR。
