# [AR006] AR 详细设计

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| 主题 | in-graph-query-aware-sparse-prefill（图内精确界查询感知稀疏 prefill） |
| 关联 srs.md | ./srs.md |
| 基线 | AR005（commit 见 tasks.md），llama.cpp `3bb63cbcbfb62056c84fe9e521b0f3d756d29c32` |
| 日期 | 2026-10-25 |

## 1. 方案总览

### 1.1 与基线的差异（一张图）

```text
AR005（现状）                              AR006（本设计）
planner: 嵌入/词法 scores → 分组行集        planner: 只给预算（页数/local/pool）
   ↓（静态、层间共用、需嵌入 1-3s）             ↓
patch: 每 query 组 gather 自己的行集  ──►   patch: 每 ubatch 每层：
   （行重复搬运，qg ttft ≈ 2× scalar）          1) K 页 min/max（真实 K，图内）
                                              2) 分片上界打分（精确上界，免训练）
                                              3) top-k 页（排除 local 页）
                                              4) 一次 gather（local∪selected∪sink）
                                              5) 因果掩码从 kq_mask 按行 gather
```

核心思想与出处：
- **Quest**（arXiv 2406.10774）：页级 K 的 min/max 给出 `q·k` 的精确上界，避免训练 router；
  我们把它从 decode 扩展到 **prefill 的 ubatch 级选择**，并直接在 ggml 图内实现（无 host 同步）。
- **MoBA**（arXiv 2502.13189）：查询分片 + 块级 top-k + **每片必看自身块**；我们的 local 分支即该语义。
- **NSA**（arXiv 2502.11089）：三分支（local/selected/compressed）；我们用 local（必留）+ selected
  （上界 top-k）+ compressed（AR004 pool 延续）显式三分支。
- **MInference/PyramidKV**：层/头预算差异化——AR006 由“逐层独立的真实上界”天然获得层自适应，
  无需静态先验；head 路由作为可选增量（FR-6）。

### 1.2 备选方案对比（设计阶段决策）

| 方案 | 选择机制 | 质量 | TTFT | 实现风险 | 结论 |
|------|---------|------|------|---------|------|
| A 嵌入 2D（AR005） | bge 组×块余弦 | 需大预算（512/组） | 含 1-3s 嵌入 + 4×gather | 低 | 保留为可选（回归） |
| B 上界 + host 预选 | 需逐层 host 同步的 K 统计 | 高 | 同步开销大 | 中 | 否决（违背单图执行） |
| C 上界 + 图内 top-k（本设计） | 真实 K 页 min/max 上界 | 高（精确上界） | 1×gather，元数据<15% | 中 | **采纳** |
| D 阈值法（Quest decode 式） | 上界阈值 vs 末 query | 预算不可控 | 1×gather | 低 | 作为 C 的降级：top_k 不可用时 |
| E 静态 MInference 模式 | 离线 pattern | 中（与提示无关） | 1×gather | 低 | 仅作 head 路由先验参考 |

### 1.3 已验证的 ggml 能力（设计前置，源码证据；T001 仍需运行时验证）

- `ggml_pool_1d`（MAX）CPU 后端实现：`ggml-cpu.c` `GGML_OP_POOL_1D`；
- `ggml_argsort` / `ggml_argsort_top_k` / `ggml_top_k`：CPU 后端 `GGML_OP_ARGSORT/TOP_K`；
- `ggml_neg`、`ggml_sum_rows`、`ggml_get_rows`（src1 支持多维/动态张量，现有 qg_idx 已用）；
- 广播：`ggml_mul/max/add` 支持 ne[2]/ne[3] 维=1 广播（现有 patch 已依赖）。

> 若 T001 运行时探针任一不满足，按 D（阈值法）或“页分数张量 host 读回一次/层”备选实现。

## 2. 功能点（FP）分解

### FP1 K 页 min/max 元数据（FR-1）

**图内构造（每层，每 ubatch）：**
1. `k` 视图 `[d, h_kv, n_kv, 1]` → `view` 为 `[d, h_kv, bs, n_pages]`（n_kv = bs·n_pages，尾部不满页
   用 `n_pages = n_kv/bs`，残余行并入最后一页）；
2. `permute(2,1,0,3)` + `cont` → `[bs, h_kv, d, n_pages]`，`permute` 后 `ne[0]=bs` 连续；
3. `kmax = pool_1d(MAX, k_cont, bs, bs, 0)` → `[1, h_kv*d, n_pages]`；
   `kmin = -pool_1d(MAX, -k_cont, bs, bs, 0)`；
4. reshape/permute 为 `[d, h_kv, n_pages]`（`KMAX`/`KMIN`，仅用于打分，不回写）。

**代价**：一次 K 的 `cont` 拷贝（≈ K 大小内存流量）+ 两次池化。实测目标 <15% 注意力耗时（NFR）。

**开关**：`llama_set_prefill_qg_bound(ctx, enabled, sel_pages, local_pages, pool_pages)`；
关闭时该分支不建节点（dense 字节等价）。

### FP2 分片上界打分与 top-k（FR-2）

**查询表示**：`q_mean = sum_rows(q_view[d,h,n_tok])`（均值；不做除以 n_tok——正比例不改变排序）。
可选 `max` 模式（union 更保守）留作配置，默认 mean。

**上界**：
```text
p1 = broadcast_mul(q_mean[d,h,1], KMAX[d,h,n_pages])   // q_d · kmax_pd
p2 = broadcast_mul(q_mean[d,h,1], KMIN[d,h,n_pages])
u  = sum_rows(max(p1,p2))                              // [1,h,n_pages] → [h,n_pages]
```
- 逐维取最大 ⇒ `u(q,p) ≥ q·k, ∀k∈page p`（srs 3.2 定义）。
- **排除 local/sink 页**：加常量偏置 `bias[n_pages]`（input 张量，构建期已知，local 页 = -INF），
  `u' = u + bias`，避免 selected 与 local 重复计数。
- **top-k**：`ggml_argsort_top_k(u_head_union...)`——先对头取并集：`u_union = max over h(u)`（保守：
  任一头认为重要则选），得到 `[n_pages]` 分数 → top-k → `sel_pages[k_sel]`（I32）。
  注：逐头掩码在 ggml 中不可行（mask 仅广播到 head 维），故选择必须为分片/ubatch 级并集。
- **行索引展开**：`row_idx[k_sel*bs] = sel_pages[:,None]*bs + arange[None,:]`（input `arange[bs]`；
  `mul`+`add` 广播）→ reshape 一维。
- **参数校验**：`bs` 必须 = 32（页粒度固定，与块选择一致）；`sel_pages*bs + local_rows ≤ 4096`
  （超出裁剪并审计）；`n_kv < bs` → Unsupported → 上层回退。

**审计**：`qg_source=attention-bound`；`bound_sel_pages_mean`（每 ubatch 平均选中页数，来自
`llama_get_prefill_qg_stats` 新增只读接口，host 侧每 ubatch 末尾读取一次计数，不读分数）。

### FP3 单次并集 gather + 掩码（FR-3）

**行集合（不相交）**：
- `local_rows`：页集合 `{0} ∪ {p | page p ∩ [lo - W, hi) ≠ ∅}`，W = `local_pages*bs`（默认 W=qcs）；
- `selected_rows`：FP2 展开行（已排除 local 页）；
- `pool_rows`：AR004 既有 pooled 行（可选，追加在其后）。

**gather（每 ubatch 每层一次）**：
```text
idx = concat(row_idx_selected, host_local_idx)      // I32 输入张量（local 部分构建期已知）
k_g = get_rows(permute(k,0,2,1,3), idx) → cont(permute(...))   // 复用 AR005 gather 代码
v_g = 同构（含 v_trans 归一）
m_g = get_rows(mask_p, idx)                         // permute(kq_mask,1,0,2,3) 后再 gather
```
- 掩码天然保持因果性（行来自 kq_mask，列为本 ubatch 查询）；
- **等价性单测**：与“逐组独立 gather + 独立 mask”的参考实现在同一随机 K/V/掩码上对比，
  `max abs diff < 1e-5`；并验证无重复行（集合不相交断言）。
- 顺序无关：softmax 对行排列不变（同权重同值），因此 top-k 返回顺序不要求有序。

**性能收益**：行搬运次数从 4×（每组一次，含 own-span+extras）降为 1×（local+selected），
目标 qg ttft ≤ scalar ×1.15。

### FP4 三分支预算（FR-4）

配置键与默认：

| 键 | 含义 | 默认 | 语义 |
|----|------|------|------|
| `qg_bound` | 图内上界门控开关 | false | 与 `query_gating` 互斥（bound 优先） |
| `qg_local_rows` | local 窗口行数 | 0（= ubatch 大小） | 0 表示“本 ubatch 自身跨度” |
| `qg_sel_pages` | selected 页预算 | 8（=256 行） | 0 关闭 selected |
| `qg_pool_pages` | 压缩行预算（页数） | 0 | 0 关闭 pool（沿用 AR004 pool 开关） |

审计列：`local_rows_mean`、`sel_rows_mean`、`pool_rows_mean`、`bound_sel_pages_mean`。
非法组合（三预算皆 0 且开关开）→ 规划层 `InvalidArgument`。

### FP5 免嵌入门控（FR-5）

- 默认 `qg_score_source=attention-bound`（新增配置键），`semantic_score_ms=0`；
- `qg_score_source=embedding-2d|lexical-1d` 保留 AR005 路径（回归用）；
- 显式 `embedding_model` + `selector-embedding on` 才加载嵌入；加载失败硬错（沿用 AR005）。

### FP6 head 路由（FR-6，可裁剪）

由 `u` 的逐头分布导出 streaming/retrieval 分类：
`spread_h = (max_p u_h - mean_p u_h) / (|mean_p u_h| + eps)`；低于阈值的头标记 streaming。
- 实现限制：ggml mask 不广播到 head 维 ⇒ 逐头可见性需拆两次 attention（retrieval/streaming 各一次
  matmul/softmax，再沿 head 维 concat）。**先作为 P2 增量实现**；不可行或收益不足时，退化为
  “head 路由只影响预算分配（所有头共享并集）”，并如实记录限制。
- 默认 off；off 时与 FP4 逐字段一致。

### FP7 复现与产品化（FR-7）

- `benchmarks/matrix/bound/*.conf`：上界门控注册表（页预算 × local × pool）；
- `benchmarks/matrix/models/qwen2.5-7b-instruct-q4_k_m.conf.tmpl`（模板，含 n_ctx/线程/显存建议）；
- `docs/server-repro.md`：≥32GB 服务器流程（模型 SHA256、命令、预期时长/区间、失败排查）；
- `expected.csv` 由实测生成（机器指纹：CPU/线程/commit/model_sha），`check_expected.py` 门控；
- README 首屏：`git clone && ./repro.sh smoke`。

## 3. 接口与数据结构

### 3.1 C API（patch 新增，libllama 公开）

```c
// 图内上界门控：enabled=1 时每个 ubatch 每层选 sel_pages 个页（bs=32），
// local_pages 为 local 窗口页数（0 = 启用默认“本 ubatch 自身跨度”）。
// 返回 0=Ok, 2=InvalidArgument, 9=Unsupported（页粒度/后端不支持）。
LLAMA_API int32_t llama_set_prefill_qg_bound(
        struct llama_context * ctx, int32_t enabled,
        int32_t sel_pages, int32_t local_pages, int32_t pool_pages);

// 只读统计：最近一次 prefill ubatch 的选中页数（-1 = 未启用）。
LLAMA_API int32_t llama_get_prefill_qg_sel_pages(struct llama_context * ctx);
```

### 3.2 C++ 侧接口（edge_prefill）

```cpp
struct BoundGatingOptions {
    bool enabled = false;
    uint32_t sel_pages = 8;      // 32 行/页
    uint32_t local_rows = 0;     // 0 = ubatch 自身跨度
    uint32_t pool_pages = 0;     // 0 = 关闭
};
// LlamaBackend:
Status set_qg_bound(const BoundGatingOptions& opts);   // -> Unsupported/InvalidArgument
int32_t qg_sel_pages() const;
```
- `plan` 层映射：`qg_bound=true` 时不再生成行集（`qg_group_rows` 空），仅下发预算；
  `Plan.reasons` 记 `qg-source=attention-bound`；
- 与 AR005 行集路径互斥：`qg_bound && query_gating` → bound 优先 + reason `qg-bound-overrides-rows`。

### 3.3 审计列（Sample/raw.csv 追加）

`qg_source`（已有，新增取值 `attention-bound`）、`bound_sel_pages_mean`、`local_rows_mean`、
`sel_rows_mean`、`pool_rows_mean`。

## 4. 关键流程（时序）

```text
planner: budget → selected tokens（沿用 AR001-AR005 选择）
  └─ qg_bound=true → 不做分组行集；下发预算给 backend（RAII，chunk 前后 set/clear）
sparse_prefill: 分块 prefill（AR004 chunk scheduler 不变）
  └─ 每 ubatch: backend.set_qg_bound(...) → llama_decode
        └─ graph per layer:
            FP1 元数据 → FP2 上界+top-k → FP3 union gather+mask → attn
    每 ubatch 末: 读取 sel_pages 计数（仅计数，不同步分数）→ stats
```

## 5. 错误处理与降级链

| 场景 | 行为 | 审计 |
|------|------|------|
| 后端不支持 top_k/动态 get_rows | 回退阈值法（Quest decode 式）；仍失败 → AR005 行集路径 → scalar → dense | `qg-bound-unsupported` |
| 页粒度≠32 / n_kv<32 | Unsupported → 回退 | `qg-bound-page-unsupported` |
| 上界非有限 | 该 ubatch dense attention | `qg-bound-nonfinite` |
| 并集超 4096 行 | 裁剪 selected（保序）| `qg-bound-clipped` |
| flash attention 开启 | 禁用（沿用现有约束）| `qg-bound-flash-disabled` |

## 6. 测试设计

### 6.1 单元/集成（无模型，快速套件）

| 用例 | 覆盖 | 断言 |
|------|------|------|
| UT-1 上界属性测试 | FR-2 | 随机 K 页/q（≥1000 组）：`u ≥ q·k ∀k∈page` |
| UT-2 页 min/max 正确性 | FR-1 | 与 host 参考实现逐元素一致（含负数、极值） |
| UT-3 top-k 选择与 bias | FR-2 | local 页永不被选；选中数=min(k,n_pages−local) |
| UT-4 并集 gather 等价性 | FR-3 | 与逐组参考实现 attention 输出 max abs diff <1e-5；无重复行 |
| UT-5 掩码因果性 | FR-3 | 未来行全 -INF；local 行必可见 |
| UT-6 预算校验/审计列 | FR-4 | 非法组合 InvalidArgument；行数与配置一致 |
| UT-7 默认路径字节等价 | FR-1/FR-8 | 全关时 logits max abs diff <1e-5（既有回归） |
| UT-8 embedding 可选回归 | FR-5 | embedding-2d 路径与 AR005 结果一致 |
| UT-9 head 路由确定性 | FR-6 | 同输入两次输出逐字节一致（若实现） |

### 6.2 真实模型（需 EDGE_PREFILL_TEST_MODEL）

| 用例 | 数据集 | 断言 |
|------|--------|------|
| IT-1 上界召回 | len16k/len32k（真实 K）| 选中页与暴力逐行 top-k 页召回率 ≥95% |
| IT-2 4K/8K 质量 | len4k/len8k | q ≥ 0.9（4 用例） |
| IT-3 16K/32K 质量+净加速 | len16k/len32k | q ≥ 0.9；净加速 ≥8×（16K）/≥10×（32K） |
| IT-4 mn8K | multineedle（12 用例）| 平均 q ≥ 0.66 |
| IT-5 qg vs scalar 成本 | len32k | qg ttft ≤ scalar ×1.15 |
| IT-6 可复现 | len32k | 同命令两次 ttft 偏差 ≤5%，质量逐位一致 |

### 6.3 异常用例（EX）

| 用例 | 触发 | 期望 |
|------|------|------|
| EX-1 后端缺 op | 强制禁用 top_k（测试钩子）| 回退阈值法，reason 审计 |
| EX-2 页粒度不足 | 提示 <32 token | Unsupported → 回退，非零不会崩溃 |
| EX-3 非有限上界 | 注入 NaN K（单测）| dense 回退 + reason |
| EX-4 并集超限 | k_sel 极大 | 裁剪 + reason，结果仍有限 |

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| top_k/get_rows 动态索引在 CPU 上的调度开销 | 抵消收益 | T001 实测计时；超预算改阈值法 |
| K `cont` 拷贝使元数据开销 >15% | TTFT 回退 | 页粒度扫描（16/32/64）；或用 `pool_1d` 直接读取 permute 视图（若后端允许非连续 ne[0] 则省拷贝） |
| 均值 query 的上界选择与真实注意力偏差 | 质量下降 | 可选 max-query 模式；预算扫描；IT-1 召回率硬门 |
| 与 pool 分支叠加的掩码复杂度 | 实现错误 | 分阶段：先 local+selected，pool 后接；UT-4 等价性 |
| CUDA 后端 op 覆盖不全 | GPU 不可用 | 显式 Unsupported 回退 + 记录；CPU 为必过路径 |
| 7B 实测缺失 | 说服力 | 模板 + 服务器指南 + 真实小跑区间（若服务器可得） |

## 8. 交付物

- patch：`patches/llama-prefill-qg-bound.patch`（或并入现有 patch，保持最小可审查）；
- C++：`BoundGatingOptions`/backend/planner/benchmark 接线、审计列；
- 配置：`benchmarks/matrix/bound/*.conf`、7B/14B 模板；
- 文档：README 一键复现、`docs/architecture.md` v0.5、`docs/research.md` v3（SOTA 对比与论证）、
  `docs/server-repro.md`、`benchmarks/matrix/expected.csv`；
- 结果：`benchmarks/results/ar006-*`（raw.csv/report.md/env.json/matrix.csv/md）；
- 发布：tag `ar006-v1.0` + Gitee 同步。

---

---

## 9. 实现对照（as-built，2026-10-25）

实现与第 1~5 节的设计有四处经实测修正的偏差，全部以真实运行数据为依据：

1. **打分函数：精确界 → 页均值。** 设计选的是 Quest 式 `u = sum_d max(q*kmax, q*kmin)`。
   实现完成后 16K 实测 quality=0（页选择偏向大范数 key）；同时 `sel_pages=0` 的
   对照 wall(290s) 约等于 dense(312s)，证明开销不在机制而在信号。已改为
   `score = q * mean(K_page)`（MoBA 门控），16K p16/p24/p32 quality=1.0。
   代码里保留注释说明否决原因；上界实现已删除。
2. **页→行展开：整数指数运算 → row_table 查表。** 设计假设可用 `arange*32+add`
   产生行号，但 ggml 没有 F32→I32 cast（T001 探针确认），该路径不可达。
   实现改为 host 侧一次性写 `row_table[32,n_pages]`（元素 `(r,p)=p*32+r`），
   图上 `get_rows(row_table, sel)` 展开为 `[32,k_sel]` 后 reshape 成行号序列。
3. **掩码转置：逐层 → 每图一次。** 设计放在 attention 调用处；实现中逐层
   `permute+cont` 会多搬 2*n_tok*n_kv 字节/层，改为 `llama-graph.h` 新增
   `bound_mask_p`，在 set_input 阶段预转置一次，门控分支只做限行 gather。
4. **head 路由（FR-6）暂缓。** ggml 的 KQ mask 只广播到 head 维，逐头不同页集
   需要拆两次 attention 子图；首版用“全头并集 + MAX 归约”保守实现（多选不少选），
   T007 保持 pending 并在 SRS 标注为后续工作。

另外两条实现细节，避免 ST 阶段误读：

- **local 窗口语义**：`qg_local_rows=0` 表示窗口 = ubatch 自身跨度 `n_tok`
  （不是 0 行）；`local_start_page` 同时用于 bias（排除 local 页）与 local 行集，
  两者必须一致，否则同一页会同时进入 local 与 selected。
- **4K/8K 退化**：当 `local_start_page==0`（local 覆盖全部 KV）时不建 selected
  分支（`k_sel=0`），图等价于 local-only gather。这是质量安全网，也解释了
  4K/8K 上不应出现加速。
