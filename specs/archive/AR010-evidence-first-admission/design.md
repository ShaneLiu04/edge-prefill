# [AR010] 详细设计：证据优先准入

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 关联 srs.md | ./srs.md |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.7） |
| 日期 | 2026-10-27 |

## 1. 设计驱动因素与 SOTA 取舍

| # | 驱动 | 量化目标 |
|---|---|---|
| D1 | 先验可独立准入 → 长上下文预算抽走 | 32K 零证据块占比 62% → ≤10% |
| D2 | 归一化不在 [0,1]，权重语义失效 | 所有信号 ∈[0,1]（单测断言） |
| D3 | 填充停止依赖数值巧合 | 显式 `stop_reason`，4 种原因可审计 |
| D4 | 32K 必须靠门控兜底 | 选择级 32K q≥0.99，组合切回 selection |
| D5 | 只有 CPU 验证 | 本机 CUDA 后端一致性通过 |

**SOTA 参考与取舍**

| 来源 | 核心思想 | 取用/否决 |
|---|---|---|
| Retrieval 的"先召回后排序"两阶段（recall then rank） | 召回与排序用不同准则，避免排序模型改变召回集合 | **采纳**：evidence = 召回准则，prior = 排序准则 |
| BM25 的饱和项（term saturation, k1） | 词频收益递减，避免单信号无限放大 | **采纳**：`gate(e)=min(1, e/0.25e_max)`，证据越弱先验贡献越小 |
| Lost in the Middle（Liu et al.） | 位置效应真实存在，中间信息利用率低 | **部分采纳**：position 保留为**排序**先验，不作为准入依据 |
| H2O / SnapKV 的预算恒定假设 | 总预算固定，按重要性分配 | **否决**：端侧更关心绝对算力，证据耗尽就该停（`fill_priors=0`） |
| RAG 的 "no answer" 门控 | 无证据时不强行生成 | **借鉴**：无证据时不再为"填满预算"而搬运噪声块 |
| DuoAttention / NSA 的粗-细两段 | 粗筛后精排 | 留作后继（图内路径） |

**关键结论**：这不是调参，而是把"可准入"与"如何排序"两件事分开。AR007/AR008/AR009
都在往 `fused` 里加东西（短语、覆盖、位置、多样性），但没有任何一层区分
"这个块有证据"与"这个块位置好/信息密度高"。先验对零证据块给出正分，是长上下文
退化的根因；而"填满预算"依赖 bug，是质量不稳定的第二层根因。

## 2. 方案对比

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A：只修 minmax | 拷贝上下界 | 一行修复 | 4K/多事实立刻回归（填不满预算） | 否决（必要但不充分） |
| B：先验权重调小 | 把 diversity/position 权重降到 0.05 | 简单 | 长上下文仍需先验（sink/recent），权重是长度相关的，定值不可扩展 | 否决 |
| C：硬门限（evidence>0 才准许入选） | 先验只在有证据时生效 | 语义清晰 | 阈值对连续量太脆；微小词法重合也会放大先验 | 部分采纳（用饱和替代硬门限） |
| D：证据/先验分离 + 饱和门控 + 显式停止 | `fused = e + gate(e)·p`，填充有停止语义 | 与长度无关；可审计；可关闭结构填充 | 需要新增 2 个开关与 2 个审计列 | **采纳**（FR-1/2/3） |
| E：学习型准入 | 训练分类器判断"块是否有证据" | 潜在最优 | 样本不足、不可解释、端侧不可接受 | 否决 |

## 3. 详细设计

### 3.1 归一化（FR-1）

```cpp
const auto [lo_it, hi_it] = std::minmax_element(...);
const double lo = *lo_it, hi = *hi_it;      // 必须拷贝：它们是同一向量的迭代器
const double range = hi - lo;
for (auto& v : *values) v = (v - lo) / range;
```

单测要覆盖三种最小值位置（开头 / 中间 / 结尾），断言结果严格落在 [0,1] 且
`min==0, max==1`。旧实现只在最小值位于向量首元素时"碰巧正确"。

### 3.2 证据 / 先验分离（FR-1.2/1.3）

```text
evidence[b] = w_lex·lexical[b] + w_phrase·phrase[b] + w_draft·draft[b]
prior[b]    = w_div·diversity[b] + w_pos·position[b]
e_max       = max_b evidence[b]
sat(e)      = min(1, e / (kPriorSat · e_max))        // kPriorSat = 0.25
fused[b]    = evidence[b] + (prior_gate ? sat(evidence[b])·prior[b] : prior[b])
```

- `kPriorSat = 0.25`：证据达到最强的 25% 即获得完整先验加权，既不惩罚次强证据，
  又让"接近零证据"的块拿不到先验。常数写死并注释理由，避免再加旋钮。
- `prior_gate=0` 与旧行为（修正后的加法融合）一致，用于消融与回退。

### 3.3 填充与停止（FR-2）

```text
while selected_tokens < target:
    best = argmax_{b∉sel} (fused[b] + span·adjacent[b] + coverage_weight·gain[b])
    if best == none:            stop_reason = no-signal;        break
    if evidence[best] <= 0:
        if !fill_priors:        stop_reason = evidence-exhausted; break
    select(best)
stop_reason = (selected_tokens >= target) ? budget : <上一个原因>
```

- 关键点：证据耗尽时**先判定**再选，不允许"先选一个零证据块再判断"，否则
  永远填满预算、`fill_priors=0` 形同虚设。
- `fill_priors=1` 时零证据块仍可入选（覆盖采样），但排在所有有证据块之后；
- 预算裁剪（`limit = ceil(target·1.05)`）仍保留，只裁非锚点、按 fused 升序。

### 3.4 审计（FR-3）

| 字段 | 含义 |
|---|---|
| `Selection.evidence_blocks` / CSV `evidence_blocks` | 入选块中 `evidence>0` 的数量 |
| `Selection.stop_reason` / CSV `stop_reason` | `budget` / `evidence-exhausted` / `no-signal` / `dense` |
| `Selection.anchor_blocks`（已有） | 锚点数量，用于判断"预算花在锚点上" |

### 3.5 组合再择优（FR-4）

复用 `portfolio_matrix.py`，组合集改为：

| combo | 语义 |
|---|---|
| `selection` | 预算曲线 + 证据优先准入（`fill_priors=1`） |
| `selection-fit` | 同上但 `fill_priors=0`（内容决定预算，端侧省算） |
| `gate-p16` / `gate-p24-local32` | AR006 图内门控（对照） |
| `dense` | 参照 |

规则不变：**达到质量地板（mean 与 min 均 ≥0.95）的最快组合，否则质量最高者**。
表由脚本生成，不手填；闸门用同会话最佳固定组合做 Pareto 校验，注入劣化必须 FAIL。

### 3.6 CPU + GPU 双后端（FR-5）

- CUDA 构建：`cmake -DGGML_CUDA=ON`，本机 Quadro RTX 5000（16GB，CUDA 12.0/驱动 556.18）；
- 一致性用例：
  1. `qg_bound=on` 下 4K/8K/16K 的 `q` 与 CPU 相同、`qg_source=attention-bound`；
  2. `budget=1.0` dense 路径贪心解码与 CPU **逐 token 相同**（温度 0）；
  3. 选择级稀疏的 `selected_tokens`/`block_indices` 与设备无关（选择阶段纯 CPU）；
- 记录：GPU 型号、驱动、CUDA、`n_gpu_layers`、线程、命令 → REPORT + server-repro。

## 4. 复杂度与资源

| 阶段 | 复杂度 | 备注 |
|---|---|---|
| minmax | O(n_blocks) | 不变 |
| evidence/prior | O(n_blocks) | 常数项 |
| 填充 | O(n_blocks × picks) | 与 AR009 相同 |
| 额外内存 | 2 × n_blocks × double | 32K 时约 16 KB |

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| 先验门控过严，长文需要位置覆盖时质量掉 | `fill_priors=1` 保留；`kPriorSat` 可调；组合层用实测表兜底 |
| 修 minmax 改变所有历史基线 | 以 AR009 实测值为基准重新测量，不改历史报告；差异写入 REPORT |
| `fill_priors=0` 使 selected_tokens 变小，可能改变 TTFT 口径 | 审计列给出 `stop_reason`，报告中同时给"预算上限"和"实际选中" |
| GPU 结果与 CPU 不一致 | 一致性用例进 ST；不一致则只声明 CPU 结论，GPU 记录为待办 |

## 6. 与既有 AR 的关系（组合视角）

| AR | 在 AR010 中的角色 | 冲突与处置 |
|---|---|---|
| AR007 | 预算曲线/锚点/MMR/span-grow 全部保留 | 停止条件与"必须填满预算"的假设冲突 → 用 `fill_priors` 显式选择 |
| AR008 | 短语信号进入 `evidence`；组合表机制复用 | 无冲突；短语从"加在 fused 里"改为"属于证据组" |
| AR009 | 覆盖增益保留在填充项 | 与 `evidence>0` 判定一致（覆盖锚点本身有证据） |
| AR004/AR006 | 不改 patch；门控作为组合臂参与择优 | 32K 若选择级达标，门控在该数据集上被表淘汰 |
| AR005 | 语义分数（draft）属于 `evidence` | 无冲突 |

## 7. 测试策略

| 层次 | 内容 |
|---|---|
| 单测 | `minmax` 三位置；`prior_gate` 语义（零证据块分数为 0）；`fill_priors` 两种停止；单事实不回归 |
| 真机 CPU | 7 数据集 × {prior_gate, fill_priors} 四象限；逐例质量与 selected_tokens |
| 真机 GPU | CUDA 构建；dense 逐 token 一致；门控等价；同一配置质量一致 |
| 组合 | 组合矩阵重测 → 生成新表 → Pareto 闸门 + 注入劣化 FAIL |
| 回归 | `-Werror` 零警告；无模型全量；干净克隆 |

## 8. 实测结果（as-built）

（开发完成后填写：见 §10。）
