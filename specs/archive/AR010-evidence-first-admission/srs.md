# [AR010] 需求规格说明书：证据优先准入（Evidence-First Admission）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 主题 | evidence-first-admission |
| 日期 | 2026-10-27 |
| 关联组件详设 | `specs/component-detail-design/组件_spec.md`（v0.7） |
| 前序 | AR001–AR009（均已归档） |
| 状态 | 需求已定稿（用户授权全自动流程，自检替代交互确认） |

## 1. 背景：一次 32K 预算审计暴露的信号缺陷

AR009 收口时，32K 选择级只有 q=0.875，因此 AR008 的组合表在 32K 上必须切到
图内门控（36.8× → 12.8×）才能拿回质量。本轮先做**预算去向审计**（不是猜），
在 `len32k`（31,051 token，budget=0.05 → 1553 token ≈ 50 块）上导出选择结果：

```
选中 50 块，其中 30 块是文档尾部连续块 941-970（占 62% 预算）
块 950：lexical=0.0000  phrase=0.0000  position=1.0654  diversity=2.4081  fused=0.7179
块 147：lexical=0.3216  phrase=0.3333  position=0.4745  diversity=2.5240  fused=0.9394
```

**62% 的预填预算花在"与问题零词法/零短语重合"的块上**，它们靠 position/diversity
两项先验挤进选择。继续追查两项先验，发现它们根本不在 [0,1]：

```
[sig] lex=[0.0000,1.0000] div=[0.0000,3.2000] pos=[0.0000,1.1569]
```

### 1.1 根因（已用探针证实，不是推测）

```cpp
void minmax(std::vector<double>* values) {
    const auto [lo, hi] = std::minmax_element(values->begin(), values->end());
    const double range = *hi - *lo;
    for (double& v : *values) v = (v - *lo) / range;   // *lo 是迭代器！
}
```

`lo` 是**指向正在被写入的向量**的迭代器。最小值元素被归一化成 0 之后，
后续元素读到的 `*lo` 变成 0，退化为 `v / range`：

| 信号 | 归一化前 | 期望 | 实测 |
|---|---|---|---|
| lexical | [0, 7.17] | [0,1] | [0,1]（min=0，恰好正确） |
| diversity | [0.4549, 0.6616] | [0,1] | [0, **3.2**] |
| position | [0.1356, 1.0] | [0,1] | [0, **1.1569**] |

因此 `diversity` 的实际权重是配置值的 3.2 倍，`position` 是 1.16 倍；而这两项
**先验**（"块内信息量""离文档头尾近"）是可以对**零证据块**给出高分的。长上下文下
先验覆盖的块数线性增长，于是预算被系统性抽走到填充段落。

### 1.2 更深一层：填充循环没有停止语义

把 `minmax` 修正后（`*lo/*hi` 先拷贝取值），归一化正确，但立刻暴露第二个缺陷：

| 数据集 | 修复前 q | 修复后 q | selected |
|---|---|---|---|
| 4K | 1.000 | **0.500** | 322 → 210 |
| 8K / 16K / mn | 1.000 | 1.000 | 不变 |
| 32K | 0.875 | 0.917 | 1566 → 1566 |
| multifact8k | 0.878 | **0.579** | 443 |
| multifact8k-hard | 0.652 | **0.338** | 445 |

原因：填充循环的停止条件是

```cpp
const double score = adjusted[b] + (adjacent[b] ? 0.5 : 0.0);
if (best == n_blocks || best_score <= 0.0) break;
```

归一化畸形时，diversity/position 恒大于 0，`best_score` 几乎不可能为 0，预算总能填满；
归一化正确后出现大量精确 0，填充在预算远未用完时**提前退出**。也就是说：
**"填满预算"这件事一直依赖一个数值 bug，而不是一个策略。** 这正是长上下文质量
不稳定、且必须靠门控兜底的深层原因。

## 2. 目标

1. **预算跟着证据走**：先验（位置/多样性）只能**排序有证据的块**，不能**创造**证据；
   零证据块只能在"预算本来就有余"时作为结构填充出现，且必须可关闭；
2. **填充有明确停止语义**：证据耗尽 / 预算用尽 / 用户禁止结构填充，三种原因可审计；
3. **组合整体变强**：32K 选择级在**不启用图内门控**的前提下达到 q≥0.99，
   从而让 AR008 组合表把 32K 从"12.8× 门控"换回"~36× 选择级"；
   同时 4K/8K/16K/多针保持 1.000，多事实族不低于 AR009 的 0.878/0.652；
4. **本地 CPU + GPU 双验证**（本机 Quadro RTX 5000 16GB）：同一语义在 CPU 与
   CUDA 后端结果一致，为服务器复现提供依据；
5. **诚实可回退**：任何一项达不到就回退，只保留数据集/负结果记录。

## 3. 功能需求

### 3.1 FR-1 归一化正确性与信号解耦

1. `minmax` 必须先把上下界拷贝成值再改写向量（修别名 bug）；
2. 块分数拆成两组语义明确的量：
   - `evidence = w_lex·lexical + w_phrase·phrase + w_draft·draft`（**可准入性**）
   - `prior    = w_div·diversity + w_pos·position`（**排序先验**）
3. `fused = evidence + gate(evidence)·prior`，其中
   `gate(e) = min(1, e / (kPriorSat · max_evidence))`（`kPriorSat = 0.25`）：
   证据越弱先验越弱，零证据块 priori 加权为 0；
4. `prior_gate=0` 时退回 `fused = evidence + prior`（先验可独立准入），用于消融。

### 3.2 FR-2 填充策略：证据优先 + 显式停止

填充循环按 `fused + span + coverage` 取最优，但**停止条件显式化**：

| 条件 | 审计值 `stop_reason` |
|---|---|
| 达到预算 | `budget` |
| 剩余块 `evidence == 0` 且 `fill_priors=0` | `evidence-exhausted` |
| 剩余块全为 0 分（含先验） | `no-signal` |
| 选择被裁到 dense | `dense` |

- `fill_priors=1`（默认）：证据耗尽后允许用先验把预算填满（覆盖式采样）；
- `fill_priors=0`：证据耗尽即停，**selected_tokens 由内容决定**而不是由长度比例决定，
  这是端侧最实际的省算路径；
- 两种模式都必须保证：sink、末块、短语锚点、覆盖锚点不被裁掉。

### 3.3 FR-3 审计与可复现

1. 新增审计列（raw.csv 末尾追加）：`evidence_blocks`（`evidence>0` 且入选的块数）、
   `stop_reason`；
2. `Selection` 增字段 `evidence_blocks` / `stop_reason`；
3. 配置键：`prior_gate`（0/1）、`fill_priors`（0/1），CLI 同名开关；
4. 单测：`minmax` 别名回归（构造 min 在中间/开头的向量）、零证据块不得准入、
   `fill_priors` 两种模式的停止语义、单事实族不回归。

### 3.4 FR-4 组合再择优（核心交付）

按 AR008 的规则**重新测量**组合矩阵（dense / selection / selection+cov /
gate-p16 / gate-p24-local32 × 4K/8K/16K/32K/多针/多事实），由测量结果重新生成
`portfolio.txt`，并跑 Pareto 闸门：

- 若 32K 选择级达标 → 表从 gate 切回 selection，**32K 净加速 12.8× → ≥30×**；
- 若某组合变差 → 该组合从表中删除，回到上一版最优表（回退）。

### 3.5 FR-5 本地双后端验证（CPU + GPU）

1. CUDA 构建（本机 RTX 5000 16GB）跑同一组配置，验证：
   - 稀疏/密集在 GPU 上语义一致（同一 prompt 贪心解码结果相同或 F1 相同）；
   - AR006 图内门控在 GPU 上的页打分与 CPU 等价（补齐 AR007 T009 遗留项）；
2. 记录 GPU 型号/驱动/CUDA 版本与命令，写入 `benchmarks/results/ar010-*/REPORT.md`
   与 `docs/server-repro.md`；
3. 结果为**同会话同机对比**，绝对延迟只作参考。

## 4. 非功能需求

| 项 | 要求 |
|---|---|
| 无 mock | 所有数字来自真实 GGUF 前向（CPU 与 CUDA 后端） |
| 兼容 | 新审计列追加在末尾；`prior_gate`/`fill_priors` 缺省行为由实测决定 |
| 性能 | 不引入额外前向；选择阶段复杂度不变 |
| 可复现 | 数据集/脚本/配置/模型 SHA256/线程/后端全部落盘 |
| 可靠 | 归一化、填充、裁剪三处都有单测；`-Werror` 零警告 |

## 5. 明确不做

| 方向 | 理由 |
|---|---|
| 重新设计先验（学习型位置先验） | 需要训练，样本不足，且本轮已能定位问题 |
| token 级选择 | 破坏 block/ubatch/prefix-cache 对齐，风险大收益不明 |
| KV 逐块淘汰 | 需要 patch 且与 AR006 互斥风险高，留作后继 |
| 端侧 INT4 KV | 已有 kv_type，但与本 AR 目标无关 |

## 6. 验收标准（硬指标）

| 指标 | 目标 |
|---|---|
| 4K / 8K / 16K / 多针 q_mean | = 1.000（不得回归） |
| 32K 选择级（无门控）q_mean | ≥ 0.99 |
| multifact8k q_mean | ≥ 0.88（AR009 = 0.878） |
| multifact8k-hard q_mean | ≥ 0.65（AR009 = 0.652） |
| 组合表 | 重新生成并通过 Pareto 闸门；32K 净加速不低于 AR008（≥12.8×） |
| 审计 | `stop_reason` 与 `evidence_blocks` 在 raw.csv 中可解释每次选择 |
| 测试 | 无模型全量 PASS；CPU 真机 PASS；CUDA 后端一致性 PASS |
