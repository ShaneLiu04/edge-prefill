# [AR012] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| AR 主题 | 句子完整准入（sentence-complete admission）+ 可解释性审计 |
| 关联 SR | 无（沿用 AR001 端侧 prefill 目标） |
| 日期 | 2026-10-28 |
| 状态 | Draft |

## 1. 背景与目标

AR011 之后单事实族已到 `q=1.000`、4K 146 token / 755 ms（dense 40 倍），
但多事实族仍是全系统最弱的一环：easy 0.836、hard **0.709（`q_min` 0.571）**。

本轮先做**预算扫描**（排除"预算不足"假设，真实 GGUF、16 例）：

| 数据集 | b=0.05 | b=0.07 | b=0.09 | b=0.12 | b=0.16 |
|---|---|---|---|---|---|
| multifact8k q | **0.836** | 0.830 | 0.830 | 0.790 | 0.771 |
| multifact8k token | 431 | 505 | 513 | 563 | 737 |

→ **加预算是负收益**：多准入的块是填充文本，只会稀释证据。所以多事实的瓶颈不是
预算，而是**准入的精度**。

再用新加的 `--dump-selected-text`（本 AR 交付）把模型真正看到的上下文打出来，
直接看到根因：

```
[pos 3904] ... The secret passcode for vault six is TEAL-LYNX-8
[... 672 dropped tokens]
[pos 4608] . The secret passcode for vault seven is SILVER-PANTHER-8782. ...
```

**口令被块边界切断**（`TEAL-LYNX-8` | `853`）。32-token 块的粒度与句子不对齐，
一个被问事实的"键 + 值"跨两个块，而填充是整块 all-or-nothing：
选中前半块就等于只准入半个答案。hard 档 6 个事实里常有 2 个被切断，
正好对应 0.709 ≈ 4.25/6。

**目标**：让准入的原子单位是"句子"而不是"块"——被准入的句子必须完整，
不足的尾巴要补齐（必要时跨块），同时不引入填充文本；并把"模型看到什么"变成
可审计的第一类能力。

## 2. 需求范围

**In Scope：**
- `src/selector.cpp` 的 span 语义重构：块内证据 → 句子（可跨块）→ 完整的 token 集合
- `Selection` 审计：`sentences_completed`、`span_tokens`（沿用）、`dump_selected_text` CLI
- 单测、`ar012_matrix.py` / `ar012_report.py`、`repro.sh ar012`
- 文档：README、`docs/research.md` §14、`docs/architecture.md`、`docs/server-repro.md`、组件详设

**Out of Scope：**
- 跨请求 KV 复用（CacheBlend/EPIC 类，另立 AR）
- 解码阶段优化、批处理调度
- 任何 mock、任何"改口径"式提分（例如把 48 token 答案长度改成别的）

## 3. 功能需求

### 3.1 FR-1 句子完整准入（sentence completion）

**描述：** 被准入的证据必须构成完整句子；句子跨块时一并准入缺失部分。

**触发条件：** `sentence_completion=1`（默认 1；`0` 时逐 token 等价 AR011/AR010）。

**期望行为：**
- 以 token piece 的句末标点（`. ! ? \n ;`）为界，建立"token → 所属句子 [s,e)"映射；
- 对每个入选证据块，块内 `per_pos_scores > 0` 的 token 所在句子整体保留；
- 句子伸进未入选的块时，把该块加入选择集（闭包一次即可，句子很短），
  然后再按句子裁剪其它 token；
- 结构锚点（sink/recent）仍整块保留（chat 模板与首 token logits）；
- 无正分 token 的块整块保留（不猜）。

**异常处理：** 句子长度超过 `span_max_tokens`（默认 128）时按块边界截断并
`log_warn`；`per_pos_scores` 长度不符时退回整块（不裁剪）。

**验收标准：**
- Given 多事实 hard 的一句口令跨两块，When 选择，Then 两块的对应 token 都在
  `token_indices` 中（单测构造该场景）；
- Given 6 事实 hard 数据、b=0.05，Then `q_mean` ≥ AR011 且 `q_min` ≥ 0.571；
- Given `sentence_completion=0`，Then 与 AR011 逐 token 一致。

### 3.2 FR-2 可解释性：`--dump-selected-text`

**描述：** 把模型真正看到的上下文写成人类可读文件。

**期望行为：** 输出 `# prompt_tokens / selected_tokens` 头，token 原文，遇到被丢弃的
token 段时插入 `[... N dropped tokens]` 与原始位置标记；不动任何计算结果。

**验收标准：** Given 任意一次 `run`，Then 文件里出现的 token 序列与 `token_indices`
一一对应（单测按位置校验）。

### 3.3 FR-3 无回归组合与闸门

**描述：** 必须在 7 个数据集上不劣于 AR011 才允许进入组合。

**期望行为：** `ar012_matrix.py`（AR011 基线 / AR012 / 关掉 completion 的对照 /
gate-p16）+ `ar012_report.py`：沿用 AR011 的"质量不降、token 不增、TTFT 不劣"闸门，
并对多事实族额外要求 `q_min` 不降。

**验收标准：** 7/7 PASS；注入劣化 FAIL。

### 3.4 FR-4 审计

新增审计列：`sentences_completed`（被补齐到句子边界的次数）、`sentences_extended`
（因跨块而追加的块数）；raw.csv 与 `--dump-selection` 同步。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|---|---|---|
| 质量 | 多事实 hard `q_mean` | ≥0.709（目标 ≥0.78），`q_min` ≥0.571 |
| 质量 | 单事实 4K/8K/16K/32K/多针 | 1.000 不降 |
| token | 各数据集 | 不超过 AR011 的 1.10×（为补句子允许少量增加） |
| 性能 | 选择阶段 | 句子映射 O(n) 一次，单次选择开销 <2 ms（31K） |
| 兼容 | `sentence_completion=0` | 与 AR011 逐 token 一致 |
| 兼容 | `budget=1.0` | 与 dense 一致 |

## 5. 约束与假设

- 无 mock；全部质量来自真实 GGUF 前向
- 允许为"证据完整"多花少量 token（≤10%），但必须在闸门里显式记录
- 多事实口径（48 新 token、部分分）不得改变，否则前后不可比

## 6. 术语

| 术语 | 定义 |
|---|---|
| 句子 | 由句末标点界定的 token 连续区间，可跨 32-token 块 |
| 句子完整准入 | 证据 token 所在句子整体准入，跨块时补齐 |
| 句子闭包 | 因补齐而把新块加入选择集的过程 |
