# AR012 验收报告：句子完整准入（sentence-complete admission）

| 项 | 值 |
|----|----|
| AR | AR012 |
| 模型 | qwen2.5-1.5b-instruct-q4_k_m（SHA256 `6a1a2eb6…e9407e`） |
| CPU | Xeon 6234 / 16 线程（WSL2）/ llama.cpp `3bb63cb` |
| GPU | Quadro RTX 5000 16 GB / CUDA 12.0（`build-cuda`） |
| 命令 | `python3 benchmarks/scripts/ar012_matrix.py --dump` + `ar012_report.py --gate` + `check_facts_admission.py` |
| 原始数据 | 本目录 `<dataset>-<arm>/{bench.conf,*_raw.csv}`、`matrix.csv`、`portfolio.json`、`facts_admission.csv`（确定性事实完整率）；逐 case 准入文本用 `ar012_matrix.py --dump` 再生，不入库（约 8 MB 文本） |
| 结论 | 多事实 easy **0.836 → 0.887**、hard **0.709 → 0.739（`q_min` 0.571 → 0.621）**，其余 5 个数据集逐 token 不变；代价是多事实 +15~18% token |

## 1. 起点：先排除"预算不足"的假设

AR011 之后系统最弱的一环是多事实：easy 0.836、hard 0.709（`q_min` 0.571）。
第一反应是"预算给少了"，于是做了一轮**预算扫描**（真实 GGUF，16 例/档）：

| 数据集 | b=0.05 | b=0.07 | b=0.09 | b=0.12 | b=0.16 |
|---|---|---|---|---|---|
| multifact8k q | **0.836** | 0.830 | 0.830 | 0.790 | 0.771 |
| multifact8k token | 431 | 505 | 513 | 563 | 737 |

**加预算是负收益**（多准入的是填充文本）。假设被数据否掉，转向"准入精度"。

## 2. 定位根因：`--dump-selected-text`

新增的 `--dump-selected-text`（以及本次加入 bench 的 `dump_selection_dir`）
把模型真正看到的上下文打出来，直接看到：

```
[pos 3904] ... The secret passcode for vault six is TEAL-LYNX-8
[... 672 dropped tokens]
[pos 4608] . The secret passcode for vault seven is SILVER-PANTHER-8782. ...
```

**口令被 32-token 块边界切断**：`TEAL-LYNX-8` | `853`。块级准入是 all-or-nothing，
选中前半块 = 只准入了半个答案。

为了摆脱模型打分的噪声，本 AR 增加了一个**确定性指标**：
`check_facts_admission.py` 直接统计"被问事实的答案字符串是否完整出现在准入上下文里"，
不调用模型打分。

| 臂 | 事实完整率（hard） | 全部事实都在的 case | 事实完整率（easy） |
|---|---|---|---|
| AR011 | 0.760 | 2/16 | 0.875 |
| gate-p16 | 0.771 | 3/16 | 0.875 |
| **AR012** | **0.990** | **15/16** | **1.000** |
| AR012 + 裁剪 | 0.990 | 15/16 | 1.000 |

## 3. 做了什么

- **FR-1 句子完整准入**：证据 token 拉入其所在**整句**（句末标点为界，可跨块）；
  句子伸进未入选的块时把该块加入选择集（闭包）；随后只在块内裁剪句子之外的 token。
- **只在多事实生效**（`coverage` 子查询 ≥2）：单事实实测没有质量收益，
  却让多针 247 → 283 token（+15%），所以不介入。
- **两种模式**：`sentence_trim=0`（默认，并集：只补不删）、`sentence_trim=1`（再裁剪）。
- **FR-2 可解释性**：`--dump-selected-text`（单次运行）与 `dump_selection_dir`（bench 每个 case
  写 `_selected.txt` / `_selection.json`），配合 `check_facts_admission.py`。
- **FR-4 审计**：`sentences_completed`（补齐的句子数）、`sentences_extended`（闭包新增块数）。
- **修复一个真 bug**：超限安全网原来按"mandatory 优先"截断，锚点集大于上限时会把证据整段丢掉
  （实测 hard 0.709 → 0.299）；改为优先保护**证据块 / 结构块 / 句子 token**。

## 4. 结果（同会话，7 数据集 × 4 臂）

| 数据集 | 臂 | q_mean | q_min | 准入 token | TTFT |
|---|---|---:|---:|---:|---:|
| multifact8k | AR011 | 0.836 | 0.429 | 431 | 2340 ms |
| | **AR012** | **0.887** | 0.429 | 509 | 2777 ms |
| | AR012+trim | 0.815 | 0.400 | 371 | 2037 ms |
| multifact8k-hard | AR011 | 0.709 | 0.571 | 445 | 2399 ms |
| | **AR012** | **0.739** | **0.621** | 513 | 2723 ms |
| | AR012+trim | 0.689 | 0.353 | 372 | 2140 ms |
| len4k / len8k / len16k / len32k / 多针 | AR011 ≡ AR012 | 1.000 | 1.000 | 146 / 188 / 280 / 525 / 247 | 不变 |

组合表（`ar012_report.py`）：多事实两档路由到 `ar012`，其余路由到 `ar011`，
**"没有免费收益"闸门 7/7 PASS**（质量不降；若无质量收益则 token/TTFT 不得增长，
若有收益最多允许 1.25×）。注入 `deep/multifact8k:ar012-trim` → FAIL（q/q_min 双违规）。

## 5. GPU（CUDA）一致性

```
cpu  q=0.739 qmin=0.621 sel=[500,518,545,512,500,484,441,488,484,578,519,551,521,563,510,506] ttft=2762.0ms
gpu  q=0.739 qmin=0.621 sel=[...完全相同...]                                                  ttft= 815.9ms
selection identical on both backends: True
```

## 6. 回退与负结果

| 尝试 | 实测 | 处置 |
|---|---|---|
| 加预算（b 0.07–0.16） | 0.836 → 0.830/0.830/0.790/**0.771** | 预算无解，转精度 |
| 句子补齐 + **裁剪**（`sentence_trim=1`） | 4K 1.000→**0.827**、32K 1.000→**0.827**、hard 0.739→0.689 | 不裁剪（默认 0），模式保留供研究 |
| 补齐用于单事实 | 多针 247→283 token，q 不变 | 门控为"多事实才启用" |
| benchmark 结果目录复用 | `--force` 未清空旧 `*_raw.csv`，跨配置均值混合 | 已修（脚本里先删旧文件） |

## 7. 限制

- 多事实 hard 仍只有 0.739：事实**全部在上下文**（0.990）后，剩余损失来自模型本身
  （6 条答案的复制/排序），不是准入层；
- 多事实 token 增加 15–18%（431→509、445→513），是本 AR 明确接受的代价（换取完整答案）；
- 32K/多针的 dense 参照沿用 AR010/AR011 口径，未在本会话重测。

## 8. 复现

```bash
./repro.sh ar012     # 4 臂矩阵（含 dump）+ 表 + 闸门 + 注入自测 + 事实完整率
```
