# AR 全景与最佳组合

这份文档把 AR001–AR012 拆成三类问题：**哪些进了最终组合**、**哪些保留为可选开关**、
**哪些被判为负结果并关掉**；最后给出"合到一起"的配置、逐数据集实测和复现命令。

所有数字来自真实 GGUF 推理（Qwen2.5-1.5B-Instruct Q4_K_M，16 线程，Xeon Gold 6234），
原始 CSV 在各 AR 的 `benchmarks/results/` 目录。不同 AR 的会话不同，**只比较同一会话内的比值**。

## 1. 清单

| AR | 做了什么 | 结论 | 最终状态 |
| --- | --- | --- | --- |
| AR001 | 架构骨架：`plan`/`bench`/CLI、`Status`+`ErrorCode`、审计字段 | 夯实地基，无指标 | 基础（默认） |
| AR002 | Quest 式块级选择 + KV gather + 端到端复现 | 稀疏 prefill 主线成立 | 基础（默认） |
| AR003 | SnapKV 式 per-head top-k gather（离线 head profile 分类） | 头级稀疏收益不稳定，`classify-heads` 有噪声 | 可选开关（默认 off） |
| AR004 | chunk 对齐 `n_ubatch`、MInference 式 head 模式路由、pyramid 先验 | 只有 chunk 对齐稳定收益 | chunk 对齐默认开；head 路由/pyramid 默认 off |
| AR005 | 语义页门控（bge-small 打分 + 图内门控） | 打分扫描 +1.0–2.2 s，质量未增 | 可选（默认无 embedding） |
| AR006 | 图内 query-aware 页门控（`qg_bound`） | 8K 9.4×、16K 6.8×、32K 5.7× | 可选；32K 已被 AR007 更省的路径取代 |
| AR007 | 预算曲线（长度→预算）、结构锚点、MMR 去重页、长度自适应调度 | **核心收益**：预算 0.05 时 8K 29× 且 q=1.000 | 默认 |
| AR008 | 位置短语准入（`phrase_weight`）+ 组合路由表（portfolio） | 短语准入在长提示上补召回 | `phrase_weight=0.5` 默认 |
| AR009 | 多事实覆盖准入（facility-location + 每事实锚点，`coverage_weight`） | 多事实从"有一半"到"覆盖到" | `coverage_weight=3.0` 默认 |
| AR010 | 证据优先准入：`prior_gate=1`（先验不能创造准入）、`fill_priors=0`（内容长度上下文） | 修复预算最小值 bug；32K 不再需要页门控 | 默认 |
| AR011 | 结构预算 `structure_budget`（sink/recent 绝对预算）+ 单事实 span 准入 `span_admission` | 4K 226→146 token、8K 228→188、多针 321→247；多事实 hard 0.667→0.709 | `structure_budget=64`、`span_admission=1` 默认 |
| AR012 | 句子完整准入 `sentence_completion`（证据按整句拉入，跨块闭包） | 多事实 easy 0.836→0.887、hard 0.709→0.739 | 仅多事实启用（默认开） |
| AR013 | 紧凑上下文：KV 按准入量分配 + 块级位置压缩 `position_compaction` | KV 896 MiB→7 MiB(4K)/21 MiB(32K)；多事实 0.887→0.975、hard 0.739→0.746；64K 文档从不可运行变为可运行 | 默认开（库默认 off） |

注意"默认"指**部署配置**（`ar012` 臂）。库内代码默认值保持保守（`structure_budget=0`、
`span_admission=0`、`sentence_completion=0`、`fill_priors=1`），必须由 conf 显式打开；
`budget=1.0` 且全部开关关闭时与 dense 数值等价（有单测保证）。

被否决（负结果，均已关闭，数据集与报告保留）：

| 方向 | 实测 | 位置 |
| --- | --- | --- |
| token 级 masked FFN（`skip_ffn` 精细化） | 质量换时间不划算 | `benchmarks/results/ar006-campaign/REPORT.md` |
| 多事实加预算（b 0.05→0.16/0.25） | 0.771 / 0.815 < 0.836，多出的 token 是填充 | research §14.1 |
| 句子补齐后裁剪残留填充（`sentence_trim=1`） | 4K/32K 1.000→0.827、hard 0.739→0.689 | research §14.6 |
| embedding 语义准入作为主信号 | 5 数据集质量差 ≤0.003，扫描 +1.0–2.2 s | research §13.5 |
| 硬负例抑制 | 零质量收益 + 额外开销 | research §12.8 |
| 单事实也做句子补齐 | q 不变、token +15%（多针 247→283） | research §14.6 |
| 只把 n_ctx 调小、不做位置压缩 | 4K 仍 28 MiB，跨度仍由文档长度决定 | research §15.5 |

## 2. 最佳组合（默认部署配置）

"最佳组合"不是把所有开关打开，而是**按层叠顺序只留仍在贡献的机制**：

```
mode = balanced                 # AR001/AR007 规划档位
attn_top_k = 24                 # AR002 块内行级 gather
budget_curve = <长度→预算表>     # AR007
anchor_ratio = 0.35             # AR007（锚点主导预算时）
phrase_weight = 0.5             # AR008
coverage_weight = 3.0           # AR009（多事实覆盖）
prior_gate = 1                  # AR010
fill_priors = 0                 # AR010 内容长度上下文
structure_budget = 64           # AR011 sink+recent 绝对预算
span_admission = 1              # AR011 单事实按句窄化
span_min_tokens = 24
sentence_completion = 1         # AR012 仅 sub_idf>=2（多事实）生效
sentence_trim = 0               # 裁剪不路由
span_max_tokens = 128
position_compaction = 1         # AR013 块级位置压缩（KV 跨度 = 准入量）
```

关掉的：`head_adaptive`/`head_pattern_routing`/`pyramid_prior`（AR003/AR004）、
`embedding_model`（AR005）、`qg_bound`（AR006，32K 已被 AR007 覆盖）、`skip_ffn`（负结果）。
`mmr_lambda=0.5`/`mmr_threshold=0.6` 保持库默认，但只作为去重兜底，不是主信号。

多事实族需要平铺小预算：`budget = 0.05`、`max_new_tokens = 48`（其余数据集用曲线 + 128）。

这个组合就是 `benchmarks/results/ar012-sentence-complete/portfolio.txt` 里的路由表：
`default = ar012`，5 个非多事实数据集 `= ar011`（AR012 在这些数据上逐 token 相同，
因为句子补齐只在多事实触发），多事实两个数据集 `= ar012`。

## 3. 组合后的逐数据集实测

同一会话（AR011 矩阵 + AR012 矩阵），dense 只作参照；多事实是 b=0.05 的稀疏档，不与 dense 比绝对 TTFT。

| 数据集 | 例数 | dense TTFT | 组合 TTFT | 加速 | 准入 token | q_mean | q_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explen/len4k | 4 | 30.5 s | **0.75 s** | 40× | 146 | 1.000 | 1.000 |
| explen/len8k | 4 | 69.7 s | **1.03 s** | 68× | 188 | 1.000 | 1.000 |
| explen/len16k | 4 | 172.6 s | **1.39 s** | 124× | 280 | 1.000 | 1.000 |
| deep/len32k | 4 | 447 s¹ | **2.33 s** | 192×¹ | 525 | 1.000 | 1.000 |
| deep/multineedle | 6 | 69.1 s | **1.16 s** | 59× | 247 | 1.000 | 1.000 |
| deep/multifact8k | 16 | — | 2.84 s | — | 509 | **0.975** | **0.600** |
| deep/multifact8k-hard | 16 | — | 2.63 s | — | 513 | **0.746** | **0.643** |
| explen/len64k | 4 | 不可运行 | 6.44 s | — | 1163 | 1.000 | 1.000 |

¹ 32K dense 是 AR010 口径的 2 例子集（447 s），AR011 矩阵没有重测 dense，所以 192× 是跨口径比值，读作"百倍量级"。

组合相对上一 AR（AR011）的全部变化：

| 指标 | AR011 | AR012 组合 | 变化 |
| --- | ---: | ---: | --- |
| 多事实 easy q_mean / q_min | 0.836 / 0.429 | **0.975** / **0.600** | +0.139 / +0.171（AR012 +0.051，AR013 再 +0.088） |
| 多事实 hard q_mean / q_min | 0.709 / 0.571 | **0.746** / **0.643** | +0.037 / +0.072 |
| 事实完整率（hard / easy） | 0.760 / 0.875 | **0.990 / 1.000** | 确定性审计（AR012 起） |
| 多事实准入 token | 431 / 445 | 509 / 513 | +18% / +15%（AR012 的完整答案代价，AR013 未再增加） |
| 其余 5 数据集 | — | 逐 token 不变 | 0 |
| KV 分配（4K / 32K） | 112 / 854 MiB（旧 bench 均 896 MiB） | **7 / 21 MiB** | AR013 位置压缩 |
| 64K 文档 | 不可运行 | **q=1.000** | AR013 解锁 |

即：**组合在所有已测数据集上没有质量回退**；AR012 用 ~70 token 换完整答案，
AR013 再把这部分 token 的 KV 成本压到 1/11，并把"事实在上下文里却答不全"的残差
（距离问题）变成质量收益。

## 4. 复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEDGE_PREFILL_LLAMA_DIR=<pinned llama.cpp>
cmake --build build -j
EDGE_PREFILL_TEST_MODEL=/path/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  ctest --test-dir build --output-on-failure      # 167 cases, no mocks

./repro.sh ar012     # 4 臂矩阵 + 逐 case dump + 闸门 + 注入 + 确定性事实审计
./repro.sh ar011     # 上一基线（含 dense 参照与加速比）
```

干净克隆（`git clone --branch ar012-v1.0`）实测：构建通过、167 用例全通过、
`ar012_report.py --gate` 7/7 PASS、`check_facts_admission.py --gate-arm ar012` 0.990、
多事实 q_mean 复现到 0.887 / 0.739 完全一致。

## 5. 下一步增量

- 多事实族仍是唯一低于 0.9 的档位：q_min 0.621 意味着还有单例整题错，事实完整率 0.990 说明
  残留误差已不在准入层，需转向答案生成侧（提示词/解码）验证。
- 32K 的 dense 参照需要同会话重测，才能把 192× 变成严格同口径数字。
- 句子补齐的 +70 token 可尝试"只补答案所在句 + 相邻句首 token"以压低 TTFT。
