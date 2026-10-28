# 服务器复现指南（AR006 图内查询感知门控 / AR007 自适应预算）

目标：在一台你没见过的机器上，用**真实模型**复现 AR006 的净 TTFT 加速与质量，
并判断你的任务是否值得开启图内页门控。所有命令都是可复制粘贴的，且不依赖本仓库
机器上的任何路径（`/tmp/one` 之类只出现在开发记录里，不在复现路径上）。

> 诚实红线：绝对延迟强依赖机器、会话、内存带宽。**只比同一台机器同一会话的 dense 与
> bound 比值**；跨机器/跨会话的绝对毫秒数没有可比性（本仓库自己的 AR004 会话与
> AR006 会话就差 2.7×）。每次运行都会写出 `*_env.json`，请在报告里带上它。

## 1. 依赖与获取

```bash
git clone https://gitee.com/liu-xingyan04/edge-prefill.git
cd edge-prefill
sudo apt-get install -y build-essential cmake python3 curl
```

模型（Q4_K_M，约 1 GB；SHA256 见 `repro.sh` 内的 `MAIN_SHA`）：

```bash
./repro.sh smoke --skip-download   # 已有模型时
./repro.sh smoke                   # 没有模型时自动下载并校验
```

`repro.sh` 会构建（llama.cpp pinned `3bb63cb`，通过 `patches/` 应用两个补丁）、
跑单测、跑真实模型冒烟。**第一步能通过，再继续下面的门控复现。**

## 2. 数据

数据集随仓库分发（`benchmarks/data/`，含 `manifest.json` 与逐文件 SHA256）：

| 目录 | 用例数 | 长度 | 用途 |
|---|---|---|---|
| `explen/len4k` | 4 | ~4K | 短提示：门控退化为 local-only 的对照 |
| `explen/len8k` | 4 | ~8K | 过渡区 |
| `explen/len16k` | 4 | ~15.5K | 主验收（niah retrieval） |
| `deep/len32k` | 4 | ~31K | 深上下文 |
| `deep/multineedle` | 12 | 4K–32K | 多针/多长度 |

`benchmarks/scripts/gen_multineedle.py` 可再生成 mn 集（固定种子，结果可复现）。

## 3. 复现命令（CPU，16 线程）

```bash
BIN=build-wsl/edge-prefill
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
D=benchmarks/data/explen/len16k
OUT=benchmarks/results/ar006-campaign/$(basename $D)

# 3.1 dense 参考（同一会话必须测一次）
$BIN bench --config benchmarks/matrix/bound/dense-ref.conf \
    --model $MODEL --data-dir $D --out $OUT-dense --bench-skip-dense

# 3.2 图内页门控 p16 / p24
$BIN bench --config benchmarks/matrix/bound/bound-p16-a24-b025.conf \
    --model $MODEL --data-dir $D --out $OUT-boundp16 --bench-skip-dense
$BIN bench --config benchmarks/matrix/bound/bound-p24-a24-b025.conf \
    --model $MODEL --data-dir $D --out $OUT-boundp24 --bench-skip-dense
```

看结果：每个目录里的 `*_raw.csv`（`ttft_ms`、`quality`、`qg_source`、`repeats`）。

```bash
python3 benchmarks/scripts/assemble_ar006.py benchmarks/results/ar006-campaign
```

生成 `matrix.csv` / `matrix.md`：每个数据集一行，dense 与 bound 的 ttft、净加速、
质量；质量 <0.9 的格子会被标红（非零退出码），便于 CI 把关。

## 4. 判读标准

| 指标 | 通过线 | 说明 |
|---|---|---|
| 质量 q | ≥0.9（16K/32K niah） | q=needle 精确命中率；多针集要求全部针 |
| 净 TTFT 加速 | 16K ≥8×，32K ≥10×（1.5B CPU） | `dense_ttft / bound_ttft`，同会话 |
| `qg_source` | `attention-bound` | 证明走的是图内门控而非行集回退 |
| `semantic_score_ms` | 0 | 证明嵌入不在关键路径上 |
| 4K/8K | 质量不降 | local 窗覆盖全部 KV 时退化为 local-only，**不应有加速**；有加速说明有 bug |

若 16K 质量掉到 0.9 以下：先把 `qg_sel_pages` 加到 24 或 32，再看
`qg_local_rows`（默认 0 = ubatch 自身跨度）。这是唯一需要调的质量旋钮。

## 5. GPU（可选）

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DEDGE_PREFILL_WITH_CUDA=ON -DGGML_CUDA=ON
cmake --build build-cuda -j
CC=gcc CXX=g++ nvcc --version   # 需 CUDA 12.x
```

门控图只用 `pool_1d` / `top_k` / `get_rows` / `concat` / `add`，均在 CUDA 后端有实现；
GPU 上跑同样的三条命令，把 `--config` 换成 `bound-p24` 与 `n_gpu_layers = 99`
（`benchmarks/matrix/bound/models/qwen2.5-7b-bound-p24.conf` 里有 7B 模板）。
注意 FlashAttention 必须关闭（`fa=0`）：门控分支走非 FA 的 KQ 路径。

## 6. 换成你自己的模型/任务

1. 复制 `benchmarks/matrix/bound/models/qwen2.5-7b-bound-p24.conf`，改 `threads` /
   `n_gpu_layers`，`qg_bound = true`，`qg_sel_pages` 从 24 起；
2. 数据目录按 `benchmarks/data/explen/len16k` 的结构准备（`manifest.json` +
   每个用例一个 context `.txt` 和 `cases.jsonl`，字段见 `src/dataset*.cpp`）；
3. 先跑 dense 参考，再跑 bound，比较 `dense_ttft / bound_ttft` 与 `quality`；
4. 32K 以上建议 `mode = balanced` + `budget = 0.25`（默认），并把 `qg_sel_pages`
   提到 32；页数越大质量越稳、加速越小，报告里两者都要写。

## 7. AR007：预算-质量曲线与自适应预算

两条独立的链路，可以只做其中一条：

### 7.1 先测你机器上的曲线（推荐，约 30-60 分钟 CPU）

```bash
./repro.sh budget        # 4K/8K，lexical+embedding，自动写 curve 资产

# 想覆盖全部四个长度（本仓库曲线的来源）：
python3 benchmarks/scripts/sweep_budget.py \
    --bin build-wsl/edge-prefill \
    --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --datasets explen/len4k,explen/len8k,explen/len16k,deep/len32k \
    --budgets 0.05,0.08,0.12 --signals lexical \
    --out benchmarks/results/ar007-sweep
```

产物：`summary.md`（人类表 + 推荐值）、`curve.json`（机器可读）、
`budget_curve.txt`（planner 输入，带 `model_sha256`）。**换了模型要重测**：
指纹不符时 runner 会清空曲线并告警，绝不套用别的模型的数字。

### 7.2 用曲线跑自动预算，并与固定预算对比

```bash
BIN=build-wsl/edge-prefill
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
CURVE=benchmarks/results/ar007-sweep/budget_curve.txt

# auto：conf 里不写 budget，只写 budget_curve
printf '%s\n' "mode = balanced" "attn_top_k = 24" "qg_bound = false" \
  "budget_curve = $CURVE" "threads = 16" "max_new_tokens = 16" "repeats = 1" \
  > /tmp/ar007_auto.conf
$BIN bench --config /tmp/ar007_auto.conf --model $MODEL \
    --data-dir benchmarks/data/explen/len16k --out /tmp/ar007-auto --bench-skip-dense

# 机器无关的自检（质量 + 审计列 + 选择 token 数，不比绝对延迟）
python3 benchmarks/scripts/check_expected_ar007.py \
    benchmarks/results/ar007-campaign benchmarks/matrix/budget/expected.csv
```

判读：

| 指标 | 通过线 | 说明 |
|---|---|---|
| `budget_source` | `model` | 证明曲线真的生效；`override` 说明 conf 里还写了 budget |
| 质量 q（8K/16K niah） | ≥0.95 | 与 `expected.csv` 同量级 |
| 净加速（同会话 dense） | ≥10× | 本仓库 8K 29×、16K 38×、32K 55×（Xeon 6234） |
| 32K 的 `qg p24+local32` 相对 AR006 | TTFT 降 ≥30% | 页打分去拷贝的收益；质量必须仍为 1.0 |
| 多针 `deep/multineedle` | 不要用 auto | 选择级稀疏在 5% 预算下丢事实（q≈0.33）；用 `qg_bound=on` |

曲线非单调是**预期行为**，不是 bug：本仓库 16K 在 b=0.05 拿到 q=1.0、
b=0.25 只有 0.36。若你的机器上推荐值明显偏离（例如 4K 也要 0.25），
先看 `summary.md` 里 `floor-missed` 标记，再决定是否手工覆盖 `budget`。

## 8. AR008：短语准入与组合路由

### 8.1 一条命令看懂收益

```bash
./repro.sh ar008
```

它做三件事：跑组合矩阵（5 数据集 × 4 组合，含 dense 参照）→ 由实测生成
`benchmarks/results/ar008-portfolio/portfolio.txt` → 用 `check_portfolio.py`
校验"路由组合不劣于最佳固定组合"（Pareto）。产物：`matrix.csv/matrix.md`。

### 8.2 单独验收短语准入（多针判别符）

```bash
BIN=build-wsl/edge-prefill
MODEL=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
for pw in 0.0 0.5; do
  printf '%s\n' "mode = balanced" "budget = 0.05" "attn_top_k = 24" \
    "qg_bound = false" "embedding_model =" "phrase_weight = $pw" \
    "threads = 16" "warmup = 0" "max_new_tokens = 16" "repeats = 1" > /tmp/pha_$pw.conf
  $BIN bench --config /tmp/pha_$pw.conf --model $MODEL \
      --data-dir benchmarks/data/deep/multineedle --out /tmp/pha_$pw --bench-skip-dense
done
```

判读：`phrase_weight=0` 应复现 AR007 的 q≈0.33（唯一判别符是单个数字，被过滤丢弃）；
`0.5` 应达到 q≥0.9。审计列 `phrase_blocks`（短语命中块数）与 `specific_terms`
（判别词数）为 0 说明你的数据里没有可判别短语——这时短语准入不会改变任何结果。

### 8.3 组合路由怎么用

```bash
# 由实测表驱动（推荐）：conf 里只写 portfolio
printf '%s\n' "mode = balanced" "attn_top_k = 24" "qg_bound = false" \
  "budget_curve = benchmarks/results/ar007-sweep/budget_curve.txt" \
  "portfolio = benchmarks/results/ar008-portfolio/portfolio.txt" \
  "threads = 16" "max_new_tokens = 16" "repeats = 1" > /tmp/routed.conf
$BIN bench --config /tmp/routed.conf --model $MODEL \
    --data-dir benchmarks/data/explen/len32k --out /tmp/routed-32k --bench-skip-dense
```

看 `raw.csv` 的 `portfolio_choice` / `portfolio_reason`：

| 值 | 含义 |
|---|---|
| `selection` | 选择级稀疏（最快，质量达标时的默认） |
| `gate-p16` / `gate-p24-local32` | 图内页门控（质量优先臂，路由在 `selection` 达不到质量底线时选择它） |
| `none` + `disabled` | 没配 `portfolio`，路由未启用（默认行为） |
| 任意 + `fallback:model-mismatch` | 表是为别的模型生成的，已自动回退到 `selection` |
| 任意 + `user-override` | conf 里显式写了 `qg_bound`，用户意图优先 |

### 8.4 判读标准

| 指标 | 通过线 | 说明 |
|---|---|---|
| 多针 q（b=0.05） | ≥0.9 | AR007 为 0.33；这是判别符修复的直接证据 |
| 单针 8K/16K q | ≥0.95 | 短语准入不得伤害原任务 |
| PHA 开销 | ≤2% TTFT | `phrase_weight=0` vs `0.5` 同会话对比 |
| 组合闸门 | `check_portfolio.py` PASS | 路由组合 Pareto 不劣于最佳固定组合 |
| 注入劣化 | 必须 FAIL | `--inject=len16k:gate-p16` 应报违规（自测闸门有效） |

## 9. AR009：多事实覆盖准入

### 9.1 一条命令

```bash
./repro.sh ar009          # 生成两档数据 + 消融 + 闸门（约 10-15 分钟 CPU）
```

### 9.2 手动跑（推荐在大机器上改规模）

```bash
# 1) 生成数据（确定性，固定 seed）
python3 benchmarks/scripts/gen_multifact.py \
  --out benchmarks/data/deep/multifact8k --words 7500 --cases 8 \
  --facts 6 --asked 3 --rows-per-context 2 --seed 41
python3 benchmarks/scripts/gen_multifact.py \
  --out benchmarks/data/deep/multifact8k-hard --words 7500 --cases 8 \
  --facts 12 --asked 6 --rows-per-context 2 --seed 43

# 2) 消融：coverage_weight=0 是 AR008，3 是 AR009
python3 benchmarks/scripts/ar009_ablation.py \
  --bin build-wsl/edge-prefill --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --out benchmarks/results/ar009-multifact --force

# 3) 闸门（地板值，不是抄录值）
python3 benchmarks/scripts/check_expected_ar009.py \
  benchmarks/results/ar009-multifact \
  benchmarks/matrix/multifact/expected_ar009.csv
```

### 9.3 判读标准

| 指标 | 通过线 | 说明 |
|---|---|---|
| multifact8k q_mean | ≥0.85（实测 0.878） | AR008 为 0.488 |
| multifact8k-hard q_mean | ≥0.60（实测 0.652） | AR008 为 0.261；6 条答案句受预算限制 |
| `coverage_queries` | ≥ 提问数−1 | 证明事实确实进了上下文 |
| 单事实族 | 选择逐位不变 | 4K/16K/mn 的 `selected_tokens` 与质量都不变 |
| 负向自测 | 必须 FAIL | 把 cw=3 行改写成 cw=0 值，闸门应报 3 类违规 |

模型规模更大时 `q` 的绝对值会变，但"cw=3 明显高于 cw=0"这一**同条件对比**
应保持；若不再成立，先查 `coverage_queries` 是否为 0（说明子查询没提取出来）。

## 10. AR010：证据优先准入与组合再择优

### 10.1 一条命令

```bash
./repro.sh ar010     # 五臂矩阵 + 重新生成组合表 + Pareto 闸门 + 注入自测
```

### 10.2 手动跑（推荐在大机器/GPU 上改规模）

```bash
# 1) 五臂矩阵（4K/8K/16K/32K/多针；dense 每数据集一次）
python3 benchmarks/scripts/ar010_matrix.py \
  --bin build/edge-prefill --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --out benchmarks/results/ar010-evidence-first --force

# 2) 多事实族（固定 budget=0.05，答案需要 48 token）
python3 benchmarks/scripts/ar010_matrix.py \
  --bin build/edge-prefill --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --out benchmarks/results/ar010-evidence-first \
  --datasets deep/multifact8k,deep/multifact8k-hard \
  --arms selection,selection-prior,gate-p16,gate-p24-local32 \
  --flat-budget 0.05 --max-new-tokens 48

# 3) 生成表并过闸门
python3 benchmarks/scripts/ar010_report.py benchmarks/results/ar010-evidence-first
python3 benchmarks/scripts/ar010_report.py benchmarks/results/ar010-evidence-first --gate
python3 benchmarks/scripts/ar010_report.py benchmarks/results/ar010-evidence-first \
  --gate --inject deep/len32k:selection-prior   # 必须 FAIL
```

### 10.3 GPU（CUDA）复现

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda -j

# 稀疏 + 门控一致性（同一 conf，--n-gpu-layers 99 打开全量卸载）
./build-cuda/edge-prefill bench --config <conf> --model M.gguf \
  --data-dir benchmarks/data/explen/len8k --out /tmp/gpu-check --n-gpu-layers 99
```

判读：`quality` 与 `selected_tokens` 必须与 CPU 完全一致（选择在主机侧），
`qg_source` 必须是 `attention-bound`。本机实测（RTX 5000 16GB）稀疏 4K 与
门控 8K 均一致，GPU 快 4.8-5.5×；dense 路径在 CUDA 上会切成数百个 graph split，
未纳入结论。

### 10.4 判读标准

| 指标 | 通过线 | 说明 |
|---|---|---|
| 4K/8K/16K/多针 q | =1.000 | 不得为了省 token 掉质量 |
| 32K q（无门控） | ≥0.99 | AR009 为 0.875，必须靠门控 |
| 32K TTFT | 低于门控臂 | 本机 2260 ms vs 5443 ms |
| 多事实 easy/hard | ≥0.80 / ≥0.60 | 更正后的 AR009 地板 |
| 组合闸门 | PASS 7/7 | 路由不劣于最佳固定臂 |
| 注入劣化 | 必须 FAIL | `--inject deep/len32k:selection-prior` |
| GPU 一致性 | q/sel 逐例一致 | 选择在主机侧，图侧需 `qg_source` 正确 |

## 11. AR011：结构锚点预算与证据预算分离

### 11.1 一条命令

```bash
./repro.sh ar011     # 7 数据集 x 3 臂 + 组合表 + 不劣于 AR010 的闸门 + 注入自测
```

### 11.2 手动跑（大机器上可以只跑一个子集）

```bash
# 全量（4K/8K/16K/32K/多针/多事实两档）；多事实的 budget 与答案长度由脚本按数据集决定
python3 benchmarks/scripts/ar011_matrix.py   --bin build/edge-prefill --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf   --out benchmarks/results/ar011-structure-bounded --force

# 只看某几个数据集（已有 raw.csv 的单元会复用，--force 会先清空该单元）
python3 benchmarks/scripts/ar011_matrix.py   --datasets explen/len4k,deep/multineedle --arms ar010,ar011   --out benchmarks/results/ar011-structure-bounded

# 生成表 + 闸门 + 注入自测（必须 FAIL）
python3 benchmarks/scripts/ar011_report.py benchmarks/results/ar011-structure-bounded
python3 benchmarks/scripts/ar011_report.py benchmarks/results/ar011-structure-bounded --gate
python3 benchmarks/scripts/ar011_report.py benchmarks/results/ar011-structure-bounded \
  --gate --inject explen/len8k:ar010
```

### 11.3 配置项（AR011）

| conf 键 | CLI | 默认 | 作用 |
|---|---|---|---|
| `structure_budget` | `--structure-budget N` | 0 | 结构锚点绝对 token 预算；0 = AR010 记账；仅在锚点主导预算时生效 |
| `anchor_span_radius` | `--anchor-span-radius N` | 0 | 最佳词法锚点窗口半径（0 = ±2 块）；实测 1 会伤多事实 |
| `span_admission` | `--span-admission 0\|1` | 0 | 单事实下把证据块裁成句子级 token 段 |
| `span_min_tokens` | `--span-min-tokens N` | 24 | 句子过短时的最小段长 |
| `coverage_reserve` | `--coverage-reserve 0\|1` | 0 | 每事实保底一块（实测无稳定收益，默认关） |

### 11.4 判读标准

| 指标 | 通过线 | 说明 |
|---|---|---|
| 所有数据集 q_mean | 不低于 AR010 −0.01 | 单针 0.95 地板，多事实以 AR010 实测为地板 |
| 准入 token | 不超过最优臂 1.05× | 4K 146、8K 188、多针 247、16K/32K 280/525 |
| TTFT | 不超过最快臂 1.05× | 同会话比较，绝对值随机器浮动 |
| 多事实 hard `q_min` | ≥0.57 | AR010 为 0.087 |
| 组合闸门 | PASS 7/7 | 全部路由到 ar011 |
| 注入劣化 | 必须 FAIL | `--inject explen/len8k:ar010` → token/TTFT 违规 |
| GPU 一致性 | q/sel 逐例一致 | 准入在 host 侧；本机 4K 122/121/170/172 逐例相同，45.2 ms vs 1091.5 ms |
| 开关全关 | 与 AR010 逐 token 一致 | 三个新开关默认 0，`selector_structure_budget_is_inert_*` 用例守护 |

## 12. 报告模板

```
机器: <CPU/GPU, 核心数, 内存>, 会话: <日期>
模型: <name> SHA256 <...>
dense  : <16K ttft> / <32K ttft>
bound  : p<k> 16K ttft / 32K ttft, quality ...
净加速 : <16K>× / <32K>×
env    : <results dir>/<ts>_env.json
曲线   : <budget_curve.txt 的四个键值 + budget_source 计数>
```
