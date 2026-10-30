# [AR016] 详细设计

## 1. 总体结构

```
CLI/conf ──► ExecPlan(请求: device/threads/ubatch/flash)          ← FR-1
                 │
                 ▼
            LlamaBackend.load_model()  ── llama_log_set 捕获 "offloaded N/M"
                 │                        + ggml_backend_dev 枚举
                 ▼
            PlacementReport{requested, offloaded, total, device_name, state}
                 │  不满足请求 → Status::Unavailable（除非 --allow-cpu-fallback）
                 ▼
BenchmarkRunner ──► CostModel{a,b,r2}（probe 落盘, FR-2）
                 │
                 ▼
   budget_target_ms>0 ? 求解最大可行预算（FR-3） : 曲线预算（AR007 原行为）
                 │
                 ▼
      select → prefill → 审计列（device/offloaded/threads/plan_source/budget_source）
```

新增文件：

| 文件 | 内容 |
|---|---|
| `include/edge_prefill/exec_plan.h` / `src/exec_plan.cpp` | `ExecPlan`、`PlanSource`、`DeviceState`、`PlacementReport`、JSON 往返 |
| `include/edge_prefill/cost_model.h` / `src/cost_model.cpp` | `CostSample`、`CostModel`、`fit_cost_model`、`predict_ttft_ms`、`max_tokens_within` |
| `src/probe.cpp`（或并入 `main.cpp` 的 `probe` 子命令） | 真实 GGUF 多长度测量 → plan JSON |

## 2. ExecPlan（FR-1）

```cpp
enum class PlanSource { Default, Conf, Cli, Probe };
enum class DeviceState { Cpu, Gpu, CpuFallback, Unverified };

struct ExecPlan {
    int32_t n_gpu_layers = 0;      // -1 = 全部
    int32_t n_threads = 0;         // 0 = auto
    int32_t n_threads_batch = 0;   // 0 = 跟随 n_threads
    uint32_t n_ubatch = 512;
    uint32_t n_batch = 2048;
    bool flash_attn = false;
    PlanSource source = PlanSource::Default;
    // 合并：低优先级被高优先级覆盖，逐字段记录来源
    static ExecPlan merge(const ExecPlan& lower, const ExecPlan& higher);
};

Status ExecPlan::validate() const;   // n_ubatch>0、n_ubatch<=n_batch、n_threads>=0 ...
Status ExecPlan::to_json(std::string* out) const;
Status ExecPlan::from_json(const std::string& text, ExecPlan* out);
```

字段级优先级：`Cli > Conf > Probe > Default`。合并以"该字段是否被显式设置"为准，
因此 `ExecPlan` 每个整数字段用 `-1` 表示未设置（`n_gpu_layers` 用 `INT32_MIN` 表示未设置，
以保留 `-1 = 全部卸载` 的语义）。

## 3. 放置校验（FR-1）

1. `llama_supports_gpu_offload()`：构建是否支持卸载。
2. `ggml_backend_dev_count()` / `ggml_backend_dev_get(i)` / `ggml_backend_dev_type()==GPU` /
   `ggml_backend_dev_name()` → 得到 `device_name`（如 `CUDA0 (Quadro RTX 5000)`）。
3. `llama_log_set(cb, &capture)` 包住 `llama_model_load_from_file`，解析
   `load_tensors: offloaded (\d+)/(\d+) layers to GPU` → `offloaded`、`total`。
   回调内不分配（固定 char 缓冲 + `snprintf` 解析），加载后恢复原回调。
4. 判定：

| 条件 | state | 行为 |
|---|---|---|
| `plan.n_gpu_layers == 0` | `Cpu` | 正常 |
| `> 0` 且 `offloaded > 0` | `Gpu` | 正常 |
| `> 0` 且 `offloaded == 0` 且允许回退 | `CpuFallback` | 记 `device_state=cpu(fallback)` + `log_warn` |
| `> 0` 且 `offloaded == 0` 且不允许回退 | — | `Status::error(Unavailable)` |
| 日志行缺失（版本差异） | `Unverified` | 不失败，但审计列写 `unverified`，闸门按 FAIL 处理 |

> 为什么不用日志以外的办法：llama.cpp 该版本没有"已卸载层数"的公开 getter；
> 日志是 llama.cpp 自己报告的 ground truth。此限制写入 design §9。

## 4. 线程/ubatch 贯通（FR-1）

- `BenchmarkConfig` 增 `n_threads`、`n_threads_batch`、`n_ubatch`、`plan_source`、`allow_cpu_fallback`。
- `main.cpp` bench 路径：`bench.n_threads = args.threads != 0 ? args.threads : cfg.n_threads;`（CLI 覆盖 conf）；
  `bench.n_gpu_layers = args.n_gpu_layers 显式给出 ? args值 : cfg.n_gpu_layers;`。
  判"显式给出"用 `args.n_gpu_layers != 0` 不够（0 是合法值），因此在 `Args` 加 `bool n_gpu_layers_set`。
- `benchmark.cpp` 建上下文处：

```cpp
ctx_opts.n_threads = cfg.n_threads;            // 0 = auto
ctx_opts.n_threads_batch = cfg.n_threads_batch;
ctx_opts.n_ubatch = cfg.n_ubatch ? cfg.n_ubatch : 512;
```

- 数值不变性：线程数只影响归约顺序，理论上 Q4 累加顺序变化可能带来末位浮点差异；
  闸门用"质量逐例相同"（f1 为定长答案，稳定）而不是逐 bit logits 相同；**同 plan 下**才要求逐 bit。

## 5. 成本模型（FR-2）

最小二乘 `ttft = a + b·tokens`：

```
b = Σ(ti-t̄)(yi-ȳ) / Σ(ti-t̄)²,  a = ȳ - b·t̄
r2 = 1 - SSE/SST,  rel = max|ŷ-y|/y
```

约束：`n>=5`、`b>0`、`SST>0`、`r2>=0.90`，否则 `Status::error(Internal, "cost-model-unfit")`。

保守预测：`predict(t) = (a + b·t) * (1 + safety)`，`safety` 默认 0.15（覆盖主机抖动）。

反解：`max_tokens(target) = floor( target / ((1+safety)·b) - a/b )`，夹到 `[lo, hi]`。

`probe` 子命令：从数据集中取真实 context，按 4 个长度切片（64/128/256/512，若原文更长则按比例截断），
每个长度跑 3 次取最小值，得到样本；输出：

```json
{"schema":1,"plan":{...},"device":{"name":"CUDA0 ...","requested":99,"offloaded":29,"total":29,"state":"gpu"},
 "cost":{"a_ms":201.3,"b_ms_per_token":5.49,"r2":0.997,"n":4,"rel_error":0.06},
 "samples":[{"tokens":146,"ttft_ms":1023.1},...]}
```

## 6. SLA 预算规划（FR-3）

在 `BenchmarkRunner` 的 case 循环内，portfolio 决定 `budget0` 之后：

```
budget = budget0
if (cfg.budget_target_ms > 0 && !snapshot_hit && !session_reuse) {
  for (iter = 0; iter < 4; ++iter) {
     sel = selector.select(tokens, budget, opts)        // 真实选择，不用模型近似
     pred = predict(sel.token_indices.size())
     if (pred <= target) break
     next = clamp(sel->tokens * target / pred, budget_min, budget_max)
     if (next <= budget_min_tokens) { budget_source="cost-model-clamped"; break; }
     budget = next
  }
  budget_source = "cost-model"
}
```

- 选择是可确定的纯 CPU 计算（无模型），重复调用不影响数值；但**计时区外的额外选择**必须计入
  `semantic_score_ms` 之外的新列 `plan_select_ms`，避免偷时间。
- 已经命中快照/会话复用时 TTFT 与预算无关，跳过规划（`budget_source=reuse`）。
- 上下界来自 `BudgetCurve`：`budget_min`/`budget_max` 若未配置，则用 `[0.01, 0.5]`。

## 7. 审计列（新增）

```
device,gpu_layers_offloaded,gpu_layers_total,n_threads,n_threads_batch,n_ubatch,
plan_source,device_state,predicted_ttft_ms,plan_select_ms
```

`budget_source` 复用既有列，新取值 `cost-model` / `cost-model-clamped` / `reuse`。

## 8. 测试

| 层级 | 用例 |
|---|---|
| 纯逻辑（无模型） | ExecPlan 字段合并优先级、JSON 往返、非法字段拒绝；成本模型拟合/r² 门槛/退化样本/保守上界/反解夹取；SLA 求解边界 |
| 真机 | 放置校验：`n_gpu_layers>0` 且无 GPU 设备 → `Unavailable`；允许回退 → `CpuFallback` + 审计列；线程贯通：`n_threads=4` 时审计列与计划一致 |
| 真机 | probe：真实 GGUF 拟合 `r2>=0.9`；SLA：目标 1500 ms 时选出的预算其预测值 ≤ 目标，且实际 TTFT 的中位数不超目标 1.5× |
| 集成 | 同一 plan 下 AR013/AR014/AR015 的质量逐例不变 |

## 9. 风险与取舍

| 风险 | 处理 |
|---|---|
| 日志行解析随上游变化失效 | 固定 commit；失败降级为 `Unverified` 并由闸门判 FAIL（不会静默通过） |
| 成本模型在抖动主机上 r² 低 | 取 min-of-3；r²<0.9 就拒绝启用 SLA（宁可不优化） |
| SLA 规划让预算变大导致质量噪声 | 硬约束：质量不得下降；不满足就只允许"降延迟"方向（预算不放大） |
| 线程数改变累加顺序 | 只承诺"质量逐例相同"，不承诺逐 bit；同 plan 才要求逐 bit |
| 本机 GPU 低 P-state | 如实记录；plan 由 probe 生成，换高频 GPU 自动受益 |

## 10. 与既有 AR 的关系

- AR007：预算曲线仍是默认；SLA 只是在其上做**夹取式**求解，不替换。
- AR013：紧凑 KV 决定 `a`（固定开销）与 `b`（每 token），成本模型自然吸收。
- AR014/AR015：命中即短路，不参与 SLA 求解。
- AR012/AR011/AR009：SLA 改变的是预算数值，准入规则不变，因此质量口径一致。

## 11. 实现顺序（TDD）

T001 ExecPlan（红：合并/JSON/校验）→ T002 后端放置信息（红：无 GPU 设备必须报错）→
T003 bench 贯通（红：审计列反映计划）→ T004 成本模型（红：r² 门槛 + 反解边界）→
T005 probe → T006 SLA 求解 → T007 矩阵/闸门 → T008 实验 → T009 文档/勘误 → T010 归档推送。

## 12. 实现差异（as-built）

### 12.1 plan 用 conf 覆盖层，不用 JSON

设计写"plan JSON"，实现改为 `plan.conf`（`key = value`）+ `probe_samples.csv` +
`probe_device.txt`。原因：仓库已经有一个经过测试的 conf 解析器，再引入第二个配置语法
只会多一条出错路径；覆盖层也能直接追加到任何实验 conf 里，复现时肉眼可读。
（`ExecPlan::to_conf()` 只输出显式设置过的字段。）

### 12.2 放置校验用日志行，而不是"假设"

该版本 llama.cpp 没有"已卸载层数"的公开 getter。实现为：
`ggml_backend_dev_*` 枚举设备（构建能力 + 设备名）+ `llama_log_set` 捕获
`offloaded N/M layers to GPU`。回调同时把每行原样转写 stderr，所以没有丢日志。
日志缺失 → `unverified`（不失败、但闸门判 FAIL），绝不假装成功。

### 12.3 成本模型允许负截距

实测 CPU `a = -3.12 ms`：prefill 次线性，固定开销被批大小摊薄。初版拒绝 `a<0`，
导致 SLA 静默失效（`cost_a_ms >= 0` 不成立）。改为**只拒绝"采样区间内预测非正"的模型**，
预测值本身夹到 ≥0。

### 12.4 SLA 求解用真实 selector 迭代，不用预算→token 模型

`budget` 到 `tokens` 的映射是非线性的（块对齐 + 锚点 + 句子补齐），拟合它等于再引入一个
易错模型。实现改为：选一次 → 预测 → 超标就按 `target/pred` 收缩预算 → 重选（≤8 轮），
并记录最终选择的预测值（初版记录的是上一轮的预测，导致 1 个用例记录 2527 ms 而实际
选择只需 2400 ms）。

### 12.5 闸门随设备调整

- CPU：要求"计划臂比硬件默认快 ≥1.10×"；
- GPU：线程与 GPU 计算无关，`thread-speedup` 记为 **n/a**，改为检查
  `placement-offload`（29/29）与"同 SLA 下的 token 数"（跨矩阵对比，写在报告中）；
- 统计量一律用 `ttft_min`，SLA 实际值用**每数据集中位数**——本机单点可差 2.4×。

### 12.6 顺带修掉的测量干扰

模型 SHA256（1 GB，秒级 CPU）原先在**每个 cell 的 portfolio 检查**里现算，与相邻 cell
的计时 prefill 争 CPU。已提升为每次 run 一次。这是"先保证测量可信"的直接产物。

### 12.7 未实现

- 跨问题共享"查询无关头部"的 KV 池：实测结构块只占准入 token 的 8–10%，收益 <15%，不做；
- 分段成本模型（token 跨度 >1024）：当前用保守系数 1.15 + 只在采样区间内规划；
- GPU 锁频（WSL 拒绝 `nvidia-smi -lgc`）：环境限制，只记录。
