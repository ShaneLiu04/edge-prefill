# [AR016] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR016 |
| AR 主题 | 执行计划一等化（设备/线程/ubatch + 放置校验）与延迟预算规划（SLA），把"跑在哪、跑多快"变成可验证、可求解的对象 |
| 关联 SR | 承接 AR007（预算曲线）、AR013（KV 布局）；修正 AR015 的设备实验记录 |
| 日期 | 2026-10-30 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 量化现状（本次 AR016 现场实测，非历史引用）

**（1）bench 的执行参数是失控的**

| 观测 | 证据 |
|---|---|
| `--threads` 在 bench 路径完全不生效 | `BenchmarkConfig` 无 `n_threads` 字段；`src/benchmark.cpp:403-404` 硬写 `n_threads = 0` → 由后端解析为 `hardware_concurrency()`（本机 16） |
| 所有已入库 conf 写了 `threads = 8`，全部被忽略 | `benchmarks/results/*/_conf/*.conf`；`src/main.cpp:1529` 只把它传给 embedding |
| `warmup` 生效，但所有已入库 conf 设为 0 | `src/benchmark.cpp:1408`；conf |
| `--n-gpu-layers`（CLI）在 bench 路径被丢弃 | `src/main.cpp:1560` 只取 `cfg.n_gpu_layers`（conf 键），CLI 值从不写入 bench |

**（2）线程数对 TTFT 的影响（本机实测，len4k 单例，270 token，`run` 子命令）**

| threads | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 3172 | 1853 | 1156 | **1006** | 1375 |
| 相对最优 | 3.15× | 1.84× | 1.15× | 1.00× | 1.37× |

即：bench 一直用的 16 线程比本机最优（12）慢 **37%**；比 8 线程还慢 19%（SMT 争用）。

**（3）"GPU 结果"其实不是 GPU 结果**

用 conf 键 `n_gpu_layers = 99` 重跑，llama.cpp 明确打印 `offloaded 0/29 layers`（CLI 传参路径）与 `offloaded 29/29`（conf 键路径）。AR015 入库的 `ar015-kv-snapshot-gpu/` 走的是 CLI 传参路径，因此**那份"Cuda 结果"实际是 CPU 结果**。这是必须更正的记录错误。

**（4）真实 GPU 在本机的上限（实测，29/29 卸载）**

| 场景 | CPU（CUDA 构建、ngl=0） | GPU（ngl=99） | 比值 |
|---|---:|---:|---:|
| dense 4K prefill（3962 token） | 35.0 s（116 tok/s） | 24.0 s（166 tok/s） | 1.4× |
| sparse 4K prefill（634 token） | 3889 ms | 2378 ms | 1.6× |

GPU 监控：`utilization.gpu=99%`、`clocks.sm=645 MHz`（上限 2100）、`pstate=P8`、`power=31 W`（上限 230 W）。
`nvidia-smi -lgc` 在 WSL 下报 `Unknown Error`，**无法锁频**。结论：本机 GPU 处于低 P-state，
算力未被释放；这既不是代码 bug，也不能靠参数修好，必须如实记录，并让框架在"GPU 变快"时自动受益。

**（5）成本模型**：TTFT 对 token 数近似仿射。用 ar013 的 8 个数据集（CPU 16 线程）拟合
`TTFT ≈ 200 ms + 5.5 ms/token`（残差见 §3.2）。这意味着**预算（token 数）与延迟之间有一个可测的线性关系**，
而 AR007 的预算曲线是静态表，二者从未连起来。

### 1.2 目标

1. **执行计划（ExecutionPlan）成为一等对象**：`(device, n_gpu_layers, n_threads, n_threads_batch, n_ubatch, flash_attn)`
   显式给出、加载后**校验真的生效**（卸载层数/实际设备），写进审计列；请求 GPU 却拿到 0 层必须**报错**而不是静默降级。
2. **成本模型可测量、可落盘**：一次 `probe` 得到 `TTFT = a + b·tokens`（按设备/模型/KV 精度），带样本证据。
3. **延迟预算规划**：给定 `budget_target_ms`，用成本模型求"在延迟目标内能负担的最大预算"，
   从而**把设备速度换成质量**（GPU 快 → 同延迟下预算更大 → 证据更多），而不是只换热启动延迟。
4. **更正 AR015 的 GPU 记录**，并把"如何验证设备真的生效"写进 README，防止再次发生。

## 2. 需求范围

**In Scope：**
- `include/edge_prefill/exec_plan.h` + `src/exec_plan.cpp`：`ExecPlan`（字段、`Status` 校验、JSON 往返）、
  `DeviceInfo`（名称、卸载层数、线程数、来源）、`CostModel{a,b,r2,n}` + `fit_cost_model` + `predict_ttft`
- `src/backend.cpp`：`loaded_gpu_layers()`（读 llama.cpp 实际卸载层数）、`device_name()`
- `src/benchmark.cpp` / `src/main.cpp` / `src/config.cpp`：`threads`/`n_threads_batch`/`ubatch`/`n_gpu_layers` 在 bench 真正生效；
  加载后校验；`plan_source` 等审计列
- SLA 规划：`budget_target_ms`；在 `BenchmarkRunner` 内按成本模型求最大可行预算（受 `budget_curve`/`budget_min/max` 约束）
- `include/edge_prefill/cost_probe.h`（或 `exec_plan` 内）：`probe` 子命令 → 生成 plan JSON（含样本与拟合）
- 矩阵/闸门脚本 `ar016_matrix.py` / `ar016_report.py`；`repro.sh ar016`
- 文档：README（"如何确认设备真的生效"）、research §18、architecture、portfolio、组件详设、AR015 勘误

**Out of Scope：**
- 修改 llama.cpp 上游（只读其 API/日志）
- 跨机分布式；GPU 锁频/超频（WSL 不支持，明确不尝试）
- 猜测/模拟设备性能（禁止 mock：GPU 不可用时必须标记为 `device=cpu(fallback)` 而不是假装）

## 3. 功能需求

### 3.1 FR-1 执行计划贯通与放置校验

- `ExecPlan` 字段：`n_gpu_layers`、`n_threads`、`n_threads_batch`、`n_ubatch`、`flash_attn`、`source`（`conf|cli|probe|default`）。
- 优先级：CLI > conf > probe > default；任何来源的组合必须能被审计复现（审计列写 `plan_source`）。
- 加载模型后查询**实际**卸载层数 `offloaded_layers` 与设备名；若 `plan.n_gpu_layers > 0 && offloaded == 0`
  → 返回 `ErrorCode::Unavailable`，除非显式 `--allow-cpu-fallback`，此时 `device_state = cpu(fallback)` 写入审计列并在 stderr 告警。
- 审计列新增：`device`、`gpu_layers_offloaded`、`n_threads`、`n_threads_batch`、`n_ubatch`、`plan_source`、`device_state`。

### 3.2 FR-2 成本模型探针

- `edge-prefill probe --model M [--data-dir D] --threads N ... --out plan.json`：
  用真实 GGUF 在若干 token 长度（例如 64/128/256/512/1024，取自真实数据集切片）上测 prefill TTFT，
  最小二乘拟合 `TTFT = a + b·tokens`，输出 `{a_ms, b_ms_per_token, r2, n, samples[], plan{}, device_info{}}`。
- `predict_ttft(model, tokens)` 必须给出**保守上界**（`a + b·tokens` 再乘 `1 + k·max(残差相对误差)`，k 可配）。
- 拟合质量门槛：`r2 >= 0.90` 且样本数 ≥ 5；否则 `Status::error(Internal)`，不得用坏模型去规划。

### 3.3 FR-3 延迟预算规划（SLA → 预算）

- 新配置 `budget_target_ms`（默认 0 = 关闭，保持既有行为）。
- 开启时：`budget = min(budget_cap, largest b s.t. predict_ttft(a,b) <= target)`，
  其中 `budget_cap` 取 `max(curve_budget, budget_max)`；结果写入 `budget_source = cost-model`。
- **不得**突破既有安全边界：求得的 token 数与 `budget_curve` 给定值相比只允许在 `[budget_min, budget_max]` 内变化；
  越界时退回曲线值并记 `budget_source = cost-model-clamped`。
- 质量不下降是硬约束：矩阵必须验证"同 SLA 下 q ≥ 默认曲线 q"（若做不到，就只允许降延迟，不改质量）。

### 3.4 FR-4 设备间的质量-延迟前沿

在同一 `budget_target_ms` 下分别用 CPU 计划与 GPU 计划跑 8 个数据集，产出
`(device, budget_used, ttft, quality)` 前沿；要求：
- 质量不劣于默认组合（逐例 f1 允许 ±0 容差，与 AR013/AR015 同口径）；
- 至少一个数据集上 TTFT 降低 ≥ 20%（线程修正的红利）；
- GPU 计划若无法给出更好前沿，**如实记录 `no-signal`**，不得包装。

### 3.5 FR-5 闸门与注入

- 闸门（`ar016_report.py`）：放置真实（`gpu_layers_offloaded` 与请求一致）、线程生效（`n_threads` 与 conf 一致）、
  TTFT 改善、质量不变、成本模型 r²、SLA 不被突破。
- 注入：①`fake-placement`（把 offload 校验关掉/伪造）必须 FAIL；②`cost-model-skew`（把 b 改小 50%）必须 FAIL（SLA 被突破）。

### 3.6 FR-6 AR015 记录更正

- 新增勘误段落：`benchmarks/results/ar015-kv-snapshot-gpu/REPORT` 与 `docs/research.md` §17、README 中
  的 GPU 数字标注为"实为 CPU（0 层卸载）"，并以 AR016 的真实 GPU 数据替换。
- 保留原 CSV（不删证据），在新勘误里给出复现命令。

## 4. 非功能需求

| 项 | 要求 |
|---|---|
| 数值等价 | 执行计划只影响"怎么跑"，不影响"算什么"：同 plan 下逐 token logits 与改前一致 |
| 错误处理 | 设备不可用 → `Status` + 稳定 `ErrorCode`；禁止静默降级 |
| 可复现 | plan JSON 入库，含设备名/驱动/线程/样本；README 给出验证命令 |
| 性能 | 计划解析/校验开销 < 1 ms（除 probe 外不影响计时区） |
| 兼容性 | `budget_target_ms=0` 与不传 `threads` 时行为与 AR015 完全一致 |

## 5. 约束与假设

- llama.cpp 固定 commit `3bb63cbc…`；不修改上游，只读 `llama_model_n_gpu_layers`/日志。
- 本机 GPU 处于 P8（645 MHz）、WSL 无法锁频 —— 这是**环境事实**，AR 不试图绕过；设计必须让
  "换到高频服务器 GPU" 时自动受益（plan 由 probe 生成，不硬编码）。
- 主机 TTFT 抖动大（同 cell 跨会话可差 2–5×），因此对比只用**同会话、同 plan、交错重复**的最小值。

## 6. 组合分析（为什么是这个 AR）

| 候选方向 | 判断 |
|---|---|
| **本 AR：执行计划 + SLA 预算** | 已有实测证据（37% 线程损失、假 GPU 记录、可拟合的仿射成本模型）；不与任何既有 AR 争夺同一收益通道 |
| KV 池 + 跨问题共享头 | 实测结构只占准入 token 的 ~10%（`structure_tokens 47-55` vs `selected 146-1163`），共享头收益 <15%，**不做**（记负结果） |
| 段级 KV 分页（AR015 §7.6） | llama.cpp 无部分序列 API，`seq_cp` 只能整段复制；不做 |
| 更大大模型/更多解码优化 | 超出 prefill 主题；且解码侧改动会污染质量口径 |

组合关系：AR007 的静态预算曲线被 AR016 的成本模型**约束**而不是替换；AR013 的 KV 布局决定成本模型的截距 `a`；
AR014/AR015 的复用路径在 SLA 规划中作为"已命中则不规划预算"的前置短路。

## 7. 术语说明

- **执行计划（ExecPlan）**：一次运行采用的设备/线程/ubatch/flash 组合，及其来源。
- **放置校验**：加载后核对"请求的 GPU 层数"与"实际卸载层数"，不一致即失败。
- **成本模型**：`TTFT = a + b·tokens` 的仿射拟合，带 r² 与样本。
- **SLA 规划**：给定延迟上限，求最大可行预算（token 数）。
