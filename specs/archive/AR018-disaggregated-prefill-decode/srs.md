# [AR018] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR018 |
| AR 主题 | 预填充/解码分离（disaggregated prefill–decode）：把瓶颈阶段的 prefill 交给 GPU、把带宽受限的 decode 交给 CPU，用跨上下文 KV 交接把两个阶段流水线化 |
| 关联 SR | 承接 AR012/AR013（稀疏上下文与紧凑 KV）、AR015（快照，本 AR 不启用）、AR016（执行计划/放置校验）、AR017（批解码）；对标 DistServe / Mooncake / vLLM 的 prefill-decode 分离与 KV 迁移 |
| 日期 | 2026-10-31 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 量化现状（本机实测，2026-10-31）

AR017 把解码变成"一次前向服务 K 个序列"之后，**瓶颈从解码转移到预填充**：

| 口径（explen/len4k，4 用例，12 线程） | 每用例成本 |
|---|---:|
| 串行（K=1） | TTFT 2578 ms + 解码 6.9 s ≈ 9.4 s |
| AR017 批解码（K=4） | TTFT 2603 ms + 解码 3.2 s/4 例 ≈ 0.8 s → **3.4 s** |
| 其中 prefill 占比 | **76%** |

同一台机器上两个设备的擅长点恰好相反（`/tmp/ar018_spike_a`）：

| 设备 | prefill TTFT（len4k） | 批解码 ms/step（K=2 / K=4） |
|---|---:|---:|
| CPU（12 线程） | 2578–2603 ms | 41.8 / 67.1 |
| GPU（RTX 5000，29/29 层，P8 645 MHz） | **1657 / 1339 ms（1.55–1.94×）** | 139.9 / 161.6（差 2.3×） |

即：**GPU 预填充快 1.6–1.9×，CPU 解码便宜 2.3×/序列**。而两者可以真并行
（`/tmp/ar018_spike_c`：CPU 解码 bench 与 GPU 预填充 bench 并发跑，
并发墙钟 24.5 s = 各自 solo 的 24.5 s，`overlap_efficiency = 2.00×`）。

### 1.2 目标

把"预填充"和"解码"从**同一个上下文里的顺序两段**，变成**两个上下文上的流水线**：

```
            ┌──────────── GPU: prefill ctx (n_seq_max=1) ───────────┐
case i  →   │ 稀疏 prefill（AR012/13）→ get_seq_state → 交接 ──────┐│
            └──────────────────────────────────────────────────────┘│
                                                                    ▼
            ┌──────────── CPU: decode ctx (n_seq_max=K) ────────────┐
            │ set_seq_state → AR017 批解码 → 生成 → 释放序列        │
            └──────────────────────────────────────────────────────┘
```

- **吞吐**：N 个用例的墙钟从 `Σ(prefill_i + decode_i)` 降到 `Σ max(prefill_i, decode_i/K)`；
- **单例质量**：逐例答案与串行路径**完全相同**（KV 是精确交接，不是近似）；
- **设备利用率**：报告 GPU prefill 与 CPU decode 各自的负载与重叠率；
- **回退安全**：任何一步失败（交接校验、放置不符、队列溢出）都回退到 AR017 串行/批路径，
  审计列如实记录，不允许静默降级。

### 1.3 与已有 AR 的关系（组合而非叠加）

| AR | 在 AR018 中的角色 | 冲突处理 |
|---|---|---|
| AR012/AR013 稀疏 prefill + 紧凑位置 | prefill 阶段原样使用 | 交接的 KV 带位置，解码上下文按 `seq_pos_max` 续写 |
| AR016 执行计划 | 两个上下文各自独立放置并校验 | 任一上下文放置不符 → 失败（`allow_cpu_fallback` 才降级） |
| AR017 批解码 | 解码阶段原样使用 | 交接后 seq_id 重编号（prefill 恒为 0） |
| AR015 快照 | **不启用**（快照绑定单上下文/seq 0） | `snapshot_*` 在 pipeline 模式忽略并告警 |
| AR014 会话 | **不启用**（会话依赖跨轮同上下文 KV） | 会话用例走非流水线路径 |

## 2. 需求范围

### 2.1 范围内（FR）

- **FR-1 双后端与独立放置**：`pipeline_mode=1` 时创建两个后端（各自 load 同一模型文件）：
  `prefill` 与 `decode`，可分别指定 `n_gpu_layers`、线程数、`n_ctx`、`n_seq_max`；
  两者都走 AR016 的放置校验，审计列分别给出 `prefill_device_state` / `decode_device_state`。
- **FR-2 跨上下文 KV 交接**：`get_seq_state(seq)` / `set_seq_state(seq, data)` 两个原语；
  交接内容为 llama.cpp 的序列状态（KV + 位置）；审计 `handoff_bytes` / `handoff_ms`；
  **等价性要求**：交接后在解码上下文继续解码，与在预填充上下文继续解码的 logits 一致
  （max|diff| ≤ 1e-6），否则报错而不是继续。
- **FR-3 流水线调度**：预填充在独立线程上执行，解码线程消费就绪队列；
  队列深度 `pipeline_queue`（默认 2×K）；解码按 AR017 的锁步批推进；
  批满或队列空时刷新；所有阶段的失败都要回传主线程并终止（返回 `Status`）。
- **FR-4 参考路径隔离（修既有 bug）**：批/流水线模式下，dense 参考实现必须使用
  独立的串行上下文（按用例所需 `n_ctx` 定容），不得复用批上下文——
  当前实现在 `decode_batch_size>1` 且未加 `--bench-skip-dense` 时
  会报 `failed to find a memory slot for batch of size 512` 并让后续批 prefill 位置错乱
  （spike A 复现，属**真 bug**，必须在 AR018 修复）。
- **FR-5 审计与闸门**：
  审计列 `pipeline_mode` / `pipeline_stage` / `prefill_device` / `decode_device` /
  `handoff_bytes` / `handoff_ms` / `queue_wait_ms` / `overlap_ratio`；
  闸门至少包含：质量控制、等价性、吞吐（vs 串行与 vs AR017）、TTFT/首 token 时延、
  设备放置、流水线真的重叠（`overlap_ratio ≥ 1.3`）、容量；
  注入 `no-overlap`（退化为串行两段）、`drop-handoff`（跳过 KV 拷贝）、
  `wrong-seq`（交接写错序列号）必须 FAIL。
- **FR-6 组合评估与回退**：对 `serial / AR017-batch / pipeline / pipeline+batch`
  四种组合在同一会话内交错测量，选出**综合最优**并写入 portfolio；
  若流水线组合不能优于 AR017 组合（吞吐或单例延迟其中一项显著退化），
  按 AGENTS.md 的"组合不劣"原则回退，仅作为可选开关保留并记录负结果。

### 2.2 范围外

- 不实现跨进程/跨机器 KV 迁移（DistServe 的 full disaggregation）；
- 不实现连续批（批内中途准入）——仍是 AR017 的锁步批；
- 不改 llama.cpp patch；只使用上游 `llama_state_seq_*` 接口；
- 不在 pipeline 模式支持快照、会话、FFN skip 组合实验（后续 AR 再评估）。

## 3. 验收标准

| 编号 | 标准 | 度量 |
|---|---|---|
| AC-1 | 流水线与串行逐例答案相同 | `generated_hash` 相同（允许 AR014 浮点口径的极少数 1 token 翻转） |
| AC-2 | KV 交接精确 | 交接后 logits max\|diff\| ≤ 1e-6 |
| AC-3 | 吞吐提升 | `cases/s` ≥ 1.5× 串行；≥ 1.2× AR017 最优 K |
| AC-4 | 单例延迟不劣化 | 单例 E2E ≤ 1.35× AR017 最优 K |
| AC-5 | 真重叠 | `overlap_ratio = (Σstage)/wall ≥ 1.3` |
| AC-6 | 放置可验证 | `prefill_device_state` / `decode_device_state` 与请求一致 |
| AC-7 | 无静默降级 | 任一 FR 失败都有审计证据 + 显式回退标记 |
| AC-8 | 回归 | 无模型/真机全量测试通过，`-Werror` 零告警，AR012–AR017 闸门不退化 |

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 两个模型实例内存翻倍 | 端侧 1 GB RAM + 1 GB VRAM | 记录 RSS/VRAM；模型 mmap 共享页；`pipeline_mode=0` 时完全不创建 |
| GPU 在本机处 P8，prefill 只快 1.6× | 收益上限受限 | 如实报告；换高频 GPU 自动受益（AR016 plan） |
| 跨设备 KV 交接依赖 llama.cpp 内部布局 | 版本升级可能失效 | 交接后做 logits 等价断言（FR-2），失败即报错 |
| 线程间竞争让解码变慢 | 重叠收益被吃掉 | 队列深度与线程数可配；闸门用 `overlap_ratio` 量化，不达标就回退 |
| 主机共租导致噪声 | 单点不可信 | 同会话交错 A/B、每臂取最小（AR017 口径） |
