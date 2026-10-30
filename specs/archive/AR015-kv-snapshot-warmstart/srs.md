# [AR015] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR015 |
| AR 主题 | 持久化 KV 快照（内容寻址）+ 跨进程热启动，把"重复上下文"的首 token 延迟降到磁盘读量级 |
| 关联 SR | 无（延续 AR001 端侧稀疏 prefill 目标；承接 AR013 紧凑 KV、AR014 会话复用） |
| 日期 | 2026-10-30 |
| 状态 | Draft |

## 1. 背景与目标

### 1.1 量化现状（全部来自已入库的真实 GGUF 实测）

| 观测 | 数值 | 来源 |
|---|---:|---|
| 单次 TTFT（准入后 prefill） | 4K **782 ms**、多事实 8K **2213 ms**、32K **2980 ms** | `ar013-compact-context/matrix.csv` |
| 已用 KV（= 序列状态大小） | 500 token ≈ **14.4 MB**、1948 token ≈ **62 MB**（28.7 KB/token） | `ar013`/`ar014` raw CSV |
| KV 分配（AR013 紧凑布局后） | 4K **7 MiB**、32K **21 MiB**、会话定容 **112 MiB** | AR013/AR014 |
| 端到端（E2E）构成 | 多事实 hard：TTFT 2.68 s + 生成 16.9 s（48 token，≈296 ms/token） | `ar013` raw CSV |

即：**AR001–AR014 把"每次重新算一遍上下文"的成本压到了 0.8–3.0 s，但每次进程重启/新请求仍然重算一次**；
而这些"重复上下文"在真实端侧场景里占绝大多数：用户重问、App 冷启动、Agent 重试、同一文档的多个问题。

### 1.2 为什么现在的复用机制覆盖不到

| 机制 | 生命周期 | 覆盖的场景 |
|---|---|---|
| `PrefixCache`（AR002） | 进程内、dense | 未接入 bench；不覆盖准入后的稀疏 log |
| AR014 会话复用 | 进程内、单会话串行 | 同一会话的连续轮次；**进程退出即丢失** |

两者都依赖"进程还活着"。端侧进程被系统回收是常态，因此**跨进程**才有价值。

### 1.3 目标

让"同一份准入证据（同一 token 序列 + 同一位置）"的第二次及以后请求，**不再重算 prefill**，而是从磁盘
把序列 KV 状态直接装载回来；首 token 延迟从 0.8–3.0 s 降到**磁盘读 + 装载**量级，且**质量逐位不变**。

预期量级（按 14.4 MB / 200 MB/s 顺序读估算）：装载 70–150 ms，即 **5–20×** 热启动收益；
若快照落在 tmpfs/页缓存（端侧常见），可到 **20×+**。

## 2. 需求范围

**In Scope：**
- `include/edge_prefill/kv_snapshot.h` + `src/kv_snapshot.cpp`：内容寻址的序列状态快照（key = 模型指纹 + 准入策略指纹 + log token 哈希 + 位置信息 + 上下文/KV 精度），save/load/校验/淘汰
- `src/backend.cpp`：`save_sequence_state` / `load_sequence_state`（封装 `llama_state_seq_save_file` / `load_file`，含错误码）
- `src/benchmark.cpp`：`snapshot_dir` 配置；每个 case 先算准入 log（确定性），再用 log 的 key 尝试装载；未命中则照常 prefill 并落盘
- 审计列：`snapshot_hit`、`snapshot_bytes`、`snapshot_load_ms`、`snapshot_save_ms`、`snapshot_key`
- 与 AR014 会话的组合：每轮可落盘快照，支持"进程重启后从上一轮快照续跑"
- 矩阵/闸门脚本、文档、`repro.sh ar015`

**Out of Scope：**
- 跨模型/跨 KV 精度的快照共享（key 里带指纹，天然隔离）
- 分布式/多机 KV 传输
- 任意前缀拼接（只做**完全相同的 token 序列 + 位置**，不做 CacheBlend 式近似复用）
- 修改 llama.cpp 上游

## 3. 功能需求

### 3.1 FR-1 内容寻址的序列快照

**描述**：以"决定模型输入的全部要素"为 key，把一个序列的 KV 状态存成文件，可在新进程装载。

**触发条件**：conf `snapshot_dir` 非空（bench 模式）。

**期望行为**：
- key 由以下要素构成并经 SHA256 收敛为文件名：
  模型指纹（含 GGUF sha256、n_layer/n_head/head_dim）、KV 精度、上下文格数、
  准入策略指纹（所有影响准入的配置项 + 选择器版本）、**log 的 token 序列哈希 + 位置哈希**；
- 落盘文件：`<snapshot_dir>/<key>.kv`（二进制状态）+ `<key>.json`（manifest：token 数、位置跨度、
  KV 字节、模型指纹、策略指纹、生成时间、SHA256）；
- 装载前校验：manifest 与当前请求的 key/长度/指纹一致，且落盘文件 SHA256 与 manifest 一致；
  任一不符即**视为未命中**（不静默使用）。

**异常处理**：目录不可写 → 只警告不失败；装载失败/校验失败 → 回退到完整 prefill（并记录原因）；
文件损坏 → 丢弃并覆盖写。

**验收标准**：
- Given 一个已落盘的快照，When 新进程用同一 case 运行，Then `snapshot_hit=1` 且 `snapshot_load_ms > 0`；
- Given 篡改过的 `.kv`，When 装载，Then 判定未命中并回退（记录 `snapshot_reset_reason`）；
- Given `snapshot_dir` 为空，When 运行，Then 行为与 AR014 逐位一致。

### 3.2 FR-2 质量逐位不变

**描述**：装载快照后继续生成的结果，必须与"重新 prefill 同一 log"完全一致。

**验收标准**：
- Given 同一 case，When 分别用 cold（重算）与 warm（装载）跑，Then 生成文本逐字节相同、`quality` 相同；
- Given 装载后的上下文，When 比较首个生成 token 的 logits，Then 与 cold 路径的最大绝对差 ≤ 1e-6。

### 3.3 FR-3 热启动延迟收益

**描述**：warm 路径的 TTFT 应显著低于 cold。

**验收标准**：
- Given 8 个数据集，When 比较 cold/warm 的 `ttft_min`，Then warm ≤ 0.5× cold（逐数据集）；
- Given 端到端（E2E），When 只看 prefill 部分，Then warm 的 prefill 时间 ≤ 0.25× cold；
- Given GPU（CUDA，全层卸载），When 重复上述测量，Then 同样满足（装载走 `ON_DEVICE` 状态或等价路径）。

### 3.4 FR-4 与 AR014 会话的组合：跨进程续跑

**描述**：会话的每一轮在落盘快照后，进程重启可以从"上一轮末尾"继续，而不是从第 1 轮重算。

**期望行为**：
- 会话臂在每轮结束后按该轮 log 落盘；
- 提供"重启续跑"臂：新进程按会话最后一轮快照装载，然后只解码下一轮增量；
- 若快照的 log 前缀与当前会话期望不一致（例如准入策略变更），降级为重建（记录原因）。

**验收标准**：
- Given 8 会话 × 5 轮，When 对比"重启重算" / "重启续跑" / "进程内复用"，Then 三者 quality 逐例一致，
  且"重启续跑"的第 2–5 轮 TTFT ≤ 0.5× "重启重算"。

### 3.5 FR-5 组合闸门与回退

**描述**：`ar015_report.py --gate` 同时检查命中率、质量等价、热启动延迟、快照体积、会话组合；
任何一项不达标即回退该机制（默认关闭 `snapshot_dir`）。

**验收标准**：
- Given 矩阵结果，When 跑闸门，Then 全 PASS；
- Given 注入 `explen/len4k:no-hit`（强行全部未命中），Then 延迟项 FAIL；
- Given 注入 `explen/len4k:quality-drift`（人为改一个答案），Then 质量项 FAIL；
- Given 快照体积 > 1.2× 已用 KV，Then 体积项 FAIL。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 延迟 | warm TTFT | ≤ 0.5× cold（逐数据集），≤ 0.25× cold（prefill 部分） |
| 质量 | 生成文本 | 逐字节相同；首 token logits 最大差 ≤ 1e-6 |
| 体积 | 单快照 | ≤ 1.2× `kv_used_bytes` |
| 可靠性 | 损坏/不匹配 | 100% 回退到重算，不产生错误答案 |
| 确定性 | 同输入 | key 稳定（同配置两次运行 key 相同） |
| 兼容 | `snapshot_dir` 为空 | 与 AR014 逐位一致 |

## 5. 约束与假设

**约束：**
- 只复用**完全相同**的 token 序列与位置（不比 CacheBlend 式的近似拼接）；
- 不引入新依赖；C++17；`-Werror`；无跨 API 异常；
- 所有数字来自真实 GGUF，不用 mock；GPU 与 CPU 都要实测。

**假设：**
- 端侧存储顺序读 ≥ 100 MB/s（eMMC/UFS/NVMe 均满足），因此 14 MB 快照装载 < 150 ms；
- 同一文档的重复请求（重问、冷启动、Agent 重试）足够多，命中率有实际意义；
- `llama_state_seq_save_file/load_file` 能完整还原含位置空隙的序列（本 AR 用 bit-exact 测试证明）。

**风险：**
- 序列状态在 llama.cpp 中不保证跨版本兼容 → key 中带 llama.cpp commit 指纹，版本不符即未命中；
- 状态装载后 `n_ctx`、KV 精度必须与保存时一致 → 已在 key 与 manifest 双重校验；
- CUDA 上状态装载可能走 H2D 拷贝，收益取决于带宽 → ST 必须实测，若不达标则 CUDA 只启用 ON_DEVICE 路径或关闭。

## 6. 组合分析（为什么是这个 AR，以及和其它 AR 的关系）

| AR | 与本 AR 的关系 | 结论 |
|---|---|---|
| AR013 紧凑 KV | 快照体积 = 已用 KV = 28.7 KB/token；AR013 前 bench 恒定 896 MiB，快照不可行 | **强协同**：AR013 让 AR015 在磁盘上可行 |
| AR014 会话复用 | 进程内复用；AR015 是它的"跨进程"补齐，key 用同一份 log 哈希 | **强协同**：FR-4 |
| AR007/AR011/AR012 准入 | 准入结果决定 log；AR015 不改准入，只改"谁来算这段 log" | **正交**：质量不变 |
| AR002 PrefixCache | 进程内 dense 前缀缓存 | **被取代**（bench 不使用），保留代码但默认关闭 |
| AR006 图内门控 / AR005 语义打分 | 需要额外模型/图改造 | 无关，不引入 |

**组合增益的判据**：AR015 必须在"AR014 组合"之上让**热启动路径**显著更快（≥2×）、质量不变、
体积可控；若实测不能，则回退为默认关闭（`snapshot_dir` 空），并在报告中记录负结果。

## 7. 术语说明

| 术语 | 定义 |
|------|------|
| 快照 | 一个 llama sequence 的 KV 状态落盘文件（含 token 数/位置跨度） |
| log | 准入后的 token 序列（AR014 起 append-only，含位置） |
| cold / warm | 重算 prefill / 从快照装载 |
| 命中 | key 与 manifest 双重校验通过并成功装载 |
