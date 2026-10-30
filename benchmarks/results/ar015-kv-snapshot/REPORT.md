# AR015 验收报告：持久化 KV 快照 + 跨进程热启动

| 字段 | 内容 |
|------|------|
| AR | AR015 |
| 日期 | 2026-10-30 |
| 模型 | Qwen2.5-1.5B-Instruct Q4_K_M，sha256 `6a1a2eb6…9407e` |
| 主机 | Xeon Gold 6234（8C/16T），WSL2，8 线程；GPU 段：RTX 5000 16 GB / CUDA 12.0，全层卸载 |
| 命令 | `./repro.sh ar015`（内部：`ar015_matrix.py` + `ar015_report.py --gate`） |
| 结论 | warm TTFT 为 cold 的 **0.027×–0.046×**（8 数据集，22–36×）；质量**逐例完全相同**；快照 = 已用 KV + 0.6 MB logits；多轮重启续跑 40/40 命中（0.10×） |

## 1. 问题

AR014 的复用只在进程内。端侧进程被回收是常态（重问、冷启动、Agent 重试），而这些请求面对的
往往是同一份准入证据。AR013 之后这份证据的 KV 已降到 15–34 MB，完全放得进磁盘：

| 观测（本机） | 数值 |
|---|---:|
| 单次 TTFT（4K / 32K / 64K） | 3292 / 7422 / 5074 ms |
| 已用 KV | 19.0 / 34.0 / 34.0 MB |
| 快照装载（读 + 校验 + 恢复） | 114 / 198 / 186 ms |

## 2. 机制

| 机制 | 开关 | 作用 |
|---|---|---|
| 内容寻址 key | 默认（`snapshot_dir` 非空时） | SHA(版本｜模型指纹｜准入策略指纹｜KV 精度｜n_ctx｜准入 token 哈希｜位置哈希) |
| 快照 payload | 同上 | 序列状态（`llama_state_seq_get_data`）+ **末位 prompt token 的 f32 logits** |
| 三重校验 | 同上 | manifest 逐字段相等 + 状态/ logits 哈希一致 + token 数一致；任一不符即 `miss:<reason>` 并回退重算 |
| 原子写 + LRU | `snapshot_max_bytes`（默认 512 MiB） | `.kv`/`.lg` 先写、`.meta` 最后写；按体积整组淘汰 |
| 会话落盘 | 与 AR014 组合 | 每轮一个快照；新进程重放会话时逐轮命中 |

**为什么存 logits**：最初只存 KV，warm 路径重新解码最后一个 token 取 logits。实测该做法
与"整段 prefill 的末位 logits"最大绝对差 **0.79–0.90**（批次形状不同 → ggml tiling 不同），
且单 token 解码约 300 ms，比整个装载（114–198 ms）还贵。改存 logits 后 warm 路径**零计算**，
且首 token 逐字节相同。

## 3. 结果（单轮，真实 GGUF，CPU 8 线程）

| 数据集 | cold TTFT（min） | warm TTFT（min） | 比值 | 加速 | 快照 | q_mean |
|---|---:|---:|---:|---:|---:|---:|
| explen/len4k | 3292 ms | **117 ms** | 0.036× | 28× | 19.0 MB | 1.000 → 1.000 |
| explen/len8k | 6142 ms | **190 ms** | 0.031× | 32× | 32.6 MB | 1.000 → 1.000 |
| explen/len16k | 5979 ms | **195 ms** | 0.033× | 31× | 33.0 MB | 1.000 → 1.000 |
| deep/len32k | 7422 ms | **203 ms** | 0.027× | 36× | 34.0 MB | 1.000 → 1.000 |
| deep/multineedle | 6174 ms | **188 ms** | 0.030× | 33× | 32.6 MB | 1.000 → 1.000 |
| deep/multifact8k | 2356 ms | **93 ms** | 0.040× | 25× | 15.2 MB | 0.975 → 0.975 |
| deep/multifact8k-hard | 2073 ms | **95 ms** | 0.046× | 22× | 15.3 MB | 0.746 → 0.746 |
| explen/len64k | 5074 ms | **191 ms** | 0.038× | 27× | 34.0 MB | 1.000 → 1.000 |

快照体积 = `kv_used_bytes + logits`（0.6 MB），逐数据集比 1.009–1.032，闸门阈值 1.30。
质量列是**逐例 f1 完全相同**（不是均值接近：`cases.csv` 逐 case 差 0）。

## 4. 与 AR014 组合：跨进程续跑

`deep/multiturn`（8 会话 × 5 轮）跑三臂，同一份快照目录：

| 臂 | 含义 | q_mean | TTFT（min） | 命中 |
|---|---|---:|---:|---:|
| cold | AR014 进程内复用，不落快照 | 0.861 | 3103 ms | — |
| warm | AR014 + 快照 | 0.861 | **313 ms**（0.101×） | 40/40 |
| resume | `session_reuse=0`（每轮都像"重启"）+ 快照 | 0.861 | **302 ms**（0.097×） | 40/40 |

- 三臂 `quality` 逐例相同，`session_log_hash` 逐轮相同；
- `resume` 模拟"进程每轮都死一次"的最坏情形，仍然只付装载成本；
- 单轮与多轮的差别正好来自快照体积：单轮 15–34 MB（93–203 ms），多轮每轮 ~47 MB（~300 ms）。

> **AR016 勘误**：本节数字是在 CPU 上测的。当时 bench 只认 conf 里的
> `n_gpu_layers`，CLI 的 `--n-gpu-layers 99` 被丢弃，llama.cpp 实际打印
> `offloaded 0/29`。原始 CSV 保留为证据，完整说明见
> `benchmarks/results/ar015-kv-snapshot-gpu/ERRATUM.md`；真实 GPU 数据见
> `benchmarks/results/ar016-exec-plan/`。

## 4.5 GPU（RTX 5000 16 GB，CUDA 12.0，全层卸载）

`build-cuda/edge-prefill`，同两个数据集，`--gpu-layers 99`：

| 数据集 | cold TTFT | warm TTFT | 比值 | 命中 | q_mean |
|---|---:|---:|---:|---:|---:|
| explen/len4k | 3688 ms | **116 ms** | 0.031× | 4/4 | 1.000 → 1.000 |
| deep/multifact8k | 2671 ms | **95 ms** | 0.035× | 16/16 | 0.975 → 0.975 |

- 快照体积、装载耗时与 CPU 基本一致（112–94 ms）：**装载是磁盘 + H2D 拷贝，与设备算力无关**，
  这正是它比"重算"更稳的原因；
- 闸门在 GPU 结果上同样全 PASS（`ar015_report.py benchmarks/results/ar015-kv-snapshot-gpu --gate`）；
- 值得记录的是：小模型 + 稀疏 gather 下 **GPU 的 cold TTFT 并不比 CPU 快**（4K 3688 vs 3292 ms），
  快照把两者的首 token 延迟拉平到同一量级（~100 ms），这对端侧"混合有 GPU 也可能没有"的
  部署形态更重要。

## 5. 闸门（22 项全 PASS）

- hit：每个 warm case 都命中（没有静默回退）；
- quality：逐例 f1 相同；
- latency：逐数据集 warm ≤0.5× cold（实测 0.027–0.046×）；
- size：快照 ≤1.30× `kv_used_bytes`（实测 1.009–1.032）；
- resume / resume-latency：多轮三臂质量与哈希一致，续跑 ≤0.5×（实测 0.097×）。

注入劣化必失败（已实测）：
`--inject explen/len4k:no-hit` → hit FAIL；`--inject explen/len4k:quality-drift` → quality FAIL。

## 6. 单元与集成测试

`tests/test_kv_snapshot.cpp`（5 例）：

- `snapshot_restores_sequence_and_logits`（真机）：含**位置空隙**的 128-token 序列，冷路径
  prefill + 存状态/ logits，warm 路径装载后继续生成 4 个 token，**与冷路径逐 token 相同**；
- `snapshot_misses_are_explicit`（真机）：缺失 / token 不符 / payload 篡改 → 三种 miss 原因；
- `snapshot_key_is_stable_and_sensitive`：key 稳定；模型/策略/KV 精度/n_ctx/token/位置
  **任一变则 key 必变**；
- `snapshot_manifest_roundtrip_and_rejects`：manifest 往返 + 无版本/坏行/坏数字全部拒绝；
- `snapshot_store_evicts_oldest_and_ignores_tmp`：超限淘汰最旧整组、忽略 `.tmp`、保留未写完的 key。

## 7. 实测踩坑（均已修复）

| 现象 | 根因 | 修复 |
|---|---|---|
| 会话 warm 命中 0/40 | 默认 512 MiB 上限下，40 轮 × 47 MB 让 LRU 把早轮次淘汰 | 矩阵按会话体量配置上限；文档写明默认是 LRU 策略 |
| 会话 warm TTFT 报 0 ms | 装载发生在计时区之前 | 命中时 TTFT 取装载耗时（含读+校验） |
| `selected_tokens` 在 warm 臂异常大 | key 用整段 prompt 而非准入 token | key 与 `save/load` 都改用准入 token 序列 |
| 快照装载后 logits 与 cold 差 0.79–0.90 | 重新解码末 token 的批次形状不同 | 直接存 logits |

## 8. 口径与局限

- **页缓存**：本机 warm 装载 114–198 ms 含"刚写过 + 页缓存热"的有利条件；冷盘 200 MB/s 估算
  100–200 ms，收益仍是 20×+。矩阵未做 drop_caches（WSL 上不可靠），因此报告只宣称
  "读 + 校验 + 恢复"这条路径的实测值；
- 快照只对**完全相同的准入 token 序列与位置**有效，不做近似拼接；
- 快照不跨 KV 精度 / n_ctx / llama.cpp 版本共享（key 已隔离）；
- 多轮每轮一个快照，磁盘随轮数线性增长，靠 `snapshot_max_bytes` 的 LRU 兜底。

## 9. 复现

```bash
./repro.sh ar015          # 多轮三臂 + len4k/len64k cold/warm + 闸门 + 两种注入
python3 benchmarks/scripts/ar015_matrix.py --datasets deep/multiturn,explen/len4k,explen/len64k --force
python3 benchmarks/scripts/ar015_report.py --gate
# GPU
python3 benchmarks/scripts/ar015_matrix.py --bin build-cuda/edge-prefill \
    --datasets explen/len4k,deep/multifact8k --gpu-layers 99 \
    --out benchmarks/results/ar015-kv-snapshot-gpu
```

原始数据：本目录 `<dataset>-<arm>/{bench.conf,*_raw.csv}`、`matrix.csv`、`cases.csv`；
快照文件不入库（可再生，`snapshots/` 每次 `--force` 重建）。
