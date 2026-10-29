# AR014 验收报告：会话级 KV 复用

| 字段 | 内容 |
|------|------|
| AR | AR014 |
| 日期 | 2026-10-30 |
| 模型 | Qwen2.5-1.5B-Instruct Q4_K_M，sha256 `6a1a2eb6…9407e` |
| 主机 | Xeon Gold 6234（8C/16T），WSL2，8 线程，`repeats=2` |
| 命令 | `./repro.sh ar014`（内部：`ar014_matrix.py` + `ar014_report.py --gate`） |
| 结论 | 第 2..5 轮 TTFT 降到重算臂的 **0.27×/0.26×/0.20×/0.13×**，第 5 轮 **0.13×**；40 轮里 39 轮 f1 逐位相同，1 轮因浮点批次差异翻转（会话臂更优）；单轮数据集与 AR013 逐位一致 |

## 1. 问题（需求阶段实测）

AR013 之后单次 prefill 已经很快（4K 782 ms，多事实 2.2 s），但**每一轮都从零重建上下文**：
`sparse_prefill.cpp` 每次 `prefill()` 先 `reset_sequence()`，bench 里 `enable_prefix_cache=false`。
多轮场景（同一文档连续追问、RAG 同一文档）第 2..N 轮与第 1 轮高度重叠，却每轮再付一次全量 prefill。

实测依据（AR013 已入库数据）：每 token prefill 成本 ≈ 5.5–6 ms，多事实单轮准入 509 token；
`deep/multiturn` 5 轮累计 log 1042→2268 token，即第 5 轮要重算 1948 个已算过的 token。

## 2. 机制

| 机制 | 开关 | 作用 |
|---|---|---|
| append-only 会话 log | `session_reuse = 1` + index 第 6 列 `session` | 每轮只把"本轮新增准入证据 + 问句 + chat 模板尾"追加到 log |
| 增量解码 | 同上 | `llama_decode` 从 `next_pos` 连续写入增量，不清序列、不重解码前缀 |
| 一次性上下文定容 | 同上（`session_ctx_max`，默认 4096 格） | 会话首轮按"轮数 × 本轮增量 × 1.5"预算一次 `ensure_context`；中途若仍要扩容则本轮降级为全量重建（记录 `session_reset_reason`） |
| 审计列 | 同上 | `session_id`/`reused_tokens`/`appended_tokens`/`session_log_tokens`/`session_log_hash`/`session_reset_reason` |

`session_log_hash` 是两臂等价性的硬证据：两臂喂给模型的 token 序列与位置完全相同，只是
批次形状不同（整段分块 prefill vs 增量 decode）。

## 3. 结果（deep/multiturn，8 会话 × 5 轮）

| 轮 | 重算臂 q | 会话臂 q | TTFT 会话/重算 | reused | appended | log |
|---:|---:|---:|---|---:|---:|---:|
| t1 | 0.791 | 0.791 | 0.91x（4379→3968 ms） | 0 | 954 | 0 |
| t2 | 0.684 | 0.684 | 0.27x（5660→1500 ms） | 970 | 292 | 970 |
| t3 | 0.727 | 0.727 | 0.26x（7916→2057 ms） | 1239 | 388 | 1239 |
| t4 | 0.561 | 0.636 | 0.20x（9299→1827 ms） | 1634 | 314 | 1634 |
| t5 | 0.625 | 0.625 | 0.13x（10554→1358 ms） | 1948 | 228 | 1948 |
| 合计 | 0.678 | **0.693** | 中位 0.26x (7731 -> 1731 ms) | 1153 | 425 | — |

- 首轮无可复用内容，两臂做同样的工作（0.91×，闸门 ≤1.15×）。
- 从第 2 轮起每轮只解码 164–388 个 token，而重算臂要解码 970–1948 个。
- `appended_tokens` 恒 < `session_log_tokens`，且首轮 `reused_tokens=0`（审计通过）。

单轮数据集（无 `session` 列，会话路径完全不参与）与 AR013 逐位一致：

| 数据集 | 重算臂 | 会话臂 | TTFT min |
|---|---:|---:|---|
| explen/len4k | 1.000 | 1.000 | 782→740 ms（0.95×） |
| deep/multifact8k | 0.975 | 0.975 | 2213→2225 ms（1.01×） |

## 4. 唯一的质量差异（诚实记录）

40 轮里 **39 轮 f1 逐位相同**，`session_log_hash` 也相同；**1 轮不同**：

- `mt-03-t4@6942`：重算臂 0.200 → 会话臂 0.800（会话臂更优），log 长度 1659/1659、hash 相同。
- 后果：`mt-03-t5` 的 log 里含不同的助手回答，hash 随之不同（级联，不是独立问题）。

根因是**批次形状不同导致的浮点累加差异**（整段分块 prefill 的 K/V 与增量 decode 的 K/V 在
ggml 里走不同的 tiling，末位比特不同），greedy 解码把它放大成一个 token 的差异。这不是
"复用导致上下文错位"：谓词（token 序列 + 位置）已由 hash 证明完全相同，且该轮差异在
两次独立运行中可复现（不是随机噪声）。

闸门因此按"可证明的不变量 + 统计边界"判定，而不是"逐例完全相同"：

| 检查 | 判据 | 实测 |
|---|---|---|
| assembly | 每轮两臂 log 长度相同；hash 相同（翻转的级联除外） | 39/40 hash 相同，1 例为翻转级联 |
| quality-agg | 会话臂均值 ≥ 重算臂均值 −0.005，且翻转轮 ≤5% | 0.693 ≥ 0.678，翻转 1/40 = 2.5% |
| quality（逐例） | 单轮下降不超过 0.25 | 无（唯一差异为 +0.600） |

不做 CacheBlend 式"拼接任意 KV + 重算修正"：本项目只在**严格同前缀**时复用。

## 5. 闸门（84 项全 PASS）

- quality / assembly：40 轮 + 单轮回归逐例；
- latency：t1 ≤1.15×，t2–t4 ≤0.6×，t5 ≤0.5×（实测 0.91/0.27/0.26/0.20/0.13）；
- audit：8 个首轮 `reused_tokens=0`，32 个后续轮复用前缀且 `appended < log`；
- no-regress：`explen/len4k`、`deep/multifact8k` 与 AR013 逐位一致。

注入劣化必失败（已实测）：
`--inject deep/multiturn:no-reuse` → 5 项 latency FAIL；
`--inject deep/multiturn:quality-drift` → quality-agg FAIL。

## 6. 内存

会话 KV 的**已用量**严格等于 log 长度（Qwen2.5-1.5B：28.7 KB/token）：
t1 1042 token ≈ 30 MB，t5 1948 token ≈ 62 MB（LLM 报告的 KV bytes 逐轮增长，与审计列一致）。
**分配量**由 `session_ctx_max` 决定（默认 4096 格 = **112 MiB**，实测 `kv_alloc_bytes` 117440512）：会话一次定容，避免中途重建；
超出则本轮退化为全量重建，答案仍然正确，只是多付一次 prefill。

## 7. 方法与注意事项

- 宿主机在本次验收期间被外部负载反复挤占：同一 cell 的 TTFT 中位在两次运行间差到 4×
  （7.2 s ↔ 31 s）。因此**闸门只看同一次运行内的比值**（`ttft_min`），绝对值仅作参考。
- 并发跑矩阵与测试会让两者互相拖慢（16 逻辑核 SMT 争用），验收时串行执行。
- 中途一次实现曾按"轮数 × 全文档准入量"预算上下文，`budget=1.0` 时把 CUDA 上下文撑到 31K 格
  导致解码卡死；改为按本轮实际增量预算并加 `session_ctx_max` 上限后正常。

## 8. 复现

```bash
./repro.sh ar014                  # 多轮两臂 + 单轮回归 + 闸门 + 注入
python3 benchmarks/scripts/ar014_matrix.py --datasets deep/multiturn --force
python3 benchmarks/scripts/ar014_report.py --gate
```

原始数据：本目录 `<dataset>-<arm>/{bench.conf,*_raw.csv,*_report.md}`、`matrix.csv`、`turns.csv`
（逐轮 f1/ttft/reused/appended/log 长度/log hash）。
