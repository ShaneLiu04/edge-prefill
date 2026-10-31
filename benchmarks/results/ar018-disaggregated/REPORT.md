# AR018 预填充/解码分离（prefill–decode disaggregation）— 实验报告

日期：2026-10-31
设备：CPU（Xeon Gold 6234，12 线程）+ GPU（Quadro RTX 5000 16 GB，29/29 层，P8 645 MHz）
模型：`qwen2.5-1.5b-instruct-q4_k_m.gguf`（SHA256 `6a1a2eb6…49407e`）
llama.cpp：`3bb63cbcbfb62056c84fe9e521b0f3d756d29c32`
命令：`bash repro.sh ar018`（四臂交错矩阵 + 闸门 + 三种注入）

## 1. 结论

| 指标 | 结果 |
| --- | --- |
| 吞吐 vs 串行（最差数据集） | **2.09×**（8K 多事实 2.85×、8K hard 2.90×） |
| 吞吐 vs AR017 批解码 | **2.06–2.65×** |
| 质量 | **逐例 f1 与串行完全相同**（36 用例） |
| 答案稳定性 | 5/108 臂-用例（4.6%）差 1 个 token（浮点口径，上限 5%） |
| KV 交接 | 2.9–3.5 ms / 16–18 MB，逐例精确 |
| 设备放置 | prefill=gpu、decode=cpu，两侧均经 AR016 放置校验 |
| 阶段重叠 | 1.08×（8K）、1.16×（8K hard） |
| 单例延迟 | 3.16–3.37× AR017（排队代价，见 §6） |
| 闸门 | 9/9 PASS；注入 `no-overlap`、`drop-handoff`、`device-swap` 均 FAIL |

**定位**：这台机器上 GPU 的**预填充比 CPU 快 1.6–1.9×**，而**CPU 的解码比 GPU 便宜 2.3×/序列**
（AR018 SRS §1.1 的实测）。把两段拆到两个上下文、用 KV 状态交接，等于让每个设备只做它擅长的事：
吞吐相对串行 2.1–2.9×，相对 AR017 批解码 2.1–2.7×。

## 2. 四臂对照（同会话交错，每臂取最小墙钟）

### deep/multifact8k（16 用例）

| 臂 | 墙钟 | TTFT | 单例 E2E | 解码墙钟 | f1 | 阶段和/墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 108.2 s | 1893 ms | 4703 ms | 44.3 s | 0.975 | 0.68 |
| AR017 batch (K=4) | 100.4 s | 1868 ms | 3578 ms | 7.0 s | 0.975 | 0.36 |
| gpuall (K=4, 全 GPU) | 118.1 s | 1721 ms | 8797 ms | 29.1 s | 0.975 | 0.47 |
| **pipeline (GPU prefill ‖ CPU decode)** | **38.0 s** | 1938 ms | 12056 ms | 10.2 s | 0.975 | **1.08** |

### deep/multifact8k-hard（16 用例）

| 臂 | 墙钟 | TTFT | 单例 E2E | 解码墙钟 | f1 | 阶段和/墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serial | 113.7 s | 1786 ms | 4933 ms | 49.7 s | 0.746 | 0.69 |
| AR017 batch | 102.1 s | 1796 ms | 4007 ms | 8.8 s | 0.746 | 0.37 |
| gpuall | 126.7 s | 1648 ms | 10744 ms | 36.3 s | 0.746 | 0.49 |
| **pipeline** | **39.3 s** | 1994 ms | 12651 ms | 13.6 s | 0.746 | **1.16** |

### explen/len4k（4 用例）

| 臂 | 墙钟 | TTFT | 单例 E2E | f1 | 阶段和/墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial | 32.9 s | 2308 ms | 4930 ms | 1.000 | 0.61 |
| AR017 batch | 32.4 s | 2312 ms | 2891 ms | 1.000 | 0.30 |
| gpuall | 34.5 s | 2219 ms | 4776 ms | 1.000 | 0.32 |
| **pipeline** | **15.7 s** | 2586 ms | 7439 ms | 1.000 | 0.72 |

4 用例时模型加载（两个模型实例）占了墙钟的大头，阶段和/墙钟只有 0.72，所以重叠闸门只对 ≥8 用例的数据集生效。

## 3. 为什么快：三个来源，分开记账

1. **设备专长**：GPU 预填充 1.6–1.9×（`gpuall` 的 TTFT 1648–1721 ms vs 串行 1786–1893 ms）；
2. **批解码**（AR017 复用）：解码墙钟 44–50 s → 7.0–8.8 s；
3. **阶段重叠**：阶段和/墙钟 1.08–1.16（8K/8K-hard），即解码的 7–14 s 大部分藏在预填充后面。

只用 1 和 2 解释不了 2.6–2.9×：`gpuall` 有同样的 GPU 预填充与批解码，却因为 GPU 解码贵 2.3×/序列而**比串行还慢**（118.1 s vs 108.2 s）。这正是"分工"的证据。

## 4. KV 交接的代价与正确性

- 每次交接 **16.2 MB / 2.9 ms**（8K 多事实，span ≈ 500–600 token × 28 层 × 2 × f16）；
- 交接后 `set_seq_state` 前先 `seq_rm`，避免上一用例的残留格子；
- 单元测试 `seq_state_handoff_is_lossless`：跨上下文恢复后 decode 的 logits `max|diff| ≤ 1e-6`，
  包括"恢复到一个被别的文本污染的序列"这种情形；
- `pipeline_matches_serial`：端到端 `generated_hash` 与串行逐例相同。

## 5. 闸门（9 项）与注入

| 检查 | 阈值 | 实测 | 结果 |
| --- | --- | --- | --- |
| quality | 逐例 f1 相同 | 36 用例一致 | PASS |
| answer-stability | ≤5% 臂-用例 | 5/108 = 4.6% | PASS |
| handoff-integrity | bytes>0 且 <100 ms | 2.9–3.5 ms | PASS |
| throughput-vs-serial | ≥1.5× | 2.09× | PASS |
| throughput-vs-batch | ≥1.1× | 2.06× | PASS |
| overlap | ≥1.02（≥8 用例） | 1.08 / 1.16 | PASS |
| placement | prefill=gpu, decode=cpu | 一致 | PASS |
| latency-honesty | ≤4.0× AR017 | 3.16–3.37× | PASS（见 §6） |
| arms-as-labelled | serial=1、batch>1 | 一致 | PASS |

注入：`no-overlap`（把重叠比拉平）、`drop-handoff`（交接字节清零）、`device-swap`（把 prefill 标成 cpu）→ 三种都 FAIL。

## 6. 诚实记录：这是吞吐模式，不是延迟模式

- 单例 E2E 3.16–3.37× AR017。原因是**排队**：解码槽位按 K=4 成批发放，
  一个用例平均要等 1.5 批（`queue_wait_ms` 7.0–7.5 s，占 E2E 的 55–60%）。
- 因此按 AGENTS.md 的"组合不劣"原则，**不把 pipeline 提升为默认组合**：
  portfolio 仍以 AR017（`decode_batch_size=2`）为推荐点，pipeline 作为服务侧吞吐模式（`pipeline_mode=1`）。
- 若要同时压延迟，需要连续批（批内动态准入 + 变小批），那是下一步，不在本 AR 范围。
- 首次矩阵曾出现 8 个用例 f1 掉到 0：根因是队列比一批长时**尾部没被 drain**（只 flush 一次），
  样本停在占位值。已修为 `while (!batch_jobs.empty()) flush_batch(batch_k)`，并在 REPORT 保留这段记录。
- 实施中还踩到三个必然的坑（都已修、都已写进 design.md）：
  两侧 KV **流数必须一致**（否则 `n_stream mismatch`）、
  预填充上下文要按 `required×2×K` 定容（否则 `failed to find a memory slot`）、
  写线程里的异常与 RAII join（否则 `std::terminate`）。

## 7. 复现

```bash
export EDGE_PREFILL_TEST_MODEL=/path/to/qwen2.5-1.5b-instruct-q4_k_m.gguf
bash repro.sh ar018
```

原始数据：`benchmarks/results/ar018-disaggregated/{matrix.csv,_conf/,<dataset>-<arm>-r<N>/}`。
