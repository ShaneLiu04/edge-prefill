# [AR018] 任务清单

| ID | 任务 | 依赖 | 状态 | 验收要点 |
|---|---|---|---|---|
| T001 | 后端：`get_seq_state(seq)` / `set_seq_state(seq, data)` + 真机等价测试（同上下文往返、跨上下文往返后 logits 一致） | - | passing | 先写测试：往返后 decode 的 logits max\|diff\| ≤ 1e-6 |
| T002 | 双后端骨架：`pipeline_mode` 配置 + 两个 `LlamaBackend` 各自放置/校验/审计，`pipeline_mode=0` 行为逐位不变 | T001 | passing | K=1 串行路径零回归 |
| T003 | 流水线调度器：预填充线程 + 就绪队列 + 解码线程（AR017 锁步批）+ 失败回传 | T002 | passing | 队列满/空、EOS、异常路径都要收敛 |
| T004 | 参考路径隔离：批/流水线模式下 dense 参考使用独立串行上下文（修 spike A 复现的 `no memory slot` bug） | T002 | passing | 不 skip-dense 也能跑通 K>1，且位置不再错乱 |
| T005 | 审计列 + 单测/真机测试：`pipeline_mode/stage/prefill_device/decode_device/handoff_bytes/handoff_ms/queue_wait_ms/overlap_ratio` | T003,T004 | passing | 审计列与实际调度一致 |
| T006 | `ar018_matrix.py` / `ar018_report.py`：四组合（serial / AR017 / pipeline / pipeline+batch）同会话交错 + 7 项闸门 + 3 种注入 | T005 | passing | 注入必须 FAIL |
| T007 | 组合评估与回退：按闸门结果确定 portfolio 最终组合，写清不达标时的回退 | T006 | passing | 组合不劣于 AR017，否则回退并记录 |
| T008 | 文档：README / research §20 / architecture / portfolio / 组件详设 / repro.sh ar018 | T007 | passing | 与实测口径一致 |
| T009 | ST 验收 + 归档 + tag `ar018-v1.0` + 分片推送 + 干净克隆复现 | T008 | passing | 干净克隆可构建/测试/过闸门 |

## ST（系统测试）验收

### ST-1 单元/集成（真机）

- `seq_state_handoff_is_lossless`：跨上下文 KV 交接后 decode logits `max|diff| ≤ 1e-6`；
  恢复进被污染的序列同样精确；截断的状态被拒绝（`Cache` 错误）。
- `pipeline_matches_serial`：2 用例，流水线与串行 `generated_hash` 逐例相同，
  审计 `pipeline_mode=1`、`handoff_bytes>0`。
- 全量真机测试 **204 PASS / 0 FAIL**；`build-werror` 零告警。

### ST-2 闸门

`python3 benchmarks/scripts/ar018_report.py benchmarks/results/ar018-disaggregated --gate`
9 项全 PASS（quality、answer-stability 4.6%、handoff-integrity、throughput-vs-serial 2.09×、
throughput-vs-batch 2.06×、overlap 1.08/1.16、placement、latency-honesty 3.16–3.37×≤4.0×、arms-as-labelled）。

### ST-3 注入自测

`no-overlap` / `drop-handoff` / `device-swap` 三种注入均 GATE FAILED。

### ST-4 组合决策（按"不劣于上一组合"原则）

流水线吞吐 2.6–2.9× 提升，但单例 E2E 3.2–3.4× 变差（排队）。因此：
**默认组合仍是 AR013+AR016+AR017（K=2）**；AR018 作为可选的服务侧吞吐模式记录，
未提升默认组合。
