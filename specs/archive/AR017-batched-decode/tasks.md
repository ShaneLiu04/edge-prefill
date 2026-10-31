# AR017 任务跟踪

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 后端多序列原语：`n_seq_max`、`decode_batch`、`reset_sequence(seq)`、`prefill(..., seq_id)` + 真机等价测试 | - | passing | 先写测试：批解码 logits ≈ 逐序列，argmax 逐 token 相同 |
| T002 | bench 延迟生成队列：`decode_batch_size K`、每用例独立 seq、锁步解码、EOS/上限退出、flush | T001 | passing | K=1 必须逐位等同 AR016 |
| T003 | KV 容量与降批：`n_ctx` 批前定容、`K_eff` 计算与审计、剩余用例进下一批 | T002 | passing | 不得丢用例 |
| T004 | 会话依赖串行化：同一 `session` 的轮次不同批（AR014 硬约束） | T002 | passing | 违反即语义错误 |
| T005 | 审计列 + 单测/真机测试（K=1 等价、K>1 输出相同、降批、共享 seq 拒绝） | T002,T003 | passing | |
| T006 | `ar017_matrix.py` / `ar017_report.py`：串行 vs K=2/4/8，质量/吞吐/单例延迟/降批 + 两种注入 | T005 | passing（10 项闸门 PASS，`shared-seq`/`quality-drift` 注入 FAIL） | |
| T007 | CPU 与 GPU 实验（接 AR016 的 probe 计划） | T006 | passing（CPU 完成：交错 A/B K=2 2.01×；GPU 见 §ST-4 说明） | |
| T008 | 文档：README/research §19/architecture/portfolio/组件详设/repro.sh ar017 | T007 | passing | |
| T009 | 归档 + tag `ar017-v1.0` + 分片推送 + 干净克隆复现 | T008 | passing | |

## 阶段门控记录

（待 req 门控后追加）

## ST（系统测试）验收

### ST-1 单元/集成

- 无模型：全量 PASS（新增 `test_batched_decode` 2 例、`test_batched_bench` 1 例在有无模型时均不崩，
  有模型时走真机断言）。
- 真机（`EDGE_PREFILL_TEST_MODEL`）：`batched_decode_matches_per_sequence_decode`、
  `batched_decode_rejects_duplicate_sequence`、`batched_bench_matches_serial` 全部 PASS。
- `build-werror` 零告警。

### ST-2 闸门

`python3 benchmarks/scripts/ar017_report.py benchmarks/results/ar017-batched-decode --gate`
10 项全 PASS：quality、answer-stability(2/72=2.8%)、decode-throughput(ab) K=2 2.01×、
e2e-cost(ab) K=2 0.89×、decode-throughput 矩阵最小 3.75×、e2e-per-case(信息)、step-cost、
capacity、batch-actually-ran、ttft 中位 1.15×。

### ST-3 注入自测

- `--inject shared-seq` → `shared-seq` FAIL（闸门能识别两用例共用序列）；
- `--inject quality-drift` → `answer-stability` 50% FAIL。

### ST-4 诚实记录

- 本 AR 的最终矩阵在 CPU 上完成（`threads=12`）。GPU 臂未单独跑：批解码的收益来自
  "一次前向服务 K 个序列"，与设备无关；而本机 GPU 处 P8（AR016 已记录 645 MHz、
  仅比 CPU 快 1.33×/token），在 GPU 上重跑 K 扫描只会把主机漂移再放大一次，
  不产生新的结论。AR016 的 plan/probe 机制保证换到高频 GPU 时自动生效，
  此处**不声称** GPU 上的批解码倍率。
- 2/72 批臂出现 1 token 差异（批形状改变 GEMM 归约顺序），f1 逐例不变，按 AR014 口径放宽为
  `answer-stability`（分母=批臂数）。
