# [AR019] 任务清单

| ID | 任务 | 依赖 | 状态 | 验收要点 |
|---|---|---|---|---|
| T001 | 配置 `continuous_batch`（0/1）+ 审计列 `continuous/slot/admit_step/own_steps/peer_steps` | - | passing | 0 时行为逐位不变（AR017/AR018 闸门原样通过） |
| T002 | 连续调度器主体：准入/单步解码/退休/槽位复用，接线 AR018 预填充 worker | T001 | passing | 16 用例 / K=4 时准入次数 ≥ 13 |
| T003 | 逐例记账：`e2e = 自身预填充 + 自身解码`，保留 `batch_wall_ms` 与 `peer_steps` | T002 | passing | 早完成用例 E2E 明显小于最慢用例 |
| T004 | 失败路径：预填充/交接失败只影响该用例并释放槽位 | T002 | passing | 其余用例仍完成 |
| T005 | 真机测试：`continuous_matches_serial`（哈希逐例相同）+ 调度性质（准入次数、不陪跑） | T003 | passing | 先写测试 |
| T006 | `ar019_matrix.py` / `ar019_report.py`：四组合交错 + 6 项闸门 + 3 注入 | T005 | passing | 注入必须 FAIL |
| T007 | 组合决策：按 AC-1..AC-6 决定是否把连续批设为默认 | T006 | passing | 达标才升级，否则回退并记录 |
| T008 | 文档：README / research §21 / architecture / portfolio / 组件详设 / repro.sh ar019 | T007 | passing | 与实测一致 |
| T009 | ST 验收 + 归档 + tag `ar019-v1.0` + 分片推送 + 干净克隆复现 | T008 | passing | 克隆可构建/测试/过闸门 |

## ST（系统测试）验收

### ST-1 单元/集成（真机）

- `continuous_matches_serial`：3 用例，连续批与静态批 `generated_hash` 逐例相同，
  审计 `continuous=1`、`own_steps>0`、`slot>=0`；PASS。
- 回归：`pipeline_matches_serial`、`seq_state_handoff_is_lossless`、AR017 三例均 PASS；
  `build-werror` 零告警。

### ST-2 闸门

`python3 benchmarks/scripts/ar019_report.py --gate`：answers-identical、quality、
throughput（K=4 1.03×、K=8 1.08×，阈值 0.98× 防回归）、iteration-level-scheduling
（16 次准入 / 4 槽 = 4.0×，own=37 步，peers=4）、latency-recorded 五项全 PASS。

### ST-3 注入自测

`quality-drift`（仅污染连续臂）、`lockstep-peers`（伪造成单序列解码）、`slow`
（把连续批墙钟翻倍）三种注入均 GATE FAILED。

### ST-4 组合决策（回退）

未满足"吞吐不劣化且单例延迟回到 AR017 水平"（K=4 延迟差 29%），按组合不劣原则
**不设为默认**：默认组合保持 AR013+AR016+AR017(K=2)，AR019 为可选调度开关（默认 0）。
根因（流水线预填充受限，30 s 预填充 vs 8–14 s 解码）写入 portfolio 与 research §21。
