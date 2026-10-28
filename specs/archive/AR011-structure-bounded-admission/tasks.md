# [AR011] 任务跟踪

  字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-27 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 选择器审计基线：记录 AR010 在 7 数据集上的 selected/structure/evidence 构成 | - | passing | 4K 310 token/9 锚点；mfh 422/13 锚点（见 srs §1） |
| T002 | FR-1：`structure_budget` 绝对锚点预算 + `structure_tokens`/`evidence_tokens` 审计 | T001 | passing | 0 = AR010；**并且只在 `anchor_tokens > target/2` 时介入**（16K 强制介入会 280→320 token） |
| T003 | FR-2：`coverage_reserve` 每事实保底 + `uncovered_facts` 审计 | T002 | passing | 实现完成；实测无稳定收益（b=0.03 +0.019、b=0.02 −0.007、b=0.04 0）→ **默认关**，只留审计 |
| T004 | FR-3：`span_admission` 子块准入（**句子级**、跨块合并、锚点整块）+ `span_tokens` 审计 | T002 | passing | 盲目 ±12 窗口实测 0.625（切掉答案）→ 改为句子边界；多事实禁用 |
| T005 | 配置/CLI/CSV 接线：5 个开关 + 4 个审计列 | T002,T003,T004 | passing | 顺带修复 `run` 路径未传 phrase/coverage/prior/fill 的遗留 bug |
| T006 | 单测：锚点有界 / 非主导时不介入 / span 裁剪并保留末 token / 多事实不裁 / 覆盖保底审计 | T005 | passing | 5 例；无模型全量 163 PASS |
| T007 | FR-5：`Selection::to_json` 审计字段补齐 | T005 | passing | 新增 8 个字段（含 AR008–AR010 缺失项） |
| T008 | 前提实验归档：lexical vs embedding 负结果（0 质量收益 / +1.0–2.2s 扫描） | - | passing | srs §1，写入 research §13 |
| T009 | `ar011_matrix.py` + `ar011_report.py`（不劣于 AR010 的质量/token/TTFT 闸门 + `--inject`） | T006 | passing | 7/7 PASS；`--inject explen/len8k:ar010` 报 token+TTFT 违规 |
| T010 | 质量/性能验收 | T009 | passing | 4K −35% token/−28% TTFT、8K −18%、多针 −23%/−35%、hard q_min 0.087→0.571；16K/32K 持平；全部 q 不降 |
| T011 | GPU（CUDA）一致性 | T009 | passing | sel=[122,121,170,172] 逐例一致；45.2 ms vs CPU 1091.5 ms |
| T012 | 文档：README/research §13/architecture/server-repro §11/组件详设 | T010 | passing | 含 embedding 负结果、radius=1 代价、repro.sh `--llama-dir` 离线构建 |
| T013 | Review：`-Werror` 零警告、无模型全量、真机全量、ASan | T010 | passing | build-werror 零告警；无模型 163 PASS；真机全量 **163 PASS / 0 FAIL**；ASan 选择器用例通过 |
| T014 | ST：`repro.sh ar011`、干净克隆、归档、tag `ar011-v1.0`、Gitee 分片推送 | T013 | passing | `./repro.sh ar011`（离线 `--llama-dir`）跑通；Gitee 克隆复验；见 §ST |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-27 需求阶段

- 建立领域知识：读 `docs/architecture.md`、`include/edge_prefill/{selector,metrics,config}.h`、
  `src/main.cpp` 计时点、AR010 结果矩阵。
- 实测（真实 GGUF）：TTFT ≈ `selected_tokens / 155 token·s⁻¹`；4K NIAH b=0.08 准入 310 token，
  其中 9/10 块为 mandatory；多事实 hard 422 token，13/14 块为 mandatory。
- 实测负结果：embedding 选路在 5 个数据集上质量零收益，扫描 +1.0–2.2 s。
- 结论：剩余空间在"准入 token 数"，尤其是结构锚点吃掉预算与块粒度取整浪费。

## 阶段门控记录

### 2026-10-27 开发与实验

- 实现：`structure_budget`/`anchor_span_radius`/`coverage_reserve`/`span_admission`/`span_min_tokens`
  + `structure_tokens`/`evidence_tokens`/`span_tokens`/`uncovered_facts`（Selection/Sample/CSV/dump）。
- 关键实验结论（全部真实 GGUF，同会话）：
  - 4K 226→146 token（−35%）、8K 228→188（−18%）、多针 321→247（−23%）；
  - 16K/32K 持平（结构不主导预算，自适应不介入；强制介入会 280→320）；
  - 多事实 hard q 0.667→0.709、`q_min` 0.087→**0.571**（token 不变，预算再分配）；
  - embedding 准入：质量零收益、扫描 +1.0–2.2 s → 负结果；
  - span 用于多事实：easy −0.036、hard −0.225、`q_min`→0 → 按"无子查询"硬门控；
  - `anchor_span_radius=1`：32K 461 token（−12%）但多事实 −0.007/−0.040 → 默认保持 ±2。
- 矩阵脚本自身的两个坑（已修，记录以备后人）：`--force` 未清空旧 `*_raw.csv` 导致跨配置均值混合；
  多事实族需要 48 个新 token，否则部分分被系统性低估（0.576 vs 0.836）。

## ST 验收记录（2026-10-28）

| # | 用例 | 结果 |
|---|---|---|
| ST-1 | 4K NIAH：token 下降 ≥15% 且 q=1.000 | PASS（226→146，−35%，q 1.000） |
| ST-2 | 8K NIAH：token 下降且 q=1.000 | PASS（228→188，−18%） |
| ST-3 | 16K/32K：不得回退 | PASS（280/525 不变，q=1.000；结构不主导时自动不介入） |
| ST-4 | 多针：token 下降且 q=1.000 | PASS（321→247，−23%；span 裁掉 36 token） |
| ST-5 | 多事实 easy：质量不降 | PASS（0.836 → 0.836） |
| ST-6 | 多事实 hard：`q_min` 提升 | PASS（0.087 → **0.571**，q_mean 0.667→0.709） |
| ST-7 | 三个新开关全关 = AR010 逐 token | PASS（默认全 0；`selector_structure_budget_is_inert_*` 守护） |
| ST-8 | 组合表 + 不劣于 AR010 闸门 | PASS 7/7（全部路由 ar011） |
| ST-9 | 闸门有效性：注入劣化必须 FAIL | PASS（`--inject explen/len8k:ar010` → token/TTFT 违规） |
| ST-10 | GPU（CUDA）一致性 | PASS（sel 逐例相同，45.2 ms vs 1091.5 ms） |
| ST-11 | 无模型全量 / 真机全量 / -Werror / ASan | PASS（163 / 163 / 0 警告 / 通过） |
| ST-12 | `./repro.sh ar011` 一条命令 | PASS（含离线 `--llama-dir` 构建） |
| ST-13 | Gitee 干净克隆：闸门可跑 | PASS |

### 未达成 / 已回退

- 16K/32K：本 AR 无进一步收益（已到"证据驱动下限"）；32K 仍 525 token；
- `anchor_span_radius=1`：32K 461 token（−12%）但多事实 easy −0.036、hard −0.040 → **默认保持 ±2**；
- `coverage_reserve=1`：无稳定收益 → **默认关**；
- embedding 语义准入：零质量收益 + 1.0–2.2 s 扫描 → **负结果，默认 lexical**；
- span 准入用于多事实：`q_min` → 0 → **硬门控禁用**。
