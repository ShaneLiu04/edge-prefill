# [AR012] 任务跟踪

  字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-28 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 前提审计：多事实预算扫描（排除预算假设）+ `--dump-selected-text` 定位根因 | - | passing | b=0.16 反而 0.771；口令被块边界切断（TEAL-LYNX-8 / 853） |
| T002 | FR-1：句子映射 + 句子完整准入 + 跨块闭包（`sentence_completion`、`span_max_tokens`） | T001 | passing | 0 = AR011 逐 token；并集/裁剪两种模式 |
| T003 | FR-4：审计 `sentences_completed`/`sentences_extended` + CSV/dump 接线 | T002 | passing | 句子计数按唯一句首统计 |
| T004 | FR-2：`--dump-selected-text` 完善（位置标记、dropped 段） | T001 | passing | 已在前提审计中实现 |
| T005 | 单测：跨块口令补齐、关闭时与 AR011 等价、仅多事实生效、审计与索引不变量 | T002,T003 | passing | 4 例；无模型全量 167 PASS |
| T006 | `ar012_matrix.py` + `ar012_report.py`（"没有免费收益"闸门 + `--inject`）+ `check_facts_admission.py` | T005 | passing | 7/7 PASS；注入 trim 臂 FAIL |
| T007 | 验收：7 数据集不劣于 AR011，多事实 hard 提升 | T006 | passing | easy 0.836→**0.887**、hard 0.709→**0.739**（`q_min` 0.571→**0.621**）、事实完整率 0.760→**0.990**；其余 5 个数据集逐 token 不变 |
| T008 | GPU（CUDA）一致性 | T006 | passing | 16/16 例 sel 完全相同；2762 ms → 816 ms |
| T009 | 文档：README/research §14/architecture/server-repro §12/组件详设 | T007 | passing | 含"加预算负收益""裁剪伤质量"两条负结果 |
| T010 | Review：-Werror、无模型全量、真机全量、ASan | T007 | passing | -Werror 零告警；无模型 167 PASS；真机全量 **167 PASS / 0 FAIL**；ASan 选择器用例通过 |
| T011 | ST：`repro.sh ar012`、归档、tag `ar012-v1.0`、Gitee 分片推送、干净克隆复验 | T010 | passing | `./repro.sh ar012` 全绿；见 §ST |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

### 2026-10-28 需求阶段

- 领域知识：AR011 已收口（单事实 q=1.000，4K 146 token/755 ms）。
- 实测：多事实预算扫描 5 档 → 加预算负收益（0.836→0.771），排除"预算不足"。
- 新增 `--dump-selected-text`，直接看到根因：**口令被块边界切断**，6 个事实常有 2 个半残。
- 结论：AR012 做"句子完整准入"，把原子单位从块改成句子（可跨块补齐）。

## 阶段门控记录

### 2026-10-28 开发与实验

- 机制：证据 token 拉入整句 + 跨块闭包（`sentence_completion`），仅在多事实启用；
  `sentence_trim` 保留但不路由。
- 确定性指标：`check_facts_admission.py`（hard 0.760 → 0.990，easy 0.875 → 1.000）。
- 修复：超限安全网按 mandatory 优先截断 → 证据被丢（hard 0.709 → 0.299），
  改为保护证据块/结构块/句子 token。
- 负结果：加预算（b=0.16 → 0.771）、补齐后裁剪（4K/32K → 0.827）、单事实补齐（+15% token 无收益）。
- 组合：多事实 → ar012，其余 → ar011；"没有免费收益"闸门 7/7。

## ST 验收记录（2026-10-28）

| # | 用例 | 结果 |
|---|---|---|
| ST-1 | 多事实 hard 事实完整率（确定性，无模型打分） | PASS（0.760 → **0.990**，15/16 例全部事实在上下文） |
| ST-2 | 多事实 easy 事实完整率 | PASS（0.875 → **1.000**，16/16） |
| ST-3 | 多事实 easy q_mean | PASS（0.836 → **0.887**） |
| ST-4 | 多事实 hard q_mean / q_min | PASS（0.709 → **0.739**，0.571 → **0.621**） |
| ST-5 | 单事实 5 数据集不回归 | PASS（146/188/280/525/247 逐 token 与 AR011 相同，q=1.000） |
| ST-6 | 开关关闭 = AR011 | PASS（`selector_sentence_completion_off_matches_ar011`） |
| ST-7 | 仅多事实生效 | PASS（`selector_sentence_completion_is_multi_fact_only`） |
| ST-8 | 跨块答案补齐 | PASS（`selector_sentence_completion_keeps_split_answer`） |
| ST-9 | 组合闸门（没有免费收益） | PASS 7/7 |
| ST-10 | 闸门有效性：注入劣化必须 FAIL | PASS（`--inject deep/multifact8k:ar012-trim`） |
| ST-11 | GPU（CUDA）一致性 | PASS（16/16 例 sel 相同；2762 → 816 ms） |
| ST-12 | 无模型 167 / 真机 167 / -Werror / ASan | PASS |
| ST-13 | `./repro.sh ar012` 一条命令 | PASS |

### 未达成 / 已回退

- 多事实 hard 仍 0.739：事实已 99% 在上下文，剩余来自模型自身复制/排序能力；
- 多事实 token +15~18%（431→509、445→513）：为完整答案付的代价，闸门按"有收益才允许涨"限制；
- 加预算（b=0.07–0.16）负收益（0.836 → 0.771）→ 否决该方向；
- 补齐后裁剪（`sentence_trim=1`）4K/32K 掉到 0.827 → **不路由**，仅保留实现与记录；
- 单事实补齐 +15% token 无质量收益 → **门控为多事实**。
