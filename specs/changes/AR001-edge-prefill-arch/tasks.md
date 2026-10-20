# [AR001] 任务跟踪：端侧 LLM Prefill 架构创新

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-20 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 仓库骨架：CMake、目录、LICENSE、AGENTS.md、CI、格式与警告策略 | - | pending | 高质量基线 |
| T002 | llama.cpp 固定版本集成 + 可复现构建（FetchContent/子模块） | T001 | pending | 记录 commit |
| T003 | llama.cpp prefill-only FFN skip 最小 patch 与开关 | T002 | pending | mask=0 等价 |
| T004 | GGUF 后端封装：模型/上下文/分词/batch/KV 状态/计时 | T002 | pending | RAII、错误码 |
| T005 | 位置保持稀疏 prefill 与 dense 等价性测试 | T004 | pending | 核心正确性 |
| T006 | 多信号块级选择器：IDF/BM25、多样性、位置先验、可选 draft | T004 | pending | 可审计 JSON |
| T007 | 必选块、预算控制、确定性、短 prompt dense 回退 | T006 | pending | 质量保障 |
| T008 | 前缀 KV Radix/Trie 缓存 + LRU + 命中统计 | T004 | pending | 真实复用 |
| T009 | 自适应分块 prefill 与 chunk 级计时 | T004 | pending | 与一次性等价 |
| T010 | 相位感知 KV 精度配置与内存统计 | T004 | pending | f16/q8_0/q4_0 |
| T011 | 策略规划器：latency/balanced/quality/reuse + 配置序列化 | T005,T007,T009,T010 | pending | 可审计计划 |
| T012 | 真实基准数据集：NIAH、抽取式 QA、WikiText PPL | T004 | pending | 脚本 + 校验和 |
| T013 | 指标与报告：TTFT、总延迟、tokens/s、KV、质量保持率、Pareto | T012 | pending | CSV/Markdown |
| T014 | CLI：run / bench / eval / inspect / plan | T011,T013 | pending | 易用性 |
| T015 | 集成测试：端到端真实 GGUF、异常路径、确定性 | T005,T008,T009,T014 | pending | 无 mock |
| T016 | 8K/4K 实验矩阵与参数调优，达成 ≥2x 且 ≥97% | T015 | pending | 主验收 |
| T017 | 前缀复用、KV 量化、FFN skip、泛化模型实验 | T015 | pending | 辅助结论 |
| T018 | README、架构文档、实验文档、复现脚本、研究综述 | T016,T017 | pending | 交付 |
| T019 | 代码审查、规范合规、性能/内存复核 | T018 | pending | Review 阶段 |
| T020 | ST 验收、结果归档、Gitee 同步与 tag | T019 | pending | 归档阶段 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

## 阶段门控记录

### 2026-10-20 design 门控记录（第 9 轮）

- 门控结果：PASS（第 9 轮）
- 审查项数：15 项
- 修复的问题：G3/G11/G15 契约与矩阵语义（token_to_piece、DraftScorer、SparsePrefiller、FfnSkip、UT-PL-07 等）
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 8 轮）

- 门控结果：FAIL（第 8 轮）
- 失败项：G3/G11/G15 DraftScorer 未初始化用例不可测、UT-PL-07 budget-clamped 无出口
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 7 轮）

- 门控结果：FAIL（第 7 轮）
- 失败项：G11/G15 矩阵引用与 IT-SP-06 getter、token_to_piece 边界
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 6 轮）

- 门控结果：FAIL（第 6 轮）
- 失败项：G3/G11/G15 token_to_piece 返回类型与异常用例不一致
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 5 轮）

- 门控结果：FAIL（第 5 轮）
- 失败项：G3/G11/G15 契约与矩阵语义（DraftScorer/SparsePrefiller/FfnSkip）、UT-PL-15 不可达、IT-CH-04 规则冲突
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 4 轮）

- 门控结果：FAIL（第 4 轮）
- 失败项：G3 create_context/PhaseKVPlan 契约、G5 KV 回退路径、G11 DraftScorer/matrix、G15 SparsePrefiller 用例
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 3 轮）

- 门控结果：FAIL（第 3 轮）
- 失败项：G3 handoff 契约、G10 组件表述、G11 Planner 判据、G13 handoff 分支、G14 追溯 ID、G15 矩阵语义
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 2 轮）

- 门控结果：FAIL（第 2 轮）
- 失败项：G3 类型/错误契约、G11 条件判据、G13 分支覆盖、G14 追溯、G15 边界引用语义
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 design 门控记录（第 1 轮）

- 门控结果：FAIL（第 1 轮）
- 失败项：G3 接口契约不完整、G11 用例判据不明确、G12 功能点 10/12 缺用例、G13 分支覆盖缺失、G14 验收追溯缺口、G15 接口边界覆盖缺失
- 严重性：Important
- 审查代理：sdd-gate-reviewer（只读子代理）

### 2026-10-20 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项
- 修复的问题：G3 的 5 处模糊验收措辞已收紧（FR-1/4/6/7）
- 审查代理：sdd-gate-reviewer（只读子代理）

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。
