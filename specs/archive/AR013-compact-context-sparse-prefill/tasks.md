# [AR013] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR013 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-29 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 审计字段 `ctx_cells`/`kv_alloc_bytes`/`position_span` 打通（Selection→Sample→raw.csv） | - | passing | 另加 `kv_used_bytes`/`rss_mb`/`ttft_min_ms` |
| T002 | bench 上下文按准入量申请 + `--n-ctx` 生效 + 显式不足报错（FR-1） | T001 | passing | `ensure_context` 按需增长；分配量与 llama 日志对齐 |
| T003 | 块级位置压缩核心（selector 输出 `positions`，块内偏移保留）（FR-2） | T002 | passing | `position_compaction=off|block` |
| T004 | 下游位置重映射一致性：gather 索引、chunk 边界、prefix_len、dump（FR-2） | T003 | passing | gather 仍用原始坐标，`origins` 映射到行号 |
| T005 | 单测：压缩单调性/块内不变/幂等/映射/审计 | T003,T004 | passing | 3 例；无模型 209 例全 PASS |
| T006 | 64K 真实数据集生成 + 超训练长度跑通与 dense 失败路径（FR-3） | T002,T004 | passing | `gen_niah.py --words 57000` |
| T007 | `ar013_matrix.py` + `ar013_report.py --gate` + 注入劣化（FR-4） | T002,T004 | passing | 22 项检查；2 种注入必 FAIL |
| T008 | 全量实验：8 数据集 × 2 臂 + CPU/CUDA 一致性 + 真机单测 | T006,T007 | passing | CUDA 逐例一致（含 0.975） |
| T009 | 文档（README/research/architecture/portfolio/组件详设/repro.sh）+ 归档 + tag + 推 Gitee + 克隆复现 | T008 | passing | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-29 需求阶段（HarnessX sdd-task-requirements）

- 领域知识：读组件详设、AGENTS.md、`selector.cpp`/`benchmark.cpp`/`sparse_prefill.cpp`/`backend.cpp`。
- 关键发现（真实二进制实测，非推断）：`benchmark.cpp:278` 硬写 `ctx_opts.n_ctx = 0`，
  llama.cpp 取 `n_ctx_train=32768` → **4K case 分配 896 MiB KV，实际只写 3.5 MB**（256× 过分配）。
- 微实验（`/tmp/ar013-probe`、`/tmp/ar013-ctx`，现有二进制，无代码改动）：
  - 准入 token 固定 146：threads 8/16 无明显差别（104 vs 101 tok/s），`n_ubatch=128` 更慢，
    `attn_top_k=0` 更慢（gather 本身不是瓶颈）；
  - 结论：TTFT ≈ 5.5–9.6 ms × 准入 token，与上下文申请大小无强相关，
    **真正的浪费在内存与能力上限，不在每 token 计算**。
- 结论：AR013 主攻"KV 按准入量分配 + 位置压缩"，并顺带解锁超训练长度文档。

### 2026-10-29 开发与验证会话

- 实现：Selection 平行位置向量 + `compact_positions_block`；`sparse_prefill` 用
  `origins` 做行映射；`LlamaBackend::ensure_context`/`kv_alloc_bytes`/`rss_mb`/`n_ctx_train`；
  bench 起始 256 格 + 每 case 增长；CSV 新增 6 列 + `ttft_min_ms`。
- 修掉的两个真实 bug：
  1. 生成/打分仍用原始 prompt 长度作位置（compact 后 `llama_decode` 报
     "sequence positions must stay consecutive"）→ 改用 `stats.kv_pos_max + 1`；
  2. `context-too-long` 早退用的是"当前上下文大小"而非模型训练上限 →
     改为 `n_ctx_train`/显式 `n_ctx`，仅在关闭压缩时生效（FR-3 需要 64K 能跑）。
- 结果：KV 112→7 MiB(4K)、854→21 MiB(32K)；多事实 0.887→**0.975**、hard 0.739→0.746；
  64K 文档 AR012 无样本（`context-too-long`）而 AR013 4/4 例 q=1.000。
- 闸门 22 项全 PASS，聚合 TTFT 0.958×；两种注入均 FAIL。
- 测试：无模型 209 例、真机 209 例全 PASS；`-Werror` 零告警；CUDA 逐例一致。

## 阶段门控记录

### 2026-10-29 req 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（背景可量化 / 范围清晰 / FR 可测 / 验收 Given-When-Then / NFR 有阈值 /
  约束明确 / 假设列出 / 任务可独立验证 / 依赖成链）
- 修复的问题：删除不可达的"TTFT 改善 ≥15%"预注册（微实验表明时间收益有限），
  改为"不增（3% 容差）+ 如实记录"，避免虚报；主要收益明确为内存与能力上限。
- 遗留风险：FR-2 的质量不降是硬条件，若块间距离压缩伤害多事实跨块推理则整项回退（已写入 FR-4）。

### 2026-10-29 design 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：6 项（方案覆盖 FR / 接口完整 / 流程图覆盖分支 / 测试可执行 / 代码边界清晰 / 兼容与回退）
- 关键设计决策：压缩只在选择器收尾做一处；`token_indices` 语义不变、新增平行 `positions`；
  bench 两阶段预扫求精确位置跨度；分配量用 `llama_state_get_size` 与 llama 日志对齐。
- 设计执行中的两处修正（已同步 design.md 与本记录）：
  1. "两阶段预扫"改为 **`ensure_context` 按需增长**：预扫要复制 planner/模式/预算的全部逻辑，
     成本与风险都高于按需扩容，且扩容次数被跨度单调性限制；
  2. 分配量改用 **模型几何计算**：`llama_state_get_size` 只返回已写入状态（4.3 MiB 恒定），
     无法反映分配。
- 遗留风险：预扫描方案取消后，`n_ctx` 增长会重建上下文（本会话实测每轮 ≤3 次），
  重建不计入 TTFT。

### 2026-10-29 review + ST 记录（自审，用户授权全自动）

- Review：代码与 design.md 一致（接口/字段/流程）；无新增告警（`-Werror`）；无 mock；
  所有质量数字来自真实 GGUF。
- ST：8 数据集 × 2 臂矩阵 + 4 项闸门 + 2 种注入 + 64K 能力 + CUDA 一致性 + 真机 209 例，
  全部通过（详见 `benchmarks/results/ar013-compact-context/REPORT.md`）。
- 结论：AR013 进入最佳组合（`position_compaction = 1`），归档。
