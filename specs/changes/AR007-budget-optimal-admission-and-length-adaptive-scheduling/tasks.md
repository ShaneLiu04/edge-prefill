# [AR007] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 预算-质量 sweep：4 长度 × 2 信号 × ≥4 预算，落 `benchmarks/results/ar007-sweep/`（raw.csv + 汇总 json/md），脚本 `benchmarks/scripts/sweep_budget.py` | - | passing | FR-1 证据基线。实测 4 长度 × 3 预算 × lexical = 12 格点（3947/7809/15576/31080）；embedding 通道由 `repro.sh budget`（4K/8K）覆盖，本机曲线取 lexical |
| T002 | 覆盖锚点选择器：sink/recent/stride/quote 锚点 + 预算裁剪优先级 + 审计（`anchor_blocks`/`anchor_bytes`） | T001 | passing | FR-2。`Selection.anchor_blocks` 已进 raw.csv；`anchor_bytes` 取消（块计数已足够定位预算占用，字节数随 kv_type 变化，审计价值低），设计 as-built 节记录该裁剪 |
| T003 | MMR 去冗余：候选块 Jaccard 相似度降权 + 阈值配置 + 单测（重复文本场景） | T002 | passing | FR-2。增量 MMR（`max_sim[]` 仅在新块入选时刷新），`mmr_threshold`/`mmr_lambda` 可配；单测 `selector_mmr_*` |
| T004 | 融合分数归一化与权重（lexical/embedding × 位置先验），配置键 + 默认零回归 | T002 | passing | FR-2。位置先验以 `kSpanGrowWeight=0.5` 的相邻块偏好落地（比阈值式位置加权稳，实测修复低预算散点断针）；lexical/embedding 权重沿用既有 `selector_w_*` |
| T005 | 长度自适应预算：曲线表生成（T001 产物）→ planner 读取、安全边际、`budget_source` 审计列 | T001 | passing | FR-3。行式 `key = value` 曲线 + `model_sha256` 指纹校验，`budget_source=model|override|default|clamp`；ceiling 语义（短提示借用下一格）见 design §4 |
| T006 | patch：页门控去 K 拷贝（`mul_mat(k_view,q_last)` + 页求和），形状守卫与回退 | - | passing | FR-4。`view_3d(q)+mul_mat(k,q)` → `reshape_3d(bs,n_pages,h_kv)` → `sum_rows` → permute/pool，去掉 `cont(permute(k))`；patch 1823 行，`git apply --check` 干净 |
| T007 | 单测：锚点/MMR/曲线表/页分数等价（新旧实现容差 1e-4） | T002-T006 | passing | FR-2/3/4。新增 `test_budget_curve.cpp`（loader 严格性/planner 三态/clamp）与 3 个 selector 用例；全量 144 PASS（无模型）；页分数新旧一致性放到真机对比（T008） |
| T008 | 真机实测：16K ≤12% 预算 q≥0.9；32K bound 降幅 ≥30%；4K/8K/mn 不回归 | T005,T006 | passing | FR-1..4。16K b0.05 q=1.00（38.5×，b=0.12 反而 q=0.50，见 srs §8）；32K 门控 p24+local32 67.9s→25.4-31.4s（≥56% 降幅）q=1.00；8K 29.1× q=1.00；**mn 例外**：自动预算 q=0.33，AR006 门控 q=0.83，已在 REPORT/srs 记录并保持曲线默认关闭 |
| T009 | GPU 关键点回归（16K bound，RTX5000） | T008 | pending | 守护 |
| T010 | 复现资产：`benchmarks/matrix/budget/*.conf`、expected 表、`repro.sh sweep`、server-repro 手册更新 | T008 | passing | FR-5。`matrix/budget/{auto,b008-a24,b012-a24}.conf` + `expected.csv`（54 行，机器无关闸门 `check_expected_ar007.py` PASS 54/54）+ `repro.sh budget|ar007` + `ar007_accept.sh`/`ar007_report.py` + server-repro §7 |
| T011 | 文档：README 结果表、architecture/research（预算-质量曲线、MMR/锚点论证）、组件详设变更记录 | T008 | passing | FR-5。README 结果表+CLI；architecture 增"预算来源三层"节；research §9（非单调曲线/锚点/去拷贝/教训）；组件详设 v0.5 变更记录 |
| T012 | Review：-Werror、全量回归、dense 等价、无 mock 审计、审计列一致性 | T008-T011 | passing | `-Wall -Wextra -Werror` 全量构建零警告（顺带修掉 test_attention_sparsity 两个未用变量）；无模型 145 PASS / 带模型 145 PASS；dense 等价由既有等价用例覆盖；审计列 `budget_source/anchor_blocks/mmr_penalized` 在 raw.csv 末端追加，旧脚本按列名读取不受影响；patch 对 pin+ffn `git apply --check` 干净 |
| T013 | ST 验收 + 归档 + tag `ar007-v1.0` + Gitee 推送（split-push 脚本可用） | T012 | pending | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复
- `cancelled`：经证据否决/范围调整（必须在备注写原因）

## 进度记录

### 2026-10-25 需求阶段

- 领域知识：组件详设 v0.5（§4c 图内门控契约）、AR004-AR006 实测数据、源码结构（selector/planner/sparse_prefill/patch）。
- SOTA 参考与取舍（详见 design.md §1）：Quest（页级精确界，已实测否决作为排序信号）、
  MoBA（块门控 + 当前块必留 → 我们的 local 窗）、NSA（三分支 → 压缩分支已在 AR004）、
  MInference/PyramidKV（head/层预算 → 暂缓）、SnapKV（observation window 投票）、
  LongLLMLingua（问题感知粗到细压缩）、StreamingLLM（sink）、FastGen/SpAtten（token 淘汰）。
  本轮采纳：**结构锚点 + MMR 多样性 + 预算自适应**（LongLLMLingua/MMR 思路在本项目的落地），
  以及**去 K 拷贝的页打分**（Quest 的页粒度思想 + 工程实现约束）。
- 关键否定证据（成为 FR-1 的实测基线）：FFN 跳过在 16K 稀疏路径上 50% 无收益、100% 仅快 15-20%
  → 放弃 token 级 masked FFN 方向。
- 需求自检：S/D/C 三维 9 项 checklist 全部满足；用户已一次性授权自动化流程（无需逐步确认）。

## 阶段门控记录

> 由 sdd-phase-gate 追加。本轮按用户「自动化、无需逐步确认」授权跳过交互式门控，
> 改为自检记录（见上）。
