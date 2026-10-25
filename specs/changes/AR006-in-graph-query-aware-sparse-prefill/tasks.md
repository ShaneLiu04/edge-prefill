# [AR006] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-25 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | ggml 能力实测探针：`ggml_pool_1d(MAX)`、`ggml_neg`、`ggml_top_k`/`ggml_argsort`、动态索引 `ggml_get_rows` 在 CPU 后端的真实可用性与开销（小程序+计时，真实 ggml 后端，非 mock）；给出 in-graph 选择 vs 阈值法+host 预选的决策 | - | passing | 实测：pool_1d/neg/top_k/get_rows 可用；F32→I32 cast 无路径（改用 row_table 查表）；逐层 mask 转置过贵（提升为每图一次）；精确界否决、改 MoBA 页均值 |
| T002 | patch：K 页 min/max 元数据图内导出（页=32，逐头逐维）+ C API 开关 + 关闭时 dense 字节等价回归 | T001 | passing | 实现为页均值元数据（AVG），C API `llama_set_prefill_qg_bound`；关闭时路径不建节点 |
| T003 | patch：分片页打分（q·mean K_page）+ top-k 页选择节点 + 审计；非法页尺寸/非有限值回退 | T002 | passing | 16K 实测 q=1.0（p16/p24/p32），p8 质量下限证据 |
| T004 | patch：单次并集 gather + 可见性掩码（因果+local+selected+pool），与逐组参考实现等价性单测 | T003 | passing | 单 ubatch 一次 gather（local+selected），mask 由每图一次预转置的 kq_mask 行 gather；真实模型确定性/恢复等价用例已加 |
| T005 | 规划侧三分支预算：`qg_local_rows`/`qg_sel_pages`/`qg_pool_pages` 配置键、审计列、降级链 | T004 | passing | 配置键+CLI+审计（qg_source=attention-bound, sem=0）；pool 仍走 AR004 路径（qg_pool_pages 预留） |
| T006 | embedding 降级为可选：默认 `qg_source=attention-bound`、`semantic_score_ms==0`；保留 2D 回归 | T003 | passing | 默认路径 sem=0 实测；embedding-2d 路径保留（campaign 中 qgv2 对照） |
| T007 | head 路由 auto：由真实页统计导出 retrieval/streaming 分类（确定性、可写 profile），off 时逐字段等价 | T003 | pending | 设计预留（per-head mask 不可行 → 需拆两次 attention），优先级低于全量验证 |
| T008 | 真实模型实测矩阵：4K/8K/16K/32K/mn，1.5B CPU 16 线程，含 AR005 对照与净记账 | T005,T006 | passing | 见 `benchmarks/results/ar006-campaign/REPORT.md`；8K 9.4x/q=1.0，16K 6.8x/q=1.0（vs qg-2d 6.4x，sem 1~10s→0），mn 3.8x/q=0.83（scalar 2.8x/0.67），32K 默认 local 质量失败、`local_rows=32` 时 5.7x/q=1.0；16K/32K 的 ≥8x/≥10x 目标未达成，根因已记录 |
| T009 | GPU 关键格点守护（RTX5000，n_gpu_layers=99） | T008 | pending | top_k/get_rows 后端可用性待验 |
| T010 | 7B/14B 配置模板 + `docs/server-repro.md` + `expected.csv` | T008 | passing | `benchmarks/matrix/bound/{dense-ref,bound-p16,bound-p24}.conf`、`bound/models/qwen2.5-{7b,14b}-*.conf`、`bound/expected.csv` + `check_expected_ar006.py`（5/5 通过） |
| T011 | README 一键复现改造 + docs（architecture v0.5 / research v3） | T008 | passing | README（AR006 能力 + CLI + 结果表）、architecture §gate placement、research §8、experiments AR006 节、组件详设 §4c/变更记录 |
| T012 | Review：-Werror、全量回归、dense 等价、无 mock 审计 | T008-T011 | pending | |
| T013 | ST 验收 + 归档 + tag `ar006-v1.0` + Gitee 推送 | T012 | pending | |

## 关键路径与门

- **决策门 T001**：若 `top_k`/动态 `get_rows` 在 CPU 后端不可用或开销不可接受 → 采用
  “阈值法 + 预计算页分数张量 + host 读回一次/层”的备选实现（设计阶段已预留，见 design.md）；
  若两者都不满足性能门，则降级为 “plan 级上界打分（用上一 ubatch 的实际 K 作代理）+ 静态 gather”。
- **串行约束**：T002-T004 都是 patch 改动，必须同一构建源；T008 起需要固定二进制快照测量。
- **与 AR005 的关系**：AR005 的矩阵结果作为对照基线入库，不重复测量。

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-25 开发会话 2（测量基础设施纠错 + 战役）

- **修 `--repeats` 覆盖 conf 的缺陷**（`src/main.cpp`）：`Args::repeats` 默认值是 5 且
  `if (args.repeats > 0)` 恒真，导致命令行未给出时也会覆盖 conf 的 `repeats`，
  矩阵跑一律 5 次（时间 ×5，且 CSV 记录与实际意图不符）。改为哨兵 -1（=未给出）。
  这是测量基础设施 bug，不影响正确性，但影响本轮与 AR005 的运行时假设；AR005 数字
  因此是“5 次中位数”，统计上更稳，保留并在文档注明。
- **WSL 空闲关机**：`wsl -e` 会话全部退出后发行版会在数秒内关闭，nohup/setsid
  后台任务一并被杀（`dmesg` 可见 “Distribution init process exited”）。对策：
  用 `keepAliveOnTimeout` 的前台调用 + 一个常驻 keepalive 会话跑战役。
- **Dense 计时基线核查（A/B 实测，4K niah-01）**：repeats/warmup 组合
  (1,0)/(5,0)/(3,1)/(1,1) 的 ttft 分别为 49.0/48.9/48.4/48.6s —— 与 warmup/repeats
  无关，是稳定的机器状态；同会话 scalar(0.25) 6.2s。据此确认：稀疏路径的加速主要来自
  选择级稀疏（进入 KV/transformer 的 token 只有 ~25%），而非仅仅 attention 行 gather；
  4K 上 bound 的 local 窗覆盖全部已入 KV 行（k_sel=0）→ 数学等价 dense，质量 1.0，
  但比 scalar 慢 2~3×（local 全量 gather + 掩码行 gather 的固定开销）。结论：
  4K/8K 不是 bound 的目标区间，验收看 16K/32K。
- 战役（`benchmarks/results/ar006-campaign/`）：dense/scalar/qgv2/boundp16[/p24]
  × len4k/len8k/multineedle/len16k/len32k，同会话交错测量；汇总脚本
  `benchmarks/scripts/assemble_ar006.py`，质量门 `check_expected_ar006.py`。
- 4K 首批数据（同会话）：dense 60.6s（5 次中位）、scalar 6.2s（q=1.0）、
  qgv2 28.1s（sem 3.5~9.9s，q=1.0）、bound p16 13.9s（q=1.0）→ 净加速 4.4×，
  且 sem=0：门控路径已把嵌入从关键路径移除。

### 2026-10-25 开发会话 1（图内门控核心落地）

- 完成：patch（cparams 字段 + C API `llama_set_prefill_qg_bound`/`llama_get_prefill_qg_bound` +
  图分支 + 输入张量 + 复用守卫 + 互斥清理）、C++ 侧（backend/config/CLI/benchmark/sparse_prefill 接线、
  审计 `qg_source=attention-bound`、`semantic_score_ms=0`）、单测 2 个（配置解析）与真实模型用例 2 个
- 实证过程与结论（全部真实运行）：
  1. 精确界（Quest min/max）实现后：4K 无稀疏（local 覆盖全部）可跑；16K 质量 0（"1234567890"/"copper"），
     `sel_pages=0` 时 16K wall 290s ≈ dense 312s → 上界排序对 key 范数敏感，**否决为排序信号**；
  2. 改为 `q·mean(K_page)`（MoBA 门控）后：16K niah-01 `sel_pages=32` q=1.0 ttft=48.5s（dense 312s →
     **6.4×**）；`sel_pages=16/24` q=1.0；`sel_pages=8` q=0 → **预算下限在 8~16 页之间**；
  3. 工程坑（已解决，均记录于 patch 注释）：ggml 无逐元素 max → 上界需 relu 分解（后随否决删除）；
     ggml 无 F32→I32 cast 路径 → 页→行展开改为 `row_table[32,n_pages]` + `get_rows` 查表；
     get_rows 要求 `idx.ne[1] == src.ne[2]`（头维复制）；concat 轴（K/V 沿 ne2、mask 沿 ne0）；
     逐层 mask 转置开销 → 提升为每图一次（`bound_mask_p`）
- 待完成：16K/32K/mn 全量质量与净加速、4K/8K 回归、GPU 守护、文档与 expected 表、Review/ST

## 阶段门控记录

### 2026-10-25 req 自检记录（自动化模式）

- 门控结果：PASS（自检，第 1 轮）
- 说明：用户明确免除逐阶段人工确认（“每一步无需我确认，但要有反馈文字”），按 sdd-phase-gate 的
  S 维 checklist 自检：背景/目标可测（净 TTFT 北极星指标量化）、范围边界清晰（Out of Scope 明确）、
  8 条 FR 均有触发条件/期望行为/异常处理/可测验收、非功能指标量化、约束与假设显式（含 T001 决策门）、
  降级链与无 mock 红线继承。未发现 Critical/Important 缺口。
- 待办（进入设计后仍须落实）：T001 能力探针是 FR-2 实现形式的硬前置；7B 模板需服务器实测区间。
