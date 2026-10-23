# [AR004] 任务跟踪：NSA/MoBA 式分支稀疏架构创新与 Gitee 开放复现

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-22 |
| 前序 AR | AR003（基线能力保留，回归零破坏硬门禁）|

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 大 chunk 默认调度：chunk_size=0 自动对齐 n_ubatch、backend 不二次拆分的断言防线、原崩溃配置回归测试、全稀疏关闭 dense 等价 <1e-5 | - | passing | 2026-10-22 完成；附带修复预存缺陷（probe_argmax 未按 n_batch 分块 + classify 测试未考虑 n_ctx 运行时填充） |
| T002 | 全长度基准刷新：4K/8K/16K/32K × {dense, global, head} 大 chunk 配置重跑，刷新头条数字 | T001 | passing | 2026-10-23 完成；隔离进程+冷却间隙+--repeats 3；4K/8K attn 2.5×/2.6× q=0.44，16K attn 3.9× q=1，32K attn 12.4× q=0.25，head 全线 q=0（与历史一致）；首轮连续套件受满载降频污染保留审计；config repeats 键被 CLI 默认覆盖的坑已记录 |
| T003 | PPL 质量指标修复：定位 quality=0 根因、与 llama.cpp 官方 ppl 交叉验证、综合 q 落地 CSV | - | passing | 2026-10-22 完成；根因=exp(均值对数概率)口径；quality 改为 min(1, ppl_dense/ppl_sparse) 保持率，dense 参考 per (case,length) 缓存，原始 ppl 入 CSV 审计列；EX-8 空 answer 跳样本+警告 |
| T004 | 压缩分支 patch：`llama_set_prefill_attn_compressed` C API + ggml 池化行（reshape/mean/concat）+ 因果 mask（块内最大位置）+ Unsupported 回退 | T005 | passing | FR-2 核心 |
| T005 | 压缩分支规划与执行：planner 输出被丢块池化集合、sparse_prefill 集成、分段最低保留（防整段全丢）、审计字段 | T004 | passing | FR-2 |
| T006 | 压缩分支消融实验：attn-24 @8K/16K 开/关对比，q 0.44→≥0.75 验证 | T005, T002 | passing | 2026-10-23 完成；pool on @8K q=0.44→1.0（目标≥0.75 超额达成）、@16K 保持 q=1；压缩分支把被丢块的池化记忆保住 needle；数据 ar004-t006-t009/len*-attn24-pool |
| T007 | query-chunk 门控规划：draft per-position 分数落盘、分片×块得分矩阵、分片级行集合（必含 sink+recent+前文锚）、分片数上限降级 | T008 | passing | FR-3 |
| T008 | query-chunk 门控 patch：graph 按分片循环 gather-concat（复用 head 分组机制）、空集合强制 sink+recent、head 矩阵优先互斥 | T007 | passing | FR-3 核心 |
| T009 | query-gating 消融实验：8K/16K 多针数据 开/关质量对照，≥5pp 提升或不可达分析 | T008, T002 | passing | 2026-10-23 完成（不可达分析）；多针 mn-v1/v2/v3 @8K：scalar attn-24 q=0/0/0.2 vs qg(512/16/256, 8组激活) q=0/0/0，无≥5pp 提升；根因=1D per-pos 词法分数使各组行集退化为同源因果全局 top-k（组间仅 sink/recent/因果窗差异），且答案 token 不在问题中、干扰块 vault/passcode 词法命中稀释 needle 块均分；改进需 DraftScorer 语义分数做位置×块矩阵（已记录为 v1 局限）；数据 ar004-t006-t009/mn-* |
| T010 | head 模式路由：HeadProfile 分数 → 锚点/局部/检索三模式模板（k+必留块映射）、阈值配置、审计 | - | passing | 2026-10-23 完成；分类序 retrieval(-1)→landmark(k=32,echo≥0.5)→local(k=48,induction≥0.5)→normal(k=head_top_k)；必留锚点由 pool 必留前缀天然保障（小 k=保留 sink/recent）；reason 含各类计数审计；CLI --head-pattern-routing；默认 off 零回归 |
| T011 | 金字塔先验默认矩阵：k(l) 凸函数、与 AR003 矩阵格式兼容、无 profile 时默认启用 | T010 | passing | 2026-10-23 完成；k(l)=round(k_min+(k_max-k_min)·2\|l-(L-1)/2\|/(L-1))，单层=k_max；无 profile 时 --pyramid-prior on 启用（默认 off 保零回归，srs 触发条件为显式 on，与 tasks.md 原表述"默认启用"的偏差已注记）；CLI --pyramid-k-min/max 校验 min≤max 否则 exit 2；真实模型冒烟 pool=2009 k=48/128 通过 |
| T012 | GPU sparse 路径：phase5 CUDA 构建、scalar/head 路径 GPU 正确性、GPU sparse vs dense-FA ≥1.3x 或诚实分析 | T001 | passing | 2026-10-23 完成；CUDA 增量构建（含 pool+qg patch）通过；GPU 正确性 8K/16K q 与 CPU 同 regime（0.57/1）；16K attn24 GPU 14.4×（sparse 11.0s vs 同跑 dense 159s），vs 历史 dense-FA 基线 393tok/s→41.6s 为 3.8× ≥1.3× 达标；GPU dense fa=0 慢为 RTX5000 无 FA 实况（历史 llama-bench 59tok/s 一致）；数据 ar004-t012-gpu/ |
| T013 | Gitee 仓库工程化：git init、.gitignore（third_party/构建产物/模型）、诊断脚本归档、README 快速开始+配置表+预期结果表、docs/architecture.md + experiments.md | - | passing | 2026-10-23 完成；仓库已有AR002/AR003历史与gitee remote(笔记过时)，按模板补3个提交：specs(908行)/代码(AR003收尾+AR004 T001/T003/T010/T011)/工程化(README+docs+42组实验CSV入库+diagnostics归档13脚本)；WSL缺git已装(tuna源替换被407卡死的archive源)；README含llama commit+模型SHA256双pin |
| T014 | 模型下载与一键复现：download_model.sh（ModelScope→hf-mirror 择优+SHA256）、download_data.sh、run_experiments.sh 一键头条、sync_gitee.sh | T013 | passing | 2026-10-23 完成；download_model.sh双镜像择优+SHA256不匹配即拒绝(本地模型实测验证通过6a1a2eb6...)；download_data.sh调gen_*.py+make_deep_datasets+数据集SHA256输出；sync_gitee.sh推送前强制clean tree+README双pin校验；run_experiments.sh已有(全矩阵编排) |
| T015 | 主实验矩阵与图表：{4K..32K}×{dense/global/head/+branch/+gating/全开}×{CPU,GPU}、7 张可解释性图（TTFT 曲线/Pareto/悬崖救回/gating 消融/chunk 收割/head 热力图/CPU-GPU 对照）、16K ≥10x 且 q≥0.9 主验收 | T002, T006, T009, T011, T012 | pending | FR-8 主验收 |
| T016 | 研究文档 docs/research.md：≥6 SOTA 三段式对比、每个创新点 CSV/图佐证 | T015 | pending | FR-9 |
| T017 | 代码审查与规范合规：Status/RAII/命名/-Werror/覆盖复核、全开关关闭与 AR003 逐字段一致回归 | T015 | pending | Review 阶段 |
| T018 | ST 验收、归档与 Gitee 交付：端到端真实模型验收（≥10x/≥0.9）、harnessX 归档、结果 CSV+图入库、最终同步 + release tag | T016, T017 | pending | ST 阶段 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 关键依赖与并行策略

- T001→T002（大 chunk 先行，全部刷新基于它）；T003/T010/T013 无依赖可先行
- T004-T006（压缩分支）与 T007-T009（query 门控）为两条创新线，可交错推进
- T012 GPU 由 orchestrator phase5 后台承接，不阻塞 CPU 创新
- T015 汇聚全部数据后出主验收与图表

## 阶段性 Gitee 同步点

- T006 完成后：创新核心之一（压缩分支全链路）+ 仓库骨架首同步
- T014 完成后：复现框架与 README 同步
- T015 完成后：实验结果与图表同步
- T018：收尾同步 + release tag

## 阶段门控记录

### 2026-10-22 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（7 YES / 2 WARN / 0 NO）
- 修复的问题：G3 两处验收量化缺口——FR-6 "容差内"改为 "q 差异 ≤0.02"；FR-7 "同量级"改为 "TTFT 落在仓库 CSV 值 0.5x–2x 区间"
- 审查代理：sdd-gate-reviewer

### 2026-10-22 design 门控记录

- 门控结果：PASS（第 1 轮，有条件）
- 审查项数：15 项（9 YES / 6 WARN / 0 NO）
- 修复的问题：W1 compressed_branch 默认对齐 srs（pool，含 on 别名与回退链）+ V 路池化接线说明；W2 补 5 条 GWT 追溯用例（UT-C15/16/17、BS-8 卫生判据、BS-9 32K）+ UT-C18 审计列；W3 补全 build_*/backend 包装返回类型与错误契约；W4 补 FR-4 PlantUML 流程图；W5 补 IT-Q4/IT-CFG1 边界/BS-10/BS-11 等价用例；W6 修正 head_profile.h 路径、F13/F14 追溯显式化
- 审查代理：sdd-gate-reviewer

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

| 日期 | 进度 | 说明 |
|------|------|------|
| 2026-10-22 | 需求阶段完成 | srs.md/tasks.md 生成，待用户确认与 req 门控 |
| 2026-10-22 | T001 完成 | chunk_size=0 自动对齐 n_ubatch；单 range 旁路（chunk≥n_tokens）；attn 路径 n_ubatch 硬顶防 gather 越界；UT-C15 崩溃几何回归；真实模型全量测试 PASS（含等价 <1e-5）；顺带修复 probe_argmax n_batch 分块与 classify 测试 n_ctx 填充问题（预存缺陷，阻塞门禁）；git 提交待 T013 仓库建立后补 |
