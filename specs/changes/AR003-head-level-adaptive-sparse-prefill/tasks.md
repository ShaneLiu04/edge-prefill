# [AR003] 任务跟踪：Head 级自适应稀疏 Prefill 与 SOTA 融合创新

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成）|
| 创建日期 | 2026-10-21 |
| 前序 AR | AR002（T005–T017 并入本 AR，AR002 标记 superseded）|

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | head profile 数据结构与加载：JSON 解析、模型结构校验、planner 集成点与 `--head-profile`/`--head-adaptive` CLI | - | passing | 2026-10-21 完成 |
| T002 | 离线 head 分类器 `classify-heads`：真实模型重复序列探测（echo/induction 检测）、JSON 输出、确定性保证 | T001 | pending | 无模型 SKIP |
| T003 | per-head gather patch：llama.cpp patch 扩展（per-head indices 张量 + C API + 校验）与 backend 包装、Unsupported 回退全局 top-k | T001 | pending | 最小可审查 patch |
| T004 | per-head 执行路径：sparse_prefill per-chunk per-head 行号映射、retrieval head 稠密直通、审计字段与回归用例 | T003 | pending | dense 等价 <1e-5 |
| T005 | 层-头二维自适应规划：budget → 层×头 top-k 矩阵、draft 深层信号融合、mandatory 锚逐 head 保留、消融开关 | T002, T004 | pending | off=AR002 行为 |
| T006 | 在线质量风险控制器：draft 区分度/覆盖率统计、ok/tighten/dense-fallback 三级动作、审计与 metrics 列 | T005 | pending | 失败按 tighten |
| T007 | Radix 前缀缓存：跨请求/多轮最长前缀命中、真实 state 复用、共享 system prompt（吸收 AR002 T005）| - | pending | 与创新并行 |
| T008 | KV 分页快照管理：序列化、LRU、签名校验、恢复失败回退（吸收 AR002 T006）| T007 | pending | 防错误复用 |
| T009 | 按层 KV 精度：`llama_set_kv_type_per_layer` patch、首尾层保守策略、相位不一致 Unsupported 回退（吸收 AR002 T007）| - | pending | KV -30% 目标 |
| T010 | 自动调优 Pareto 框架：真实 BenchmarkRunner 搜索 budget/top-k/head-adaptive/KV/chunk、前沿与 recommended.conf（吸收 AR002 T008）| T006, T008, T009 | pending | 无代理指标 |
| T011 | CPU/CUDA 实验环境与数据集扩展：GGML_CUDA 构建、多模型多数据集配置（吸收 AR002 T009）| - | pending | 服务器复现基础 |
| T012 | 复现脚本与 README/docs：setup/download/run/sync 脚本、`scripts/sync_gitee.sh`（≤40KB 批次 + raw receive-pack）、大模型实验配置表（吸收 AR002 T010）| T011 | pending | 干净克隆可复现 |
| T013 | 质量保障与回归：dense 等价、无 mock 硬门禁、全能力消融/回退用例、PPL 综合指标（吸收 AR002 T011）| T006, T008, T009 | pending | 硬门禁 |
| T014 | 主实验矩阵与性能迭代：8K/4K × CPU/CUDA × {dense, AR002 全局, AR003 head 自适应} 对照；达标 ≥3.5x/≥2.5x/≥97% 或不可达分析（吸收 AR002 T012/T013）| T013, T012 | pending | 主验收 |
| T015 | 架构文档与研究分析：head 级稀疏设计依据、≥5 个 SOTA 项目对比与创新总结（吸收 AR002 T014）| T014 | pending | docs/research.md |
| T016 | 代码审查与规范合规：Status/RAII/命名/警告/测试覆盖复核（吸收 AR002 T015）| T015 | pending | Review 阶段 |
| T017 | ST 验收与结果归档：端到端真实模型验收、报告归档（吸收 AR002 T016）| T016 | pending | ST 阶段 |
| T018 | Gitee 同步与 release tag：阶段性同步（T006 后/T012 后/T014 后）、收尾全量同步与 tag（吸收 AR002 T017）| T017 | pending | 交付阶段 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 阶段性 Gitee 同步点

- T006 完成后：创新核心（head 级稀疏全链路）首次同步
- T012 完成后：复现框架与 README 同步
- T014 完成后：实验结果同步
- T018：收尾同步 + release tag

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-21 会话记录（T001）

- 完成任务：T001 head profile 数据结构与加载
- 修改文件：include/edge_prefill/head_profile.h（新增）、
  src/head_profile.cpp（新增）、include/edge_prefill/config.h（6 新字段）、
  src/main.cpp（--head-profile/--head-adaptive/--risk-guard/
  --head-retrieval-threshold CLI）、CMakeLists.txt（源与测试注册）、
  tests/test_head_profile.cpp（新增 13 用例，Red 子代理编写）
- 测试：71 PASS / 0 SKIP / 0 FAIL（真实模型全量，58 旧用例零回归）
- 覆盖率：新接口 load/validate 全分支用例覆盖（无 gcov 工具链，按 §6.1 用例
  矩阵降级核验）；-Werror 零警告
- 备注：CLI 参数在 main.cpp static parse_args 内，校验用例由后续 T002
  classify-heads 集成用例与 ST 覆盖；probe_params 六字段全量解析验证

## 阶段门控记录

### 2026-10-21 req 门控记录（第 1 轮）

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项
- 修复的问题：G2/G3 的 Minor 建议（FR-8/FR-11/FR-12 改写为 GWT、FR-8 量化
  "主实验可运行"、FR-5 补 8K 前置条件、状态字段更新为 Reviewed）已在审查后顺手修复
- 审查代理：sdd-gate-reviewer
- 交叉验证：无 mock、最小 patch、dense 等价 <1e-5、错误码契约、质量权重
  与组件详设完全一致

### 2026-10-21 design 门控记录（第 1 轮）

- 门控结果：FAIL（第 1 轮）
- 失败项：G10 组件详设未同步；G2/G14 FR-1 API 契约 srs↔design 冲突；G13 流程图
  "过滤后为空"/"can_reuse 矩阵一致"分支无用例；G12/G14/G15 FR-8 sync_gitee、
  FR-10 4K、FR-9 综合指标、接口边界（全量池/k=1/CLI 非法值/ModelLoad/错误码转换）
  用例缺口；G1/G4 缺边界矩阵与错误处理章节；G9 吸收文件清单未汇总；G8 分支计数
  失真；G11 判据模糊
- 严重性：Important
- 审查代理：sdd-gate-reviewer

### 2026-10-21 design 门控记录（第 2 轮）

- 门控结果：FAIL（第 2 轮）
- 失败项：srs FR-1 触发条件残留旧 API 名（G2/G14 修复不彻底）；FR-1 AC4 矩阵
  清空用例与校验矛盾（G11/G14/G15）；FR-3 AC1 单调性无实现映射与断言（G2/G14）；
  FR-5 AC2/FR-6 AC2 零追溯（G14）；G13 三条精确边界用例落空；G9 §4.4 文件清单
  缺口；G15 审计字段/getter/profile 不可读；G10 分层图缺新节点（次要）
- 严重性：Important（均为文档级）
- 审查代理：sdd-gate-reviewer

### 2026-10-21 design 门控记录（第 3 轮）

- 门控结果：PASS
- 第 2 轮 9 项失败全部修复确认；全维度重审新增 4 项 Minor
  （G2 echo/induction 双分数计算定义、G10 Plan 字段同步与命名映射、
  G14 FR-3 AC3 显式标注、G15 退出码 11 补录）已在门控后顺手修复
- 严重性：Minor（不阻塞）
- 审查代理：sdd-gate-reviewer
- 结论：design 阶段完成，移交 sdd-task-develop
