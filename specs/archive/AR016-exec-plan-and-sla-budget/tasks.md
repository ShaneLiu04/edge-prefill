# AR016 任务跟踪

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | `ExecPlan` 头/实现：字段、来源优先级、`Status` 校验、JSON 往返 + 单测 | - | passing | 纯逻辑，无模型 |
| T002 | 后端真实信息：`loaded_gpu_layers()`、`device_name()`、`flash_attn` 生效标志 + 单测 | T001 | passing | 读 llama.cpp 实际状态 |
| T003 | bench/config/main 贯通 `threads`/`n_threads_batch`/`ubatch`/`n_gpu_layers` + 放置校验（含 `--allow-cpu-fallback`）+ 审计列 | T002 | passing | 先写真机测试：请求 GPU 却 0 层必须 `Unavailable` |
| T004 | 成本模型：`fit_cost_model`、`predict_ttft`（保守上界）+ 单测（含 r² 门槛、退化样本） | T001 | passing | 纯数学，无模型 |
| T005 | `probe` 子命令：真实 GGUF 多长度测量 → plan JSON（样本+拟合+设备信息） | T004 | passing | 无 mock |
| T006 | SLA 预算规划：`budget_target_ms` 求最大可行预算 + 夹取 + `budget_source` 审计 + 单测 | T004 | passing | 不突破 `[budget_min,budget_max]` |
| T007 | `ar016_matrix.py`/`ar016_report.py`：CPU/GPU 计划 × 8 数据集 + 前沿 + 闸门 + 两种注入 | T003,T005,T006 | passing | |
| T008 | 实验：线程扫描、CPU/GPU 放置、SLA 前沿；质量不变验证 | T007 | passing | 交错重复、取最小值 |
| T009 | AR015 勘误（GPU 记录）+ README"如何确认设备生效"+ research §18 + architecture + portfolio + 组件详设 + repro.sh ar016 | T008 | passing | |
| T010 | 归档 + tag `ar016-v1.0` + 推 Gitee + 干净克隆复现 | T009 | passing | 分片推送 |

## 阶段门控记录

（待 req 门控后追加）

### 2026-10-30 ST 验收记录

- 单元/集成：无模型 199 例全 PASS；真机全量 199 例全 PASS；`-Werror` 零告警。
- 放置：CPU 构建请求 99 层 → `Unsupported`（测试 `plan_gpu_request_never_silently_degrades`）；
  CUDA 构建 29/29 → `device_state=gpu`，每行审计列可查。
- 成本模型：CPU `a=-3.12 b=3.460 r2=0.9994`；GPU `a=79.65 b=2.614 r2=0.979`（仿射拟合，采样 5 点 ×3 取最小）。
- CPU 矩阵（8 数据集 × 3 臂，repeats=4，ttft_min）：中位 base16/plan = 2.01×（1.42–4.88），
  中位 plan/sla = 1.98×；58 用例 × 3 臂 f1 逐例相同。
- GPU 矩阵：placement/placement-offload/quality/sla 全 PASS；线程 n/a（0.99×）；
  同 SLA 下 797–803 token vs CPU 610–622（+29–31%）。
- 闸门：CPU 9 项、GPU 9 项全 PASS；注入 `fake-placement`、`cost-skew` 均 FAIL。
- 环境事实：主机共租，16 线程臂跨轮可差 3.6×（4K：3.45/4.40/12.59 s），12 线程臂稳定在 2.33–2.58 s；
  报告保留跨轮区间而非单点；统计量一律 `ttft_min`，SLA 实际值用每数据集中位数。
- 遗留：成本模型仅仿射且只在 64–1024 token 采样；放置校验依赖 llama.cpp 日志行（无公开 getter）；
  本机 GPU 被 P8 限制且 WSL 不能锁频。
