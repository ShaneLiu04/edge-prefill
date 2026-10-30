# [AR015] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR015 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-30 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 序列状态读写与真机等价测试（复用 `LlamaBackend::save_state`/`load_state`，先写等价测试） | - | passing | 含位置空隙序列；装载后继续生成与冷路径逐 token 相同 |
| T002 | `KvSnapshotStore`：`SnapshotKey`（模型/策略/log/位置/n_ctx/KV 精度）+ manifest + SHA256 校验 + 原子写 + LRU | T001 | passing | 纯逻辑可无模型单测；key 字段逐项断言 |
| T003 | bench 接入：`snapshot_dir`/`snapshot_use`/`snapshot_max_bytes` + 命中/未命中路径 + 6 个审计列 | T002 | passing | 校验恒开（无开关）；未命中回退到原路径 |
| T004 | 会话组合：每轮落盘 + "重启续跑"臂 + 前缀校验/重建 | T003 | passing | 与 AR014 log 哈希对齐 |
| T005 | `ar015_matrix.py`（cold/warm 双进程 + 会话三臂）/ `ar015_report.py`（命中/质量/延迟/体积/会话闸门 + 两种注入） | T003 | passing | |
| T006 | CPU 全量实验：8 数据集 cold/warm + 会话三臂 | T005 | passing | 真实 GGUF |
| T007 | GPU（CUDA）实验：cold/warm（状态走 host 缓冲 + H2D） | T006 | passing | RTX 5000：0.031×/0.035× |
| T008 | 文档（README/research/architecture/portfolio/组件详设/repro.sh）+ 归档 + tag + 推 Gitee + 克隆复现 | T007 | in_progress | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

### 2026-10-30 需求阶段（HarnessX sdd-task-requirements）

- 领域知识：AR001–AR014 已把单次 prefill 压到 0.78–3.0 s（4K–32K），KV 已用 28.7 KB/token；
  AR014 的会话复用只在进程内有效；`PrefixCache` 未接入 bench。
- 量化依据：TTFT 0.78/2.21/2.98 s（4K/多事实/32K）；已用 KV 14.4 MB（500 token）、62 MB（1948 token）；
  E2E 中生成占大头（48 token ≈ 16.9 s），因此本 AR 聚焦 TTFT/首 token，而不是 E2E 吞吐。
- SOTA 对照：vLLM automatic prefix caching、SGLang RadixAttention（进程内前缀树）、LMCache（KV 卸载到
  CPU/磁盘）、Mooncake（KVCache 为中心）、llama.cpp `--prompt-cache`、Anthropic 服务端 prompt caching、
  CacheBlend（近似复用 + 重算修正）。本项目差异：**准入后的稀疏 log 才是前缀**，key 必须包含
  准入策略指纹与位置信息；且只做严格相同序列，不做近似拼接。
- 结论：AR015 = 内容寻址的序列 KV 快照 + 跨进程热启动，补齐 AR014 的生命周期缺口；
  与 AR013 强协同（快照体积直接由紧凑 KV 决定）。

## 阶段门控记录

### 2026-10-30 req 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：10 项（背景可量化 / 范围清晰 / FR 可测 / 验收 Given-When-Then / NFR 有阈值 /
  约束明确 / 假设与风险列出 / 任务可独立验证 / 依赖成链 / 组合关系明确）
- 关键约束：只复用完全相同的 token 序列与位置；key 与 manifest 双重校验；不匹配必须回退重算。
- 遗留风险：序列状态在 llama.cpp 内的跨版本兼容性（key 带 commit 指纹）；CUDA 状态装载带宽待实测。

### 2026-10-30 design 门控记录（自审，用户授权全自动）

- 门控结果：PASS（第 1 轮）
- 审查项数：7 项（方案对比完整 / 接口完整 / 流程图覆盖 cold-warm 与会话两条分支 /
  测试设计覆盖正常与异常 / 代码边界清晰 / 与 AR013/AR014 的兼容与回退明确 / 关键风险有验证手段）
- 关键设计决策：快照只覆盖 `log[0..n-1)`，warm 路径只解码最后一个 token →
  首 token logits 可比对，且不会出现"状态最后格与 decode 不一致"。
- 否决方案：段级 KV 分页（llama.cpp 无部分序列读写 API，成本/收益不成立）、
  进程内 KV 池（已是 AR014 范围）。
- 遗留风险：`llama_state_seq_load_file` 对含位置空隙的序列的还原正确性 → 由 T001 的真机
  等价测试（生成逐字节一致 + logits ≤1e-6）先行验证；CUDA 装载带宽 → T007 实测，不达标则关闭 GPU 路径。

### 2026-10-30 ST 验收记录

- 单元/集成：无模型 181 例全 PASS；真机（`EDGE_PREFILL_TEST_MODEL`）快照 5 例全 PASS，
  含"位置空隙序列装载后继续生成与冷路径逐 token 相同"；`-Werror` 零告警。
- CPU 矩阵（8 数据集 cold/warm + 多轮三臂）：warm TTFT 0.027×–0.046×，质量逐例相同；
  多轮 warm 0.101×（40/40 命中）、resume 0.097×（40/40，`session_log_hash` 与 cold 一致）。
- GPU（RTX 5000 / CUDA 12.0 / 全层卸载）：4K 0.031×、多事实 0.035×，质量逐例相同，
  查门在该结果上同样全 PASS。
- 闸门：`ar015_report.py --gate` 22 项全 PASS；注入 `no-hit` → hit FAIL，
  注入 `quality-drift` → quality FAIL。
- 实测踩坑（已修复，写入报告 §7）：LRU 默认上限导致多轮 0 命中；会话 warm TTFT 漏计装载；
  key 误用整段 prompt；"只存 KV 重解末 token"与冷路径差 0.79–0.90 且更慢。
- 遗留：快照装载受页缓存影响（未做 drop_caches），报告中已写明口径；
  段级 KV 分页（跨 log 共享前缀）因 llama.cpp 无部分序列 API 而不做。
