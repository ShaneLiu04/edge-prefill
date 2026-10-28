# [AR012] 详细设计：句子完整准入

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| 关联 srs.md | ./srs.md |
| 状态 | Draft |
| 日期 | 2026-10-28 |

## 1. 目标与不变量

把"准入的原子单位"从 32-token 块改成**句子**：证据句子必须完整（可跨块），
不完整就等于自愿丢掉半个答案；同时把多余填充裁掉。

| 编号 | 不变量 | 验证 |
|---|---|---|
| I1 | `sentence_completion=0` → 与 AR011 逐 token 完全一致 | 回归用例 |
| I2 | `budget=1.0` 且开关关闭 → 与 dense 一致 | 既有等价用例 |
| I3 | `token_indices` 升序、唯一、含 `n_tokens-1`、且每个 token 属于 `block_indices` | 扩展的不变量用例 |
| I4 | 结构锚点（sink/recent）整块保留 | 单测 |

## 2. 根因（实测）

多事实 hard 的失败不是预算问题（b=0.16 反而更差：0.771 vs 0.836），
而是**证据句子被块边界切断**。`--dump-selected-text` 直接给出证据：

```
[pos 3904] ... for vault six is TEAL-LYNX-8
[... 672 dropped tokens]
[pos 4608] . The secret passcode for vault seven is SILVER-PANTHER-8782. ...
```

`TEAL-LYNX-8` | `853`：前半块入选、后半块未入选。填充以块为单位 all-or-nothing，
所以 6 个事实里被切断 2 个 ≈ 0.709，与实测吻合。

## 3. 算法

### 3.1 句子映射（O(n)）

```text
for i in [0, n_tokens):
    ends = exists c in pieces[i]: c in ".!?;\n"
    sentence_end[i] = ends
// 反向扫描得到 sentence_start：
start[i] = (i == 0 || sentence_end[i-1]) ? i : start[i-1]
end[i]   = 第一个 j >= i 使 sentence_end[j] 为真（含 j），否则 n_tokens
```

注意：标点 token 属于**它结束的那句**。

### 3.2 句子完整 + 闭包

```text
keep = [0]*n
selected = 已有块集合
// 1) 结构块整块保留
for b in selected ∩ structure: keep[block(b)] = 1

// 2) 证据句子：块内所有 per_pos_scores>0 的 token
for b in selected（证据块）:
    for i in block(b) 且 per_pos[i] > 0:
        [s,e) = sentence(i)
        if e - s > span_max_tokens: 裁剪到块边界并 log_warn
        keep[s:e] = 1
        if e > block_end: 需要块 b_end = e / block_size 加入 selected（闭包）
        if s < block_begin: 需要块 s / block_size 加入 selected

// 3) 闭包：新加入的块也整体保留（仅当它没有任何证据 token；有证据则回到 2）
//    句子很短，最多迭代 2 次

// 4) 无证据块（stride/position 填充）按原样整块保留
```

实现上把 (2)(3) 合并成两遍循环：第一遍标记句子 token 并记录需要的块，
第二遍对新增块重复（最多 2 轮）。

### 3.3 裁剪

`token_indices = { i : i in 某个 selected 块 且 keep[i] }`。
被裁掉的 token 数记入 `span_tokens`。

关键点：裁剪只在**块内**发生，块集合（`block_indices`）保持
"证据 + 结构"语义，因此注意力池与 AR004/AR006 路径不受影响（I3）。

## 4. 接口

```cpp
// SelectionOptions
int sentence_completion = 1;     // 0 = AR011 行为
uint32_t span_max_tokens = 128;  // 单句上限（防御异常长句）

// Selection
uint32_t sentences_completed = 0;  // 补齐到句子边界的次数
uint32_t sentences_extended = 0;   // 因跨块闭包新增的块数
```

CLI：`--sentence-completion 0|1`、`--span-max-tokens N`；
raw.csv 追加 `sentences_completed,sentences_extended`；
`--dump-selected-text P`（FR-2）。

## 5. 与既有 span（AR011）的关系

AR011 的 `span_admission` 是"单事实下按句裁剪、多事实禁用"。
AR012 把语义统一为**句子完整**（先补齐、再裁剪），因此：

- `span_admission=1 + sentence_completion=1`：新语义（默认组合）；
- `sentence_completion=0`：回到 AR011 的逐块/逐窗行为（I1）；
- 多事实不再需要禁用 span：句子完整性正是它缺的东西。

## 6. 测试设计

| 用例 | 断言 |
|---|---|
| `selector_sentence_completion_keeps_split_answer` | 构造跨块口令，补齐后两段都在 `token_indices` |
| `selector_sentence_completion_off_is_ar011` | 0 时与 AR011 逐 token 一致（固定 fixture） |
| `selector_sentence_indices_invariants` | 升序/唯一/含末 token/属于 block_indices |
| `selector_sentence_completion_keeps_structure_whole` | sink/recent 整块 |
| `dump_selected_text_matches_indices` | dump 中的位置序列 == `token_indices` |

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| 补齐句子导致 token 增加 | 上限 1.10×，闸门检查；同时裁掉句子外的填充 |
| 闭包把无关块拉进来 | 只在句子跨块时补，最多 2 轮，审计 `sentences_extended` |
| 异常长句（无标点文档） | `span_max_tokens=128` 截断 + 警告；无正分 token 的块整块保留 |
| 多事实仍不达标 | 保持 AR011 组合，记录回退 |

## 8. 实施顺序

T001 句子映射 + 完整准入 → T002 审计/CLI/dump → T003 单测（I1 先绿）→
T004 矩阵 + 闸门 → T005 GPU → T006 文档 → T007 ST。
