#!/usr/bin/env python3
"""Dissect the attention-gather selection dump for the niah 8K case.

Answers three questions:
1. Which block holds the needle, and what is its fused rank?
2. Did the selection keep that block? Did the row gather keep its rows?
3. What fraction of kept blocks' rows survived the gather, per rank band?
"""
import json
import os
import sys

RESULTS = r"D:\prefill\benchmarks\results"
CTX = r"D:\prefill\benchmarks\data\niah\contexts\niah-01.txt"

path = os.path.join(RESULTS, "exp-sel-dump-attn", "niah-01@7961.json")
with open(path) as fh:
    sel = json.load(fh)
with open(CTX, errors="ignore") as fh:
    text = fh.read()

total = sel["total_tokens"]
block = 32
needle_off = text.find("MAGENTA-FALCON")
frac = needle_off / len(text)
needle_token = int(frac * total)
needle_block = needle_token // block
print(f"total_tokens={total} needle_byte={needle_off}/{len(text)} "
      f"needle_token~{needle_token} needle_block={needle_block}")

signals = sel["signals"]
order = sorted(range(len(signals)), key=lambda i: -signals[i]["fused"])
rank = {b: r for r, b in enumerate(order)}
sig = signals[needle_block]
print(f"needle block fused={sig['fused']:.4f} lexical={sig['lexical']:.4f} "
      f"rank={rank[needle_block]}")

kept = set(sel["block_indices"])
mand = set(sel["mandatory_blocks"])
print(f"blocks total={len(signals)} kept={len(kept)} mandatory={sorted(mand)}")
print(f"needle block kept={needle_block in kept} mandatory={needle_block in mand}")

attn_idx = sel.get("attention_indices") or []
print(f"attn_top_k={sel.get('attention_top_k')} attention_indices={len(attn_idx)}")
if attn_idx:
    needle_rows = [i for i in attn_idx if i // block == needle_block]
    print(f"needle rows gathered: {len(needle_rows)} / {block}")
    print(f"last token {total-1} gathered: {total-1 in set(attn_idx)}")
    kept_rows = sum(1 for i in attn_idx if i // block in kept)
    print(f"gather rows inside kept blocks: {kept_rows}/{len(attn_idx)}")
    import collections
    bands = collections.Counter(i // block for i in attn_idx)
    top20 = order[:20]
    covered = sum(1 for b in top20 if bands.get(b, 0) > 0)
    print(f"top-20 fused blocks with >=1 gathered row: {covered}/20")
    for b in order[:8]:
        print(f"  block {b:4d} fused={signals[b]['fused']:.4f} "
              f"mand={b in mand} rows_gathered={bands.get(b, 0)}")
