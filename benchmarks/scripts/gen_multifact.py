#!/usr/bin/env python3
"""Generate deterministic multi-fact (multi-hop-free) retrieval cases.

Each context plants `--facts` distinct facts. Every index row asks for
**several** of them at once ("List the passcodes for vault 1, vault 3 and
vault 5"), so a selector that only keeps the strongest lexical match provably
fails: the answer needs one block per fact. The row-level quality is the
fraction of requested codes that appear in the generated answer, which makes
partial credit visible instead of binary.

The single-fact family (`gen_multineedle.py`) cannot measure this: it asks one
vault per row, so a top-1 selector scores 1.0 whenever it happens to pick the
right block.

Deterministic for a fixed --seed; same file layout as the other generators
(index.tsv + contexts/ + manifest.json with per-context SHA256).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from _common import base_parser, finish, insert_at_depth, make_context

# Fact keys must (a) not collide with the passcodes and (b) survive the
# tokenizer as a single whole word. Digits fail (a): "8491" tokenises into
# "849" + "1", so a vault *number* leaks into the answer strings. Rare
# uppercase codenames fail (b): "AURORA" splits into "AUR" + "ORA" and the
# discriminator never reaches the selector as a term. Spelled-out numbers
# satisfy both: they are single tokens, absent from the filler vocabulary and
# absent from the passcodes.
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve"]
COLORS = ["MAGENTA", "COBALT", "AMBER", "VIOLET", "CRIMSON", "TEAL", "SILVER",
          "INDIGO", "SCARLET", "OLIVE"]
ANIMALS = ["FALCON", "OTTER", "PANTHER", "HERON", "BISON", "LYNX", "RAVEN",
           "JACKAL", "MARLIN", "CONDOR"]


def join_natural(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def main() -> int:
    parser = base_parser("Generate multi-fact retrieval cases")
    parser.add_argument("--facts", type=int, default=6,
                        help="distinct facts planted per context")
    parser.add_argument("--asked", type=int, default=3,
                        help="facts asked per row (must be <= --facts)")
    parser.add_argument("--rows-per-context", type=int, default=3,
                        help="questions per context (each asks different facts)")
    args = parser.parse_args()
    if args.asked < 2 or args.asked > args.facts:
        raise SystemExit("--asked must be in [2, --facts]")
    root = Path(args.out)
    rng = random.Random(args.seed)
    rows = []
    for case in range(args.cases):
        sentences = make_context(rng, args.words)
        codes: dict[int, str] = {}
        names: dict[int, str] = {}
        for k in range(args.facts):
            idx = case * args.facts + k
            color = COLORS[idx % len(COLORS)]
            animal = ANIMALS[(idx * 7) % len(ANIMALS)]
            codes[k + 1] = f"{color}-{animal}-{8200 + idx * 97}"
            names[k + 1] = NUMBER_WORDS[idx % len(NUMBER_WORDS)]
            depth = 0.08 + 0.84 * (k / max(1, args.facts - 1))
            sentences = insert_at_depth(
                sentences,
                f"The secret passcode for vault {names[k + 1]} is {codes[k + 1]}.",
                depth)
        context = " ".join(sentences)
        case_id = f"mf-{case + 1:02d}"
        rel = Path("contexts") / f"{case_id}.txt"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        # Ask for disjoint fact subsets so every row needs its own blocks.
        for r in range(args.rows_per_context):
            start = (r * args.asked) % args.facts
            vaults = sorted({((start + j) % args.facts) + 1 for j in range(args.asked)})
            if len(vaults) < args.asked:
                continue
            question = (f"List the secret passcodes for the vaults "
                        f"{join_natural([names[v] for v in vaults])}.")
            answer = " ".join(codes[v] for v in vaults)
            rows.append({
                "id": f"{case_id}-r{r + 1}",
                "task": "niah",
                "context_path": rel.as_posix(),
                "question": question,
                "answer": answer,
                "context_sha256": digest,
            })
    finish(root, rows)
    print(f"wrote {len(rows)} multi-fact rows to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
