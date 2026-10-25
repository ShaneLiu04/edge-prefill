#!/usr/bin/env python3
"""Generate deterministic multi-needle-in-a-haystack cases.

Each context plants `--needles` distinct facts at spread depths and yields
one index row per fact (same context, different question), so a single
dataset measures multi-needle recall: the mean over rows is the recall
fraction, and a block selector that drops any single fact block shows up
as a row-level miss.

Deterministic for a fixed --seed; emitted files match `gen_niah.py`
(index.tsv + contexts/ + manifest.json with per-context SHA256).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from _common import base_parser, finish, insert_at_depth, make_context

COLORS = ["MAGENTA", "COBALT", "AMBER", "VIOLET", "CRIMSON", "TEAL", "SILVER"]
ANIMALS = ["FALCON", "OTTER", "PANTHER", "HERON", "BISON", "LYNX", "RAVEN"]


def main() -> int:
    parser = base_parser("Generate multi-needle NIAH cases")
    parser.add_argument("--needles", type=int, default=3,
                        help="distinct facts planted per context")
    args = parser.parse_args()
    if args.needles < 1:
        raise SystemExit("--needles must be >= 1")
    root = Path(args.out)
    rng = random.Random(args.seed)
    rows = []
    for case in range(args.cases):
        sentences = make_context(rng, args.words)
        facts = []
        for k in range(args.needles):
            idx = case * args.needles + k
            color = COLORS[idx % len(COLORS)]
            animal = ANIMALS[(idx * 3) % len(ANIMALS)]
            code = f"{color}-{animal}-{7100 + idx * 131}"
            depth = 0.1 + 0.8 * (k / max(1, args.needles - 1)) if args.needles > 1 else 0.5
            sentences = insert_at_depth(
                sentences, f"The secret passcode for vault {idx + 1} is {code}.", depth)
            facts.append((idx + 1, code))
        context = " ".join(sentences)
        case_id = f"mn-{case + 1:02d}"
        rel = Path("contexts") / f"{case_id}.txt"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        for vault, code in facts:
            rows.append({
                "id": f"{case_id}-n{vault}",
                "task": "niah",
                "context_path": rel.as_posix(),
                "question": f"What is the secret passcode for vault {vault}?",
                "answer": code,
                "context_sha256": digest,
            })
    finish(root, rows)
    print(f"wrote {len(rows)} multi-needle rows to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
