#!/usr/bin/env python3
"""Generate deterministic needle-in-a-haystack cases (single line, one fact)."""

from __future__ import annotations

import random
from pathlib import Path

from _common import base_parser, finish, insert_at_depth, make_context, write_case

COLORS = ["MAGENTA", "COBALT", "AMBER", "VIOLET", "CRIMSON", "TEAL", "SILVER"]
ANIMALS = ["FALCON", "OTTER", "PANTHER", "HERON", "BISON", "LYNX", "RAVEN"]


def main() -> int:
    args = base_parser("Generate NIAH cases").parse_args()
    root = Path(args.out)
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.cases):
        color = COLORS[i % len(COLORS)]
        animal = ANIMALS[(i * 3) % len(ANIMALS)]
        code = f"{color}-{animal}-{7000 + i * 137}"
        fact = f"The secret passcode for vault {i + 1} is {code}."
        depth = 0.15 + 0.7 * ((i * 7) % args.cases) / max(1, args.cases - 1)
        sentences = insert_at_depth(make_context(rng, args.words), fact, depth)
        context = " ".join(sentences)
        case_id = f"niah-{i + 1:02d}"
        rows.append(write_case(
            root, case_id, "niah", context,
            f"What is the secret passcode for vault {i + 1}?", code))
    finish(root, rows)
    print(f"wrote {len(rows)} NIAH cases to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
