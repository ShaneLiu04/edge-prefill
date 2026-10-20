#!/usr/bin/env python3
"""Generate deterministic extractive QA cases from synthetic world facts."""

from __future__ import annotations

import random
from pathlib import Path

from _common import base_parser, finish, insert_at_depth, make_context, write_case

CITIES = ["Kelvara", "Norhaven", "Brastholm", "Dunmoor", "Pellmark", "Ostrend", "Veyra"]
RIVERS = ["Ardan", "Lumen", "Sable", "Corrin", "Mistral", "Voran", "Elska"]


def main() -> int:
    args = base_parser("Generate extractive QA cases").parse_args()
    root = Path(args.out)
    rng = random.Random(args.seed + 1)
    rows = []
    for i in range(args.cases):
        city = CITIES[i % len(CITIES)]
        river = RIVERS[(i * 2) % len(RIVERS)]
        fact = f"The capital city of Zorbonia province {i + 1} is {city}, built beside the {river} river."
        depth = 0.2 + 0.6 * ((i * 5) % args.cases) / max(1, args.cases - 1)
        sentences = insert_at_depth(make_context(rng, args.words), fact, depth)
        context = " ".join(sentences)
        case_id = f"qa-{i + 1:02d}"
        rows.append(write_case(
            root, case_id, "qa", context,
            f"What is the capital city of Zorbonia province {i + 1}?", city))
    finish(root, rows)
    print(f"wrote {len(rows)} QA cases to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
