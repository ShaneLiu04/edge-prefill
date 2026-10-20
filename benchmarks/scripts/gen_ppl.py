#!/usr/bin/env python3
"""Generate continuation cases for answer-log-probability (PPL proxy) quality."""

from __future__ import annotations

import random
from pathlib import Path

from _common import base_parser, finish, make_context, write_case


def main() -> int:
    args = base_parser("Generate PPL continuation cases").parse_args()
    root = Path(args.out)
    rng = random.Random(args.seed + 2)
    rows = []
    for i in range(args.cases):
        start = i + 1
        sequence = ", ".join(str(start + k) for k in range(4))
        expected = " " + str(start + 4)
        tail = f"Sequence puzzle {i + 1}: {sequence},"
        context = " ".join(make_context(rng, args.words)) + " " + tail
        case_id = f"ppl-{i + 1:02d}"
        rows.append(write_case(
            root, case_id, "ppl", context,
            f"Continue sequence puzzle {i + 1} with the next number.", expected))
    finish(root, rows)
    print(f"wrote {len(rows)} PPL cases to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
