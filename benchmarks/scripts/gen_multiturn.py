#!/usr/bin/env python3
"""Generate deterministic multi-turn sessions over a single document (AR014).

Each session plants `--facts-per-session` facts in one context and then asks for
a disjoint pair of them per turn, in document order. That models the edge case
the session cache exists for: several questions about the same document, where
every turn's evidence overlaps what earlier turns already put in the KV cache.

The index carries a sixth column (`session`) so the benchmark can keep one KV
sequence per value; rows of the same session must be adjacent and in turn order.

Deterministic for a fixed --seed; same layout as the other generators
(index.tsv + contexts/ + manifest.json with per-context SHA256).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from _common import base_parser, finish, insert_at_depth, make_context
from gen_multifact import ANIMALS, COLORS, NUMBER_WORDS, join_natural


def main() -> int:
    parser = base_parser("Generate multi-turn session cases")
    parser.add_argument("--facts-per-session", type=int, default=10)
    parser.add_argument("--turns", type=int, default=5,
                        help="questions asked per session (2 facts each)")
    parser.add_argument("--facts-per-turn", type=int, default=2)
    args = parser.parse_args()
    if args.facts_per_turn < 1 or args.facts_per_turn * args.turns > args.facts_per_session:
        raise SystemExit("--facts-per-turn * --turns must not exceed --facts-per-session")
    root = Path(args.out)
    rng = random.Random(args.seed)
    rows = []
    for session in range(args.cases):
        sentences = make_context(rng, args.words)
        codes: dict[int, str] = {}
        names: dict[int, str] = {}
        for k in range(args.facts_per_session):
            idx = session * args.facts_per_session + k
            color = COLORS[idx % len(COLORS)]
            animal = ANIMALS[(idx * 7) % len(ANIMALS)]
            codes[k + 1] = f"{color}-{animal}-{9200 + idx * 89}"
            names[k + 1] = NUMBER_WORDS[idx % len(NUMBER_WORDS)]
            depth = 0.06 + 0.88 * (k / max(1, args.facts_per_session - 1))
            sentences = insert_at_depth(
                sentences,
                f"The secret passcode for vault {names[k + 1]} is {codes[k + 1]}.",
                depth)
        context = " ".join(sentences)
        session_id = f"mt-{session + 1:02d}"
        rel = Path("contexts") / f"{session_id}.txt"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        for turn in range(args.turns):
            vaults = [turn * args.facts_per_turn + j + 1
                      for j in range(args.facts_per_turn)]
            question = (f"List the secret passcodes for the vaults "
                        f"{join_natural([names[v] for v in vaults])}.")
            answer = " ".join(codes[v] for v in vaults)
            rows.append({
                "id": f"{session_id}-t{turn + 1}",
                "task": "niah",
                "context_path": rel.as_posix(),
                "question": question,
                "answer": answer,
                "context_sha256": digest,
                "session": session_id,
            })
    finish(root, rows)
    print(f"wrote {len(rows)} multi-turn rows ({args.cases} sessions x {args.turns}) to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
