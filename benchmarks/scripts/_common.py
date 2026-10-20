"""Shared deterministic dataset helpers for EdgePrefill benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

WORDS = (
    "river valley stone forest window garden market bridge candle silver "
    "morning summer winter travel letter number music island mountain harbor "
    "engine signal station platform railway factory village castle meadow "
    "shadow thunder weather harvest orchard pottery lantern compass anchor "
    "voyage desert glacier canyon prairie cavern tunnel beacon ferry cargo "
    "timber marble copper iron cotton wool spice coffee cocoa pepper salt "
    "sailor painter builder weaver farmer hunter shepherd scholar teacher "
    "captain pilot driver guard keeper watcher seeker runner walker rider"
).split()

CONNECTORS = (
    "and", "while", "because", "although", "after", "before", "near", "beside",
    "across", "through", "toward", "around", "under", "above", "beyond", "within",
)


def make_sentence(rng: random.Random) -> str:
    n = rng.randint(7, 15)
    parts = []
    for i in range(n):
        if i > 0 and rng.random() < 0.22:
            parts.append(rng.choice(CONNECTORS))
        parts.append(rng.choice(WORDS))
    text = " ".join(parts)
    return text[0].upper() + text[1:] + "."


def make_context(rng: random.Random, words: int) -> list[str]:
    sentences = []
    count = 0
    while count < words:
        s = make_sentence(rng)
        sentences.append(s)
        count += len(s.split())
    return sentences


def insert_at_depth(sentences: list[str], fact: str, depth: float) -> list[str]:
    idx = min(len(sentences), max(0, int(len(sentences) * depth)))
    return sentences[:idx] + [fact] + sentences[idx:]


def write_case(root: Path, case_id: str, task: str, context: str, question: str,
               answer: str) -> dict:
    rel = Path("contexts") / f"{case_id}.txt"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(context, encoding="utf-8", newline="\n")
    return {
        "id": case_id,
        "task": task,
        "context_path": rel.as_posix(),
        "question": question,
        "answer": answer,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }


def finish(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = ["id\ttask\tcontext_path\tquestion\tanswer"]
    for row in rows:
        fields = [row["id"], row["task"], row["context_path"], row["question"], row["answer"]]
        for field in fields:
            if "\t" in field or "\n" in field:
                raise ValueError("index fields must be single-line and tab-free")
        lines.append("\t".join(fields))
    (root / "index.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "cases": len(rows),
        "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "rows": rows,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        newline="\n")


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out", required=True, help="output dataset directory")
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--words", type=int, default=6000,
                        help="approximate filler word count per context")
    parser.add_argument("--seed", type=int, default=20261020)
    return parser
