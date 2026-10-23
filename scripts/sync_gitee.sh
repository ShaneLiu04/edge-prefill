#!/bin/bash
# AR004 FR-7 / T014: push the repository to Gitee.
# Verifies the working tree is clean and the pinned llama.cpp commit and
# model SHA256 are present in the README before pushing, so the public
# mirror always points at a reproducible state.
set -eu
cd "$(dirname "$0")/.."

REMOTE=${1:-origin}
BRANCH=${2:-master}

if [ -n "$(git status --porcelain)" ]; then
  echo "error: uncommitted changes; commit first" >&2
  git status --short >&2
  exit 1
fi

for pin in "3bb63cbcbfb62056c84fe9e521b0f3d756d29c32" \
           "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"; do
  if ! grep -q "$pin" README.md; then
    echo "error: pin $pin missing from README.md" >&2
    exit 2
  fi
done

git push "$REMOTE" "$BRANCH"
echo "pushed $BRANCH -> $REMOTE"
