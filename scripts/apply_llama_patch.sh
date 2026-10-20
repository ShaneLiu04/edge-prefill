#!/usr/bin/env bash
set -euo pipefail

# Run from the llama.cpp source root (FetchContent PATCH_COMMAND cwd).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${here}/../patches/llama-prefill-ffn-skip.patch"

if grep -q "llama_set_prefill_mode" include/llama.h 2>/dev/null; then
    echo "[edge-prefill] llama.cpp patch already applied"
    exit 0
fi

if [[ ! -f "${patch_file}" ]]; then
    echo "[edge-prefill] missing patch: ${patch_file}" >&2
    exit 1
fi

if command -v patch >/dev/null 2>&1; then
    patch -p1 --forward < "${patch_file}"
else
    git apply --whitespace=nowarn "${patch_file}"
fi
echo "[edge-prefill] applied llama-prefill-ffn-skip.patch"
