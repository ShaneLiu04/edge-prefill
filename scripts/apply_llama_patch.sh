#!/usr/bin/env bash
set -euo pipefail

# Run from the llama.cpp source root (FetchContent PATCH_COMMAND cwd).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_dir="${here}/../patches"
patch_ffn="${patch_dir}/llama-prefill-ffn-skip.patch"
patch_attn="${patch_dir}/llama-prefill-attn-top-k.patch"

apply_patch() {
    local p="$1"
    if [[ ! -f "${p}" ]]; then
        echo "[edge-prefill] missing patch: ${p}" >&2
        exit 1
    fi
    if command -v patch >/dev/null 2>&1; then
        patch -p1 --forward < "${p}"
    else
        git apply --whitespace=nowarn "${p}"
    fi
}

# Patch 1: phase-asymmetric FFN skip + sparse position gaps.
if ! grep -q "llama_set_prefill_mode" include/llama.h 2>/dev/null; then
    apply_patch "${patch_ffn}"
    echo "[edge-prefill] applied llama-prefill-ffn-skip.patch"
fi

# Patch 2: attention-level top-k gather.
if ! grep -q "llama_set_prefill_attn_top_k" include/llama.h 2>/dev/null; then
    apply_patch "${patch_attn}"
    echo "[edge-prefill] applied llama-prefill-attn-top-k.patch"
fi

