#!/bin/bash
# AR004 FR-7 / AR005 FR-5: fetch benchmark models with SHA256 verification.
#
# Modes:
#   download_model.sh [OUT_DIR]            main model (Qwen2.5-1.5B-Instruct q4_k_m)
#   download_model.sh --draft [OUT_DIR]    draft model (Qwen2.5-0.5B-Instruct q4_k_m)
#   download_model.sh --embed [OUT_DIR]    embedding model (bge-small-en-v1.5 q8_0)
#   download_model.sh --all [OUT_DIR]      all three
#
# Draft fallback: --draft-model may also point at the main model file
# itself (same vocab family, so LlamaDraftScorer accepts it) at the cost
# of a slower draft forward; the 0.5B download is the fast path.
#
# Tries ModelScope first (best inside CN networks), then hf-mirror.com.
# A hash mismatch aborts the download so no unverified model ever
# produces a benchmark row.
set -eu

MAIN_NAME=qwen2.5-1.5b-instruct-q4_k_m.gguf
MAIN_SHA=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
MAIN_URLS=(
  "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/$MAIN_NAME"
  "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$MAIN_NAME"
)

DRAFT_NAME=qwen2.5-0.5b-instruct-q4_k_m.gguf
DRAFT_SHA=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
DRAFT_URLS=(
  "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/$DRAFT_NAME"
  "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/$DRAFT_NAME"
)

EMBED_NAME=bge-small-en-v1.5-q8_0.gguf
EMBED_SHA=f046db1dc724cf4f6f0a0c5917e922823b73eb1d27b8f9a9c2797f7866974804
EMBED_URLS=(
  "https://modelscope.cn/models/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/master/$EMBED_NAME"
  "https://hf-mirror.com/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/$EMBED_NAME"
)

# fetch <name> <sha256> <urls...> <outdir>
fetch() {
  local name=$1 sha=$2 outdir=$3; shift 3
  local out="$outdir/$name"
  mkdir -p "$outdir"
  if [ -f "$out" ]; then
    echo "exists: $out"
  else
    local ok=0 url
    for url in "$@"; do
      echo "trying: $url"
      # curl | apt install curl if missing; fall back to wget
      if command -v curl > /dev/null 2>&1; then
        if curl -L --fail --connect-timeout 20 -o "$out.part" "$url"; then ok=1; fi
      elif command -v wget > /dev/null 2>&1; then
        if wget -q --timeout=20 -O "$out.part" "$url"; then ok=1; fi
      else
        echo "error: need curl or wget" >&2
        exit 3
      fi
      if [ "$ok" = 1 ]; then break; fi
      echo "unreachable, trying next mirror"
    done
    if [ "$ok" != 1 ]; then
      echo "error: all mirrors failed for $name" >&2
      rm -f "$out.part"
      exit 2
    fi
    mv "$out.part" "$out"
  fi

  echo "verifying SHA256 ($name)..."
  local actual
  actual=$(sha256sum "$out" | awk '{print $1}')
  if [ "$actual" != "$sha" ]; then
    echo "error: SHA256 mismatch for $name" >&2
    echo "  expected: $sha" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi
  echo "OK: $out ($sha)"
}

MODE=main
OUT_DIR=models
for arg in "$@"; do
  case "$arg" in
    --draft) MODE=draft ;;
    --embed) MODE=embed ;;
    --all)   MODE=all ;;
    *)       OUT_DIR=$arg ;;
  esac
done

case "$MODE" in
  main)
    fetch "$MAIN_NAME" "$MAIN_SHA" "$OUT_DIR" "${MAIN_URLS[@]}"
    echo "export EDGE_PREFILL_TEST_MODEL=$(cd "$OUT_DIR" && pwd)/$MAIN_NAME"
    ;;
  draft)
    fetch "$DRAFT_NAME" "$DRAFT_SHA" "$OUT_DIR" "${DRAFT_URLS[@]}"
    echo "export EDGE_PREFILL_DRAFT_MODEL=$(cd "$OUT_DIR" && pwd)/$DRAFT_NAME"
    ;;
  embed)
    fetch "$EMBED_NAME" "$EMBED_SHA" "$OUT_DIR" "${EMBED_URLS[@]}"
    echo "export EDGE_PREFILL_TEST_EMBEDDING=$(cd "$OUT_DIR" && pwd)/$EMBED_NAME"
    ;;
  all)
    fetch "$MAIN_NAME" "$MAIN_SHA" "$OUT_DIR" "${MAIN_URLS[@]}"
    echo "export EDGE_PREFILL_TEST_MODEL=$(cd "$OUT_DIR" && pwd)/$MAIN_NAME"
    fetch "$DRAFT_NAME" "$DRAFT_SHA" "$OUT_DIR" "${DRAFT_URLS[@]}"
    echo "export EDGE_PREFILL_DRAFT_MODEL=$(cd "$OUT_DIR" && pwd)/$DRAFT_NAME"
    fetch "$EMBED_NAME" "$EMBED_SHA" "$OUT_DIR" "${EMBED_URLS[@]}"
    echo "export EDGE_PREFILL_TEST_EMBEDDING=$(cd "$OUT_DIR" && pwd)/$EMBED_NAME"
    ;;
esac
