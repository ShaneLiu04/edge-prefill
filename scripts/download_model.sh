#!/bin/bash
# AR004 FR-7 / T014: fetch the benchmark model with SHA256 verification.
# Tries ModelScope first (best inside CN networks), then hf-mirror.com.
# Both hosts must serve the exact q4_k_m GGUF we report numbers for;
# a hash mismatch aborts the download so no unverified model ever
# produces a benchmark row.
set -eu

MODEL_NAME=qwen2.5-1.5b-instruct-q4_k_m.gguf
SHA256_EXPECTED=6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
OUT_DIR=${1:-models}
OUT="$OUT_DIR/$MODEL_NAME"
mkdir -p "$OUT_DIR"

URLS=(
  "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/$MODEL_NAME"
  "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$MODEL_NAME"
)

if [ -f "$OUT" ]; then
  echo "exists: $OUT"
else
  ok=0
  for url in "${URLS[@]}"; do
    echo "trying: $url"
    # curl | apt install curl if missing; fall back to wget
    if command -v curl > /dev/null 2>&1; then
      if curl -L --fail --connect-timeout 20 -o "$OUT.part" "$url"; then ok=1; fi
    elif command -v wget > /dev/null 2>&1; then
      if wget -q --timeout=20 -O "$OUT.part" "$url"; then ok=1; fi
    else
      echo "error: need curl or wget" >&2
      exit 3
    fi
    if [ "$ok" = 1 ]; then break; fi
    echo "unreachable, trying next mirror"
  done
  if [ "$ok" != 1 ]; then
    echo "error: all mirrors failed" >&2
    rm -f "$OUT.part"
    exit 2
  fi
  mv "$OUT.part" "$OUT"
fi

echo "verifying SHA256..."
actual=$(sha256sum "$OUT" | awk '{print $1}')
if [ "$actual" != "$SHA256_EXPECTED" ]; then
  echo "error: SHA256 mismatch" >&2
  echo "  expected: $SHA256_EXPECTED" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "OK: $OUT ($SHA256_EXPECTED)"
echo "export EDGE_PREFILL_TEST_MODEL=$(cd "$OUT_DIR" && pwd)/$MODEL_NAME"
