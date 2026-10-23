#!/bin/bash
for f in /tmp/diag-fix1.log /tmp/diag-sel-only.log /tmp/diag-k62-mc.log /tmp/diag-sel-mc.log; do
  echo "=== $f ==="
  grep -a "generated" "$f" | head -2
  tail -3 "$f" | grep -av "^\s*$"
  echo ""
done
echo "=== the needle (expected secret) ==="
grep -ao "secret[^.]*" /tmp/diag-prompt.txt | head -3
grep -ao "magic[^.]*" /tmp/diag-prompt.txt | head -3
