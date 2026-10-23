#!/bin/bash
for f in /tmp/diag-k62q-new.log /tmp/diag-selq-new.log /tmp/diag-k24q-new.log; do
  echo "=== $f ==="
  grep -a "generated=\|passcode\|MAGENTA\|spider\|vault 1 is" "$f" | head -3
  echo ""
done
