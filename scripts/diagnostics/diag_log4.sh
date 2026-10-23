#!/bin/bash
for f in /tmp/r.sh /tmp/r4.sh /tmp/ra.sh /tmp/w.sh; do
  echo "== $f (mtime $(stat -c %y "$f" 2>/dev/null))"
  head -12 "$f" 2>/dev/null
done
