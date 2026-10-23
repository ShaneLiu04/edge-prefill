#!/bin/bash
for d in /mnt/d/prefill/specs/changes/AR*; do
  echo "== $d  (mtime $(stat -c %y "$d" | cut -d' ' -f1))"
  ls "$d"
  grep -a "门控结果" "$d/tasks.md" 2>/dev/null | tail -3
  grep -a "status:" "$d/tasks.md" 2>/dev/null | sort | uniq -c | head -6
  echo ""
done
