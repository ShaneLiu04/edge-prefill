#!/bin/bash
# Phase 4 watcher: start the deep-dive CPU matrix once the GPU baseline
# (phase 3) has finished, keeping all timing windows mutually exclusive.
while pgrep -f "phase3_gpu[.]sh" > /dev/null; do
  sleep 60
done
sleep 30
bash /mnt/d/prefill/scripts/phase4_experiments.sh
