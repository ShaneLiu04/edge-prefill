#!/bin/bash
# Wait for run_experiments.sh to finish, then run phase 2 (head profile,
# len2k fix, head-adaptive sweeps) so timings never overlap.
while pgrep -f "run_experiments[.]sh" > /dev/null; do
  sleep 60
done
sleep 30
bash /mnt/d/prefill/scripts/phase2_experiments.sh
