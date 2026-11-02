#!/bin/bash
# usage: alt.sh newbinary "systems"
S=/tmp/claude-1001/-home-katouda-work-dielectric-response-pj-GPUMD/e8515210-3788-4777-872c-6503ac9338a6/scratchpad/p5
for r in 1 2; do
  bash $S/run.sh base gpumd_base $2
  bash $S/run.sh new $1 $2
done 2>&1 | grep "${3:-force_large|SUM}" -E
