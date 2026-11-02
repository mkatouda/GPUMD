#!/bin/bash
# usage: run.sh tag binary xyz...
S=/tmp/claude-1001/-home-katouda-work-dielectric-response-pj-GPUMD/e8515210-3788-4777-872c-6503ac9338a6/scratchpad/p5
PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python
tag=$1; bin=$2; shift 2
for x in "$@"; do $PY $S/bench.py $bin $tag $x.xyz; done
