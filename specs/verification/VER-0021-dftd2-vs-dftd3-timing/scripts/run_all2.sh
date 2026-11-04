#!/bin/sh
# usage: run_all2.sh REPO SCRATCH  (re-run of gate points, then cutoff scan)
R=$1; S=$2; D=$(dirname $0)/..
P=/home/katouda/miniforge3/envs/py312-gpumd/bin/python
$P $D/scripts/ratio.py $R $R/src/gpumd $S/w_re $D/results/ratio_rerun.json --set main --only w81,w648,cu4000 > $D/results/ratio_rerun.log 2>&1
$P $D/scripts/ratio.py $R $R/src/gpumd $S/w_cut $D/results/ratio_cut.json --set cut > $D/results/ratio_cut.log 2>&1
