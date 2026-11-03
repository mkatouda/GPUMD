#!/bin/bash
# usage: build_variants.sh REPO_ROOT SCRATCH [VARIANT...]
# Copies REPO_ROOT/src into SCRATCH/src (the repository is not modified), builds one gpumd binary per
# timing-only variant of src/force/dftd3.cu with `make gpumd`, and stores it as SCRATCH/bin/gpumd_<variant>.
set -e
REPO=$1; S=$2; shift 2
VARS=${@:-base noc6 f32atomic nlf32 n1only regacc empty count}
mkdir -p $S/bin
[ -d $S/src ] || cp -a $REPO/src $S/src
cp $REPO/src/force/dftd3.cu $S/dftd3_orig.cu
cd $S/src
for v in $VARS; do
  python3 $REPO/specs/verification/VER-0012-dftd3-small-box-diagnostic/scripts/patch_variants.py $S/dftd3_orig.cu $v force/dftd3.cu
  make CUDA_ARCH="-arch=sm_86" gpumd > $S/build_$v.log 2>&1
  grep -E "warning:|error:" $S/build_$v.log | head -5 || true
  cp gpumd $S/bin/gpumd_$v
  echo built $v
done
cp $S/dftd3_orig.cu force/dftd3.cu
