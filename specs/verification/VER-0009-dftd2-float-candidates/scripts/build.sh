#!/bin/bash
# build.sh NAME SRCROOT_INCLUDE_DIR(first) DFTD2_CU [extra nvcc flags]
SP=/tmp/claude-1001/-home-katouda-work-dielectric-response-pj-GPUMD/e8515210-3788-4777-872c-6503ac9338a6/scratchpad/p6c
S=/home/katouda/work/dielectric-response_pj/GPUMD/.claude/worktrees/d2-phase6c/src
N=$1; I=$2; CU=$3; shift 3
F="-std=c++14 -O3 -arch=sm_86 -I$I -I$S -I$S/force $@"
nvcc $F -c $SP/harness2.cu -o $SP/obj/h_$N.o || exit 1
nvcc $F -c $CU -o $SP/obj/d_$N.o -Xptxas -v 2>&1 | grep -i "error\|registers\|spill" | tail -3
nvcc $F -o $SP/bin/$N $SP/obj/h_$N.o $SP/obj/d_$N.o $SP/obj/box.o $SP/obj/error.o $SP/obj/run_input.o
