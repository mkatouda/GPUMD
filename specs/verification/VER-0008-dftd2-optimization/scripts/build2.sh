#!/bin/bash
# usage: build2.sh NAME path/to/dftd2.cu   -> bin/NAME (uses worktree headers)
P=/tmp/claude-1001/-home-katouda-work-dielectric-response-pj-GPUMD/e8515210-3788-4777-872c-6503ac9338a6/scratchpad/p6
S=/home/katouda/work/dielectric-response_pj/GPUMD/.claude/worktrees/d2-phase6b/src
F="-std=c++14 -O3 -arch=sm_86 -I$S -I$S/force"
nvcc $F -c $2 -o $P/obj/$1.o -Xptxas -v 2>&1 | grep -i "error\|registers\|spill" | tail -4
nvcc $F -o $P/bin/$1 $P/obj/$1.o $P/obj/harness.o $P/obj/box.o $P/obj/error.o $P/obj/run_input.o
