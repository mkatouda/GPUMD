#!/bin/bash
# usage: ptxas_report.sh REPO_ROOT SCRATCH   -> registers/spills of the D3 kernels (sm_86) and an occupancy estimate
set -e
cd $1/src
/usr/local/cuda/bin/nvcc -std=c++14 -O3 -arch=sm_86 -Xptxas -v -I./ -c force/dftd3.cu -o $2/dftd3_ptxas.o > $2/ptxas.log 2>&1
python3 - $2/ptxas.log <<'PY'
import re, sys
cur = None
info = {}
for l in open(sys.argv[1]):
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m:
        cur = m.group(1)
        info[cur] = {}
    if cur and 'spill stores' in l:
        info[cur]['spill'] = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', l).groups()
    if cur and 'Used' in l:
        info[cur]['regs'] = int(re.search(r'Used (\d+) registers', l).group(1))
        m = re.search(r'(\d+) bytes smem', l)
        info[cur]['smem'] = int(m.group(1)) if m else 0
for k, v in info.items():
    if 'dftd3' in k or 'small_box' in k:
        name = re.sub(r'^_ZN\d+_GLOBAL__N__\w+?_dftd3_cu_\w{8}\d+', '', k)
        name = re.sub(r'^_ZN\d+_GLOBAL__N__\w+?_dftd3_cu_\w{8}', '', name).split('EN5DFTD3')[0].split('EffiE')[0][:48]
        r, sm = v['regs'], v['smem']
        print(f"{name:48s} regs={r:3d} smem={sm:5d} B spill_st/ld={v['spill'][0]}/{v['spill'][1]} B")
PY
