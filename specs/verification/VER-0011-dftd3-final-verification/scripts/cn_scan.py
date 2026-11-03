"""Dependence of the D3 extra time on the coordination-number cutoff (rc = 12 A fixed), slope method as timing.py.

usage: cn_scan.py BIN_TAG=PATH[,BIN_TAG=PATH...] NEP_TXT WATER27_XYZ WORKDIR OUT_JSON
Per-step time = (T(500) - T(100)) / 400, T = 'Time used' of the whole gpumd process (min of 2 runs).
D3 extra = (run with `dftd3 pbe RC CN`) - (run without D3).  NVE, dt 0.5 fs, velocity 300 K seed 1.
Cases: water N = 648 and 5184, rc = 12, cn = 3, 6, 12.
"""
import json, re, shutil, subprocess, sys
from pathlib import Path
from ase.io import read, write

bins = dict(kv.split('=') for kv in sys.argv[1].split(','))
nep, w27, work, out = sys.argv[2:6]
water = read(w27)
cnt = [0]


def T(binary, atoms, extra, steps):
    cnt[0] += 1
    d = Path(work) / f't{cnt[0]}'
    d.mkdir(parents=True)
    write(d / 'model.xyz', atoms, format='extxyz')
    L = [f'potential {nep}'] + extra + ['velocity 300 seed 1', 'ensemble nve', 'time_step 0.5', f'run {steps}']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    o = subprocess.run([binary], cwd=d, capture_output=True, text=True, check=True).stdout
    shutil.rmtree(d)
    return float(re.search(r'Time used = ([0-9.]+) s', o).group(1))


def slope(binary, atoms, extra):
    t1 = min(T(binary, atoms, extra, 100) for _ in range(2))
    t2 = min(T(binary, atoms, extra, 500) for _ in range(2))
    return 1e3 * (t2 - t1) / 400


res = []
for rep in (2, 4):
    at = water.repeat((rep,) * 3)
    n = len(at)
    for tag, b in bins.items():
        off = slope(b, at, [])
        for cn in (3, 6, 12):
            on = slope(b, at, [f'dftd3 pbe 12 {cn}'])
            r = dict(system='water', N=n, rc=12, cn=cn, bin=tag, off_ms=off, on_ms=on, extra_ms=on - off)
            res.append(r)
            print(json.dumps(r), flush=True)
json.dump(res, open(out, 'w'), indent=1)
