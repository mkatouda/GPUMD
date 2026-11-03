"""D3 extra time per step by the VER-0001 slope method (100 vs 500 steps), baseline and candidate binary.

usage: timing.py BIN_TAG=PATH[,BIN_TAG=PATH...] NEP_TXT WATER27_XYZ WORKDIR OUT_JSON
Per-step time = (T(500) - T(100)) / 400, T = 'Time used' of the whole gpumd process (min of 2 runs).
D3 extra = (run with `dftd3 pbe RC CN`) - (run without D3).  NVE, dt 0.5 fs, velocity 300 K seed 1.
Cases: water N = 81, 648, 5184, 41472 (rc 12, cn 6); Cu fcc 4000 (rc 12, cn 6);
cutoff scan rc = 8, 12, 16, 20 (cn 6) for water 648 and 5184.
"""
import json, re, shutil, subprocess, sys
from pathlib import Path
from ase.io import read, write
from ase.build import bulk

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


cases = [('water', r, 12) for r in (1, 2, 4, 8)] + [('Cu', 10, 12)] + \
        [('water', r, rc) for r in (2, 4) for rc in (8, 16, 20)]
res = []
off_cache = {}
for kind, rep, rc in cases:
    at = water.repeat((rep,) * 3) if kind == 'water' else bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((rep,) * 3)
    n = len(at)
    for tag, b in bins.items():
        key = (tag, n)
        if key not in off_cache:
            off_cache[key] = slope(b, at, [])
        on = slope(b, at, [f'dftd3 pbe {rc} 6'])
        r = dict(system=kind, N=n, rc=rc, cn=6, bin=tag, off_ms=off_cache[key], on_ms=on, extra_ms=on - off_cache[key])
        res.append(r)
        print(json.dumps(r), flush=True)
json.dump(res, open(out, 'w'), indent=1)
