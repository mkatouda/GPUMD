"""Slope-method timing of the D2 and D3 terms (same as the VER-0006 script; the rc scan at N = 648 removed).

usage: timing.py GPUMD_BIN NEP_TXT WATER27_XYZ WORKDIR
Per-step time = (T(n2) - T(n1)) / (n2 - n1), T = 'Time used' of the whole gpumd process (min of 2 runs),
which removes the fixed start-up cost (VER-0001 method).  n1, n2 are chosen from a pilot slope so that
the long run takes about 20 s (100/500 steps for the small cases).  NVE, dt 0.5 fs, velocity 300 K.
"""
import json, re, shutil, subprocess, sys
from pathlib import Path
from ase.io import read, write

gpumd, nep, w27, work = sys.argv[1:5]
base = read(w27)
cnt = [0]


def T(atoms, extra, steps):
    cnt[0] += 1
    d = Path(work) / f't{cnt[0]}'
    d.mkdir(parents=True)
    write(d / 'model.xyz', atoms, format='extxyz')
    L = [f'potential {nep}'] + extra + ['velocity 300 seed 1', 'ensemble nve', 'time_step 0.5', f'run {steps}']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    o = subprocess.run([gpumd], cwd=d, capture_output=True, text=True, check=True).stdout
    shutil.rmtree(d)
    return float(re.search(r'Time used = ([0-9.]+) s', o).group(1))


def slope(atoms, extra):
    p = (T(atoms, extra, 15) - T(atoms, extra, 5)) / 10
    n2 = int(min(500, max(20, 20.0 / max(p, 1e-4))))
    n2 = max(n2, 10)
    n1 = max(3, n2 // 5)
    t1 = min(T(atoms, extra, n1) for _ in range(2))
    t2 = min(T(atoms, extra, n2) for _ in range(2))
    return 1e3 * (t2 - t1) / (n2 - n1), n1, n2


out = []
cases = [(1, 'all')] + [(2, 'all')] + [(4, 'all')] + [(8, 'all')]
for rep, _ in cases:
    at = base.repeat((rep,) * 3)
    n = len(at)
    row = dict(N=n)
    for tag, extra in [('off', []), ('d3_pbe_12_6', ['dftd3 pbe 12 6']), ('d2_rc12', ['dftd2 0.75 12']),
                       ('d2_rc30', ['dftd2 0.75 30'])]:
        ms, n1, n2 = slope(at, extra)
        row[tag] = dict(ms_per_step=ms, n1=n1, n2=n2)
        print(n, tag, f'{ms:.4f} ms/step (n1={n1}, n2={n2})', flush=True)
    out.append(row)
json.dump(dict(sizes=out), open(Path(work) / 'timing.json', 'w'), indent=1)
