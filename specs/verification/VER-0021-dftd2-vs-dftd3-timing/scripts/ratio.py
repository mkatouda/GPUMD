"""AC-7 re-measurement: D2 extra vs D3 extra per step, slope method (100 vs 500 steps, VER-0018 slope.py style).
usage: ratio.py REPO_ROOT GPUMD_BIN WORKDIR OUT.json [--set main|cut] [--reps 3]
Systems: water27 (tests_pytest/fixtures/dftd3/water27.xyz) repeated (r,r,r), r=1,2,4,8 (N=81,648,5184,41472) as VER-0010;
Cu fcc a=3.61 A 10x10x10 (4000 atoms) as VER-0018.  NEP89; NVE; dt 0.5 fs; velocity 300 seed 1.
Each repetition measures all variants in sequence (alternating); extra = variant slope - 'off' slope of the same repetition; result = median."""
import argparse, json, re, shutil, statistics, subprocess
from pathlib import Path
from ase.io import read, write
from ase.build import bulk

p = argparse.ArgumentParser()
p.add_argument('repo'); p.add_argument('gpumd'); p.add_argument('work'); p.add_argument('out')
p.add_argument('--set', default='main'); p.add_argument('--only', default=''); p.add_argument('--reps', type=int, default=3)
a = p.parse_args()
repo = Path(a.repo); W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
NEP = repo / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
w27 = read(repo / 'tests_pytest/fixtures/dftd3/water27.xyz')
SYS = {'w81': w27, 'w648': w27.repeat((2,) * 3), 'w5184': w27.repeat((4,) * 3), 'w41472': w27.repeat((8,) * 3),
       'cu4000': bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((10, 10, 10))}
for k, v in SYS.items():
    write(W / f'{k}.xyz', v, format='extxyz')
V = {'off': None, 'd3': 'dftd3 pbe 12 6', 'd2_12': 'dftd2 0.75 12', 'd2_30': 'dftd2 0.75 30'}
if a.set == 'main':
    cases = [(s, V) for s in ('w81', 'w648', 'w5184', 'w41472', 'cu4000')]
else:
    cv = {'off': None, **{f'd2_{rc}': f'dftd2 0.75 {rc}' for rc in (16, 20, 25)}}
    cases = [(s, cv) for s in ('w648', 'w5184')]
if a.only: cases = [c for c in cases if c[0] in a.only.split(',')]
cnt = [0]


def t(x, line, steps):
    cnt[0] += 1
    d = W / f'run{cnt[0]}'; d.mkdir()
    shutil.copy(W / f'{x}.xyz', d / 'model.xyz')
    (d / 'run.in').write_text(f'potential {NEP}\n' + (line + '\n' if line else '') +
                              f'velocity 300 seed 1\nensemble nve\ntime_step 0.5\nrun {steps}\n')
    o = subprocess.run([a.gpumd], cwd=d, capture_output=True, text=True).stdout
    shutil.rmtree(d)
    return float(re.search(r'Time used = ([0-9.]+) s', o).group(1))


def slope(x, line):
    return 1e3 * (t(x, line, 500) - t(x, line, 100)) / 400


res = []
for x, vs in cases:
    r = {'system': x, 'N': len(SYS[x]), 'slopes': {k: [] for k in vs}, 'extra': {k: [] for k in vs if k != 'off'}}
    for rep in range(a.reps):
        s = {k: slope(x, l) for k, l in vs.items()}
        for k in vs:
            r['slopes'][k].append(s[k])
            if k != 'off': r['extra'][k].append(s[k] - s['off'])
    r['median'] = {k: statistics.median(v) for k, v in r['extra'].items()}
    r['off_median'] = statistics.median(r['slopes']['off'])
    res.append(r)
    print(x, r['N'], 'off %.3f' % r['off_median'], {k: [round(u, 3) for u in v] for k, v in r['extra'].items()}, flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
