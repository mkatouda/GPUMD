"""S6 timing sanity: premerge (8fe8ab99) vs merged binary, slope (500-100 steps)/400, D3 (pbe 12 6) and D2 (0.75 15) extra time per step,
water 648 and 5184 (replicas of tests_pytest/fixtures/dftd2/water27.xyz). usage: timing.py SCRATCH OUT.json [reps]
SCRATCH/bin/gpumd_base (premerge), gpumd_new (merged). Median over reps; base then new alternate."""
import json, re, shutil, statistics, subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
REPO = Path(__file__).resolve().parents[4]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
SRC = REPO / 'tests_pytest/fixtures/dftd2/water27.xyz'
systems = {'w648': 2, 'w5184': 4}
from ase.io import read, write
(S / 'xyz').mkdir(parents=True, exist_ok=True)
for k, n in systems.items():
    write(S / 'xyz' / f'{k}.xyz', read(SRC).repeat(n), format='extxyz')
def t(b, x, kw, steps):
    d = S / 'work' / f'{b}_{x}_{steps}'; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copy(S / 'xyz' / f'{x}.xyz', d / 'model.xyz')
    (d / 'run.in').write_text(f'potential {NEP}\n{kw}\nvelocity 300 seed 1\nensemble nve\ntime_step 0.5\nrun {steps}\n')
    o = subprocess.run([str(S / 'bin' / f'gpumd_{b}')], cwd=d, capture_output=True, text=True).stdout
    m = re.search(r'Time used = ([0-9.]+) s', o); shutil.rmtree(d, ignore_errors=True)
    return float(m.group(1))
sl = lambda b, x, kw: 1e3 * (t(b, x, kw, 500) - t(b, x, kw, 100)) / 400
out = []
for x in systems:
    for name, kw in (('D3', 'dftd3 pbe 12 6'), ('D2', 'dftd2 0.75 15')):
        r = {'base': [], 'new': []}
        for _ in range(reps):
            for b in ('base', 'new'):
                r[b].append(sl(b, x, kw) - sl(b, x, ''))
        mb, mn = statistics.median(r['base']), statistics.median(r['new'])
        row = dict(system=x, model=name, base_ms=mb, new_ms=mn, ratio=mn / mb, runs=r); out.append(row)
        print(f"{x} {name}: extra base {mb:.4f} new {mn:.4f} ms ratio {mn/mb:.3f}", flush=True)
        json.dump(out, open(sys.argv[2], 'w'), indent=1)
