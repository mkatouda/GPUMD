"""VER-0001 slope method (100 vs 500 steps), D3 extra time per step, base vs new binary alternated.
usage: slope.py SCRATCH OUT.json [--reps 3] [--set main|cutoff]
SCRATCH/bin/gpumd_base, gpumd_new, SCRATCH/xyz/<name>.xyz. Time per step = (t500 - t100)/400 from the 'Time used' line;
D3 extra = slope(with dftd3) - slope(without). Each repetition measures base then new; result = median over repetitions.
Run: potential NEP89; [dftd3 pbe RC CN]; velocity 300 seed 1; nve; time_step 0.5."""
import argparse, json, re, shutil, statistics, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
p = argparse.ArgumentParser()
p.add_argument('scratch'); p.add_argument('out'); p.add_argument('--reps', type=int, default=3)
p.add_argument('--set', default='main')
a = p.parse_args()
S = Path(a.scratch)
MAIN = [('w81', 12, 6), ('w648', 12, 6), ('scan_2187', 12, 6), ('w5184', 12, 6), ('scan_8748', 12, 6),
        ('w41472', 12, 6), ('cu4000', 12, 6), ('clus972', 12, 6)]
CUT = [('w648', rc, 6) for rc in (8, 16, 20)] + [('w5184', rc, 6) for rc in (8, 16, 20)] + \
      [('w648', 12, cn) for cn in (3, 12)] + [('w5184', 12, cn) for cn in (3, 12)]
cases = MAIN if a.set == 'main' else CUT


def t(binary, x, d3, steps):
    d = S / 'work' / f'{binary}_{x}_{"d3" if d3 else "no"}_{steps}'
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copy(S / 'xyz' / f'{x}.xyz', d / 'model.xyz')
    (d / 'run.in').write_text(f'potential {NEP}\n' + (f'dftd3 pbe {d3[0]} {d3[1]}\n' if d3 else '') +
                              f'velocity 300 seed 1\nensemble nve\ntime_step 0.5\nrun {steps}\n')
    o = subprocess.run([str(S / 'bin' / f'gpumd_{binary}')], cwd=d, capture_output=True, text=True).stdout
    m = re.search(r'Time used = ([0-9.]+) s', o)
    shutil.rmtree(d, ignore_errors=True)
    return float(m.group(1)) if m else float('nan')


def slope(binary, x, d3):
    return 1e3 * (t(binary, x, d3, 500) - t(binary, x, d3, 100)) / 400


rows = []
for x, rc, cn in cases:
    r = {'system': x, 'rc': rc, 'cn': cn, 'base': {'d3': [], 'no': [], 'extra': []}, 'new': {'d3': [], 'no': [], 'extra': []}}
    for rep in range(a.reps):
        for b in ('base', 'new'):
            s_no = slope(b, x, None); s_d3 = slope(b, x, (rc, cn))
            r[b]['no'].append(s_no); r[b]['d3'].append(s_d3); r[b]['extra'].append(s_d3 - s_no)
    for b in ('base', 'new'):
        r[b]['extra_median'] = statistics.median(r[b]['extra']); r[b]['no_median'] = statistics.median(r[b]['no'])
    r['speedup'] = r['base']['extra_median'] / r['new']['extra_median']
    rows.append(r)
    print(f"{x:10s} rc {rc:2d} cn {cn:2d} noD3 {r['new']['no_median']:.3f} ms  extra base {r['base']['extra_median']:.3f}  "
          f"new {r['new']['extra_median']:.3f}  x{r['speedup']:.2f}  new runs {[round(v, 3) for v in r['new']['extra']]}", flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)
