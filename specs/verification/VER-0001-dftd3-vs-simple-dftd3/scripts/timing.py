import os, re, shutil, subprocess, time, json
import numpy as np
from ase.io import read, write
from dftd3.interface import DispersionModel, RationalDampingParam

GPUMD = '/home/katouda/work/dielectric-response_pj/GPUMD/src/gpumd'
NEP = '/home/katouda/work/dielectric-response_pj/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt'
BOHR = 0.529177210903
STEPS = 100
base = read('water27.xyz')
rows = []
for rep, cs in [(1, (12, 6)), (2, (12, 6)), (4, (12, 6)), (6, (12, 6)), (8, (12, 6)), (4, (25, 15))]:
    a = base.repeat((rep,) * 3)
    n = len(a)
    times = {}
    for tag, d3 in (('noD3', None), ('D3', cs)):
        d = f'time_{n}_{cs[0]}_{tag}'
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        write(f'{d}/model.xyz', a)
        inp = f'potential {NEP}\n' + (f'dftd3 pbe {d3[0]} {d3[1]}\n' if d3 else '') + \
              f'velocity 300\nensemble nve\ntime_step 0.5\nrun {STEPS}\n'
        open(f'{d}/run.in', 'w').write(inp)
        best = None
        for _ in range(2):   # repeat; keep the faster (the first call can include warm-up)
            out = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True).stdout
            m = re.search(r'Time used = ([0-9.]+) s', out)
            t = float(m.group(1)) if m else float('nan')
            best = t if best is None else min(best, t)
        times[tag] = best
    # CPU reference (simple-dftd3), same cutoffs, energy+gradient
    m = DispersionModel(a.numbers, a.positions / BOHR, a.cell.array / BOHR, a.pbc)
    m.set_realspace_cutoff(cs[0] / BOHR, cs[0] / BOHR, cs[1] / BOHR)
    tt = time.perf_counter()
    m.get_dispersion(RationalDampingParam(method='pbe'), grad=True)
    tref = time.perf_counter() - tt
    row = dict(natoms=n, rc=f'{cs[0]}/{cs[1]}', t_noD3=times['noD3'], t_D3=times['D3'],
               ms_step_noD3=1e3 * times['noD3'] / STEPS, ms_step_D3=1e3 * times['D3'] / STEPS,
               ms_step_D3_only=1e3 * (times['D3'] - times['noD3']) / STEPS, t_ref_cpu_ms=1e3 * tref)
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open('timing.json', 'w'), indent=1)
