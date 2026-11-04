"""NVE energy conservation with `dftd3 pbe 12 6`, baseline vs candidate binary (DR-001 AC-3).

usage: nve.py CAND_BIN BASE_BIN NEP_TXT WATER27_XYZ WORKDIR SYSTEM NSTEPS_NVE [SEEDS...]
SYSTEM: water648 | water5184 | cu4000
1. Common start: structure equilibrated with NEP89 only (candidate binary), NVT-BDP 300 K, 0.5 fs, 10000 steps
   (restart.xyz of that run).
2. (Unless KEEP_VEL=1 is set: the velocities of restart.xyz are kept and `velocity ... seed` is then ignored by gpumd.)
   For each velocity seed and each binary: NVE, dt 0.5 fs, `velocity 300 seed S`, thermo every 10 steps, with D3.
   D3=0 in the environment runs the same without D3 (control).  Reports E_tot std, max-min and linear-fit drift.  Same seed and start for both binaries.
"""
import os, subprocess, sys, json
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.build import bulk

cand, base, nep, w27, work, system, nsteps = sys.argv[1:8]
seeds = [int(s) for s in sys.argv[8:]] or [20261103]
nsteps = int(nsteps)
D3 = os.environ.get('D3', '1') == '1'   # D3=0: control run without dftd3
suf = ('' if D3 else '_noD3') + os.environ.get('OUTTAG', '')
work = Path(work); work.mkdir(parents=True, exist_ok=True)
eq = work / system / 'equil'; eq.mkdir(parents=True, exist_ok=True)
if system == 'water648':
    at = read(w27).repeat((2, 2, 2))
elif system == 'water5184':
    at = read(w27).repeat((4, 4, 4))
elif system == 'cu4000':
    at = bulk('Cu', 'fcc', a=3.615, cubic=True).repeat((10, 10, 10))
else:
    raise SystemExit('unknown system')
n = len(at)
if not (eq / 'restart.xyz').exists():
    write(eq / 'model.xyz', at, format='extxyz')
    (eq / 'run.in').write_text(f'potential {nep}\nvelocity 300 seed 7\nensemble nvt_bdp 300 300 100\ntime_step 0.5\n'
                               'dump_restart 10000\nrun 10000\n')
    subprocess.run([cand], cwd=eq, stdout=subprocess.DEVNULL, check=True)
start = read(eq / 'restart.xyz')
if os.environ.get('KEEP_VEL') != '1':
    # gpumd ignores `velocity` when model.xyz has velocities; drop them so that the seed selects the initial velocities
    start.arrays.pop('momenta', None); start.arrays.pop('vel', None)
res = []
for seed in seeds:
    for tag, binary in (('base', base), ('cand', cand)):
        d = work / system / f'{tag}{suf}_s{seed}'; d.mkdir(exist_ok=True)
        write(d / 'model.xyz', start, format='extxyz')
        d3line = 'dftd3 pbe 12 6\n' if D3 else ''
        (d / 'run.in').write_text(f'potential {nep}\n{d3line}velocity 300 seed {seed}\nensemble nve\n'
                                  f'time_step 0.5\ndump_thermo 10\nrun {nsteps}\n')
        subprocess.run([binary], cwd=d, stdout=subprocess.DEVNULL, check=True)
        th = np.loadtxt(d / 'thermo.out'); e = th[:, 1] + th[:, 2]
        t = np.arange(len(e)) * 10 * 0.5e-3
        slope = np.polyfit(t, e, 1)[0]
        np.savetxt(work / system / f'etot_{tag}{suf}_s{seed}.txt', np.c_[t, e], header='t_ps E_tot_eV')
        r = dict(system=system + suf, N=n, tag=tag, seed=seed, ps=t[-1], std_eV=e.std(), maxmin_eV=float(np.ptp(e)),
                 drift_meV_per_atom_ps=slope / n * 1e3, T_mean=th[:, 0].mean())
        res.append(r)
        print(json.dumps(r), flush=True)
json.dump(res, open(work / f'nve_{system}{suf}.json', 'w'), indent=1)
