"""NVE energy conservation of water 648 with and without D2 (AC-4, third clause).

usage: nve.py GPUMD_BIN NEP_TXT WATER27_XYZ WORKDIR
1. water27 repeated 2x2x2 (648 atoms, 18.6 A), equilibrated with NEP89 only: NVT-BDP 300 K, 0.5 fs, 10 ps
   (the last frame, restart.xyz, is the common start).
2. 10 ps NVE (dt 0.5 fs, 20000 steps, thermo every 10 steps), `velocity 300 seed 20261103`, same start,
   (a) no D2, (b) dftd2 0.75 (default rc = 30 A).  Reports drift (linear fit of E_tot), std, max-min.
"""
import subprocess, sys
from pathlib import Path
import numpy as np
from ase.io import read, write

gpumd, nep, w27, work = sys.argv[1:5]
work = Path(work); work.mkdir(parents=True, exist_ok=True)
eq = work / 'equil'; eq.mkdir(exist_ok=True)
write(eq / 'model.xyz', read(w27).repeat((2, 2, 2)), format='extxyz')
(eq / 'run.in').write_text(f'potential {nep}\nvelocity 300 seed 7\nensemble nvt_bdp 300 300 100\ntime_step 0.5\n'
                           'dump_restart 20000\nrun 20000\n')
subprocess.run([gpumd], cwd=eq, stdout=subprocess.DEVNULL, check=True)
start = read(eq / 'restart.xyz')
for tag, d2 in (('noD2', None), ('D2_rc30', 'dftd2 0.75')):
    d = work / tag; d.mkdir(exist_ok=True)
    write(d / 'model.xyz', start, format='extxyz')
    L = [f'potential {nep}'] + ([d2] if d2 else []) + \
        ['velocity 300 seed 20261103', 'ensemble nve', 'time_step 0.5', 'dump_thermo 10', 'run 20000']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    subprocess.run([gpumd], cwd=d, stdout=subprocess.DEVNULL, check=True)
    th = np.loadtxt(d / 'thermo.out'); e = th[:, 1] + th[:, 2]; t = np.arange(len(e)) * 10 * 0.5e-3
    slope = np.polyfit(t, e, 1)[0]
    np.savetxt(work / f'etot_{tag}.txt', np.c_[t, e], header='t_ps E_tot_eV')
    print(f'{tag:8s}: E_tot mean {e.mean():.4f} eV, std {e.std():.5f} eV, max-min {np.ptp(e):.5f} eV, '
          f'drift {slope*1e3:.3f} meV/ps (= {slope/648*1e3:.5f} meV/atom/ps), T mean {th[:,0].mean():.1f} K', flush=True)
