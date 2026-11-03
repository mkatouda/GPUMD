"""D2 cost vs rc on water 648. usage: cost.py GPUMD_BIN NEP_TXT STRUCT_DIR WORKDIR"""
import re, subprocess, sys
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
gpumd, nep, sdir, work = sys.argv[1:5]
STEPS, REP = 1000, 3
atoms = read(Path(sdir) / 'water.xyz')
def t_run(rc):
    d = Path(work) / f'cost_{rc}'; d.mkdir(parents=True, exist_ok=True)
    (d / 'model.xyz').write_text((Path(sdir) / 'water.xyz').read_text())
    L = [f'potential {nep}'] + ([f'dftd2 0.75 {rc}'] if rc else []) + \
        ['velocity 300 seed 1', 'ensemble nve', 'time_step 0.5', f'run {STEPS}']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    ts = []
    for _ in range(REP):
        o = subprocess.run([gpumd], cwd=d, capture_output=True, text=True, check=True).stdout
        ts.append(float(re.search(r'Time used for this run = ([0-9.]+)', o).group(1)))
    return ts
off = t_run(0)
print(f'off: {[round(x,3) for x in off]} s per {STEPS} steps, min {min(off)/STEPS*1e3:.4f} ms/step')
print('rc  neighbors/atom  total ms/step(min)  D2 ms/step (min_on - min_off)  all runs [s]')
for rc in (12, 16, 20, 25, 30, 40):
    on = t_run(rc)
    i = neighbor_list('i', atoms, rc)
    nb = len(i) / len(atoms)
    print(f'{rc:3d} {nb:10.0f} {min(on)/STEPS*1e3:10.4f} {(min(on)-min(off))/STEPS*1e3:10.4f}   {[round(x,3) for x in on]}')
