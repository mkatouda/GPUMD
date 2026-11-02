import os, re, subprocess, shutil, time, json
import numpy as np
from ase.io import read
from dftd3.interface import DispersionModel, RationalDampingParam

GPUMD = '/home/katouda/work/dielectric-response_pj/GPUMD/src/gpumd'
NEP = '/home/katouda/work/dielectric-response_pj/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt'
HA, BOHR = 27.211386245988, 0.529177210903
systems = ['benzene_dimer', 'graphite', 'water27', 'water27_tri']
cutsets = [(12.0, 6.0), (25.0, 15.0)]   # (potential, coordination-number) cutoffs in A

def run_gpumd(name, d3, tag):
    d = f'run_{name}_{tag}'
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    shutil.copy(f'{name}.xyz', f'{d}/model.xyz')
    inp = f'potential {NEP}\n' + (f'dftd3 pbe {d3[0]} {d3[1]}\n' if d3 else '') + \
          'ensemble nve\ntime_step 0\ndump_xyz 1 dump.xyz force precision double\nrun 1\n'
    open(f'{d}/run.in', 'w').write(inp)
    out = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True).stdout
    a = read(f'{d}/dump.xyz', index=0, format='extxyz')
    hdr = open(f'{d}/dump.xyz').readlines()[1]
    vir = np.array(re.search(r'virial="([^"]+)"', hdr).group(1).split(), float).reshape(3, 3)
    en = float(re.search(r'energy=([-0-9.eE+]+)', hdr).group(1))
    return en, a.get_forces(), vir

def reference(name, d3, converged=False):
    a = read(f'{name}.xyz')
    pbc = a.pbc.any()
    m = DispersionModel(a.numbers, a.positions / BOHR,
                        a.cell.array / BOHR if pbc else None,
                        a.pbc if pbc else None)
    if not converged:
        m.set_realspace_cutoff(d3[0] / BOHR, d3[0] / BOHR, d3[1] / BOHR)
    t = time.perf_counter()
    r = m.get_dispersion(RationalDampingParam(method='pbe'), grad=True)
    dt = time.perf_counter() - t
    return r['energy'] * HA, -r['gradient'] * HA / BOHR, r['virial'] * HA, dt

res = []
for name in systems:
    e0, f0, v0 = run_gpumd(name, None, 'none')
    for cs in cutsets:
        e1, f1, v1 = run_gpumd(name, cs, f'{int(cs[0])}_{int(cs[1])}')
        eD, fD, vD = e1 - e0, f1 - f0, v1 - v0
        for label, conv in (('same-cutoff', False), ('converged', True)):
            er, fr, vr, dt = reference(name, cs, conv)
            row = dict(system=name, natoms=len(fD), cut=f'{cs[0]:g}/{cs[1]:g}', ref=label,
                       E_gpumd=eD, E_ref=er, dE=eD - er,
                       Fmax_ref=np.abs(fr).max(), dFmax=np.abs(fD - fr).max(),
                       dFrms=np.sqrt(((fD - fr) ** 2).mean()),
                       dV_plus=np.abs(vD - vr).max(), dV_minus=np.abs(vD + vr).max(),
                       Vmax=np.abs(vr).max(), t_ref=dt)
            res.append(row)
            print(json.dumps({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}))
json.dump([{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()} for r in res], open('accuracy.json', 'w'), indent=1)
