"""Static D2(rc) - D2(off) study. usage: cutoff_static.py GPUMD_BIN NEP_TXT STRUCT_DIR WORKDIR RESULTS_DIR [--ref]"""
import re, subprocess, sys, os
from pathlib import Path
import numpy as np
from ase.io import read

gpumd, nep, sdir, work, res = sys.argv[1:6]
S6 = 0.75
RCS = [12, 16, 20, 25, 30, 40]
REPO = Path(gpumd).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tests_pytest'))
import dftd2_reference as ref
EV_A3_GPA = 160.21766208

def run(name, rc):
    d = Path(work) / f'{name}_{rc}'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'model.xyz').write_text((Path(sdir) / f'{name}.xyz').read_text())
    lines = [f'potential {nep}']
    if rc: lines.append(f'dftd2 {S6} {rc}')
    lines += ['velocity 0.0001', 'ensemble nve', 'time_step 0', 'dump_thermo 1',
              'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (d / 'run.in').write_text('\n'.join(lines) + '\n')
    subprocess.run([gpumd], cwd=d, stdout=subprocess.DEVNULL, check=True)
    fr = read(d / 'dump.xyz', format='extxyz', index=-1)
    th = np.loadtxt(d / 'thermo.out', ndmin=2)[-1]
    return dict(E=fr.get_potential_energy(), F=fr.get_forces(),
                P=th[3:6], V=abs(np.linalg.det(fr.cell.array)))

out = []
for name in os.environ.get('SYSTEMS', 'water graphite mgo').split():
    atoms = read(Path(sdir) / f'{name}.xyz')
    n = len(atoms)
    base = run(name, 0)
    d2 = {}
    for rc in RCS:
        r = run(name, rc)
        d2[rc] = dict(E=r['E'] - base['E'], F=r['F'] - base['F'], P=r['P'] - base['P'])
    for rc in RCS[:-1]:
        dE = (d2[rc]['E'] - d2[40]['E']) / n * 1e3
        dF = (d2[rc]['F'] - d2[40]['F']) * 1e3
        dP = d2[rc]['P'] - d2[40]['P']
        out.append((name, n, rc, d2[rc]['E'] / n * 1e3, dE, np.abs(dF).max(), np.sqrt((dF**2).mean()),
                    dP.mean(), *dP))
    out.append((name, n, 40, d2[40]['E'] / n * 1e3, 0, 0, 0, d2[40]['P'].mean(), *d2[40]['P']))
    if '--ref' in sys.argv and name == 'water':
        for rc in (12, 25):
            e, f, w = ref.energy_forces_virial(atoms, S6, rc)
            print(f'REF water rc={rc}: dE(gpu-ref)={d2[rc]["E"]-e:.3e} eV (E={e:.6f}), '
                  f'max dF={np.abs(d2[rc]["F"]-f).max():.3e} eV/A (max|F|={np.abs(f).max():.3e}), '
                  f'dP_gpu-ref={d2[rc]["P"].mean() - (np.trace(w)/3/base["V"]*EV_A3_GPA):.3e} GPa '
                  f'(P_ref={np.trace(w)/3/base["V"]*EV_A3_GPA:.5f}, P_gpu={d2[rc]["P"].mean():.5f})')
hdr = 'system N rc E_D2/atom[meV] dE/atom[meV] maxdF[meV/A] rmsdF[meV/A] dP[GPa] dPxx dPyy dPzz (d* vs rc=40)'
print(hdr)
for o in out:
    print(f'{o[0]:9s} {o[1]:4d} {o[2]:3d} ' + ' '.join(f'{x:12.5f}' for x in o[3:]))
