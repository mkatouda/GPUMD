"""fcc Au and Pt: D2 shifted-force energy per atom and pressure vs rc, against the continuum closed form
dE = 7 pi rho s6 C6 / (3 rc^3),  dP = rho dE   (shifted-force sum minus untruncated sum).

usage: heavy_element_closed_form.py <gpumd> <scratch dir>
D2 part = (NEP89 + dftd2 s6 rc) - (NEP89 alone), 4x4x4 conventional cells (256 atoms), s6 = 0.75,
double-precision dump.  Lattice constants (typical experimental values, room temperature): Au 4.078, Pt 3.924 A.
The untruncated limit is estimated by a least-squares fit E(rc) = E_inf + A rc^-3 over the rc >= 60 points.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests_pytest'))
import dftd2_reference as ref

exe, scratch = sys.argv[1], Path(sys.argv[2])
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
S6 = 0.75
EV_A3_TO_GPA = 160.21766208
RCS = [30, 40, 60, 80, 120, 160]


def run(atoms, d, line):
    d.mkdir(parents=True, exist_ok=True)
    atoms.write(d / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP}'] + ([line] if line else []) + \
        ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (d / 'run.in').write_text('\n'.join(lines) + '\n')
    r = subprocess.run([exe], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:]
    h = (d / 'dump.xyz').read_text().splitlines()[1]
    e = float(re.search(r'energy=([-0-9.eE+]+)', h).group(1))
    w = np.array(re.search(r'virial="([^"]+)"', h).group(1).split(), float).reshape(3, 3)
    return e, w


for sym, z, a in (('Au', 79, 4.078), ('Pt', 78, 3.924)):
    atoms = bulk(sym, 'fcc', a=a, cubic=True).repeat((4, 4, 4))
    n, vol = len(atoms), atoms.get_volume()
    rho = n / vol
    c6 = ref.PARAMS['c6'][z]
    e0, w0 = run(atoms, scratch / sym / 'off', None)
    rows = []
    for rc in RCS:
        e1, w1 = run(atoms, scratch / sym / f'rc{rc}', f'dftd2 {S6} {rc}')
        rows.append((rc, (e1 - e0) / n, np.trace(w1 - w0) / (3 * vol) * EV_A3_TO_GPA))
    rows = np.array(rows)
    sel = rows[:, 0] >= 60
    X = np.vstack([np.ones(sel.sum()), rows[sel, 0] ** -3.0]).T
    (e_inf, Ae), *_ = np.linalg.lstsq(X, rows[sel, 1], rcond=None)
    (p_inf, Ap), *_ = np.linalg.lstsq(X, rows[sel, 2], rcond=None)
    print(f'{sym}: a = {a} A, N = {n}, rho = {rho:.5f} /A^3, C6 = {c6:.3f} eV A^6, s6 C6 rho = {S6 * c6 * rho:.4f}')
    print(f'  fit over rc >= 60:  E_inf = {e_inf * 1e3:.3f} meV/atom, A_E = {Ae * 1e3:.1f} meV A^3/atom; '
          f'P_inf = {p_inf * 1e3:.2f} MPa, A_P = {Ap * 1e3:.0f} MPa A^3')
    print(f'  closed form A_E = 7 pi rho s6 C6 / 3 = {7 * np.pi / 3 * rho * S6 * c6 * 1e3:.1f} meV A^3/atom, '
          f'A_P = rho * A_E = {7 * np.pi / 3 * rho * S6 * c6 * rho * EV_A3_TO_GPA * 1e3:.0f} MPa A^3')
    print('   rc[A]   E_D2[meV/atom]  dE_meas=E-E_inf[meV]  dE_closed[meV]   P_D2[MPa]  dP_meas[MPa]  dP_closed[MPa]')
    for rc, e, p in rows:
        dEc = 7 * np.pi / 3 * rho * S6 * c6 / rc ** 3
        print(f'  {rc:6.0f} {e * 1e3:15.3f} {(e - e_inf) * 1e3:20.3f} {dEc * 1e3:15.3f} '
              f'{p * 1e3:12.2f} {(p - p_inf) * 1e3:13.2f} {dEc * rho * EV_A3_TO_GPA * 1e3:14.2f}')
