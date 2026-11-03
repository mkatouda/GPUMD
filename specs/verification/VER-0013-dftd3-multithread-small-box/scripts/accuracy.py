"""Accuracy of the D3 contribution (with D3 minus without D3, static, NEP89) for the DR-001 AC-1 reference
cases, for two binaries, plus the per-atom difference candidate - baseline (energy per atom, force, per-atom
virial).  usage: accuracy.py BASE_BINARY CAND_BINARY WORKDIR   (run with the conda env python that has ase/numpy).
Prints, per case: |dE|, max|dF|, max|dV| against the stored simple-dftd3 reference and the AC-1 tolerance of
tests_pytest/test_dftd3_reference.py, and max |cand - base| of per-atom potential, force, virial (D3 part)."""
import re, subprocess, sys
from pathlib import Path
import numpy as np
from ase.io import read

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests_pytest'))
import test_dftd3_reference as T

base, cand, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def run(binary, tag, name, d3):
    d = work / f'{tag}_{name}_{int(d3)}'
    d.mkdir(parents=True, exist_ok=True)
    ref = T.REFERENCE[name]
    atoms = read(T.FIXTURE_DIR / f'{name}.xyz')
    atoms.write(d / 'model.xyz', format='extxyz')
    lines = [f'potential {T.NEP_MODEL}']
    if d3:
        lines.append(f'dftd3 pbe {ref["potential_cutoff"]} {ref["cn_cutoff"]}')
    lines += ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force potential virial precision double', 'run 1']
    (d / 'run.in').write_text('\n'.join(lines) + '\n')
    r = subprocess.run([str(binary)], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:]
    rows = np.array([l.split()[1:] for l in (d / 'dump.xyz').read_text().splitlines()[2:2 + len(atoms)]], float)
    # columns: x y z, force (3), potential, virial (9)
    hdr = (d / 'dump.xyz').read_text().splitlines()[1]
    e = float(re.search(r'energy=([-0-9.eE+]+)', hdr).group(1))
    v = np.array(re.search(r'virial="([^"]+)"', hdr).group(1).split(), float).reshape(3, 3)
    return e, rows[:, 3:6], v, rows[:, 6], rows[:, 7:16]


for name in T.CASES:
    ref = T.REFERENCE[name]
    out = {}
    for tag, b in (('base', base), ('cand', cand)):
        e0, f0, v0, p0, w0 = run(b, tag, name, False)
        e1, f1, v1, p1, w1 = run(b, tag, name, True)
        out[tag] = (e1 - e0, f1 - f0, v1 - v0, p1 - p0, w1 - w0)
    line = [f'{name:14s}']
    for tag in ('base', 'cand'):
        e, f, v = out[tag][:3]
        d = (abs(e - ref['energy']), float(np.abs(f - np.array(ref['forces'])).max()),
             float(np.abs(v - np.array(ref['virial'])).max()))
        tol = [T.tolerance(name, q) for q in ('energy', 'force', 'virial')]
        line.append(f'{tag}: dE {d[0]:.2e}/{tol[0]:.2e} dF {d[1]:.2e}/{tol[1]:.2e} dV {d[2]:.2e}/{tol[2]:.2e}')
    b, c = out['base'], out['cand']
    line.append(f'cand-base: dE {abs(c[0]-b[0]):.1e} dF {np.abs(c[1]-b[1]).max():.1e} (max|F| {np.abs(b[1]).max():.1e}) '
                f'dpot {np.abs(c[3]-b[3]).max():.1e} dvirial_atom {np.abs(c[4]-b[4]).max():.1e} (max {np.abs(b[4]).max():.1e})')
    print(' | '.join(line), flush=True)
