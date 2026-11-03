"""Generate the structures and dftd3-lib D2-mode oracle values for test_dftd2_reference.py.

Oracle: the `dftd3` program built from dftd3-lib (commit 3c99ee981266548a37588abe3a6f5cb56092d51a,
prg/dftd3, gfortran), run as `dftd3 <file> -func <f> -old -grad [-pbc] -cutoff <rc/bohr>`.  In
this mode the pair energy is plain unshifted D2 with a sharp cutoff (r^2 > rthr skipped), so the
stored values are the UNSHIFTED limit; the shifted-force terms are added in the test.
Energies come from the `.EDISP` file (Hartree, 16 digits), gradients from `dftd3_gradient`
(Eh/bohr).  Conversions: bohr = 0.52917726 A, 1 Eh = 27.21138505 eV.  Forces = -gradient.

Usage (not part of the pytest run; needs ase + numpy and the oracle binary):

    python generate_oracle_reference.py /path/to/dftd3 [work_dir]

Reads/creates the structure files in this directory and rewrites oracle_reference.json.
The numbers are never edited by hand and never regenerated to make a failing test pass.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.data import chemical_symbols
from ase.io import read, write

HERE = Path(__file__).resolve().parent
BOHR = 0.52917726
EV_PER_EH = 27.21138505
COMMIT = '3c99ee981266548a37588abe3a6f5cb56092d51a'

# s6 values exactly representable in binary: dftd3 stores some s6 as single-precision literals
# (b-lyp 1.2 becomes 1.2000000477, a 4e-8 relative error), so those functionals are not used.
FUNC_S6 = {'pbe': 0.75, 'tpss': 1.0, 'b97-d': 1.25}


def make_structures():
    """Create the fixture files that are not copied from the D3 fixtures."""
    dimer = Atoms('CO', positions=[[0, 0, 0], [0.9, 1.7, 3.1]], cell=[30, 30, 30], pbc=False)
    write(HERE / 'dimer.xyz', dimer, format='extxyz')
    rng = np.random.default_rng(20261103)
    # 86 atoms: Z = 1..86 once, random positions in a sphere-like cluster (checks the whole table)
    pos = rng.uniform(-6.0, 6.0, size=(86, 3))
    for k in range(86):      # push apart pairs closer than 1.5 A so that no pair is unphysical
        for _ in range(200):
            d = np.linalg.norm(pos[:k] - pos[k], axis=1)
            if k == 0 or d.min() > 1.6:
                break
            pos[k] = rng.uniform(-6.0, 6.0, size=3)
    all86 = Atoms(numbers=list(range(1, 87)), positions=pos, cell=[40, 40, 40], pbc=False)
    write(HERE / 'all86.xyz', all86, format='extxyz')
    # small triclinic cell with 6 atoms, many images within rc
    cart = np.array([[0, 0, 0], [1.3, .2, .1], [2.6, 3.1, 2.2], [5, 1, 4], [6.1, 5.2, 7.7], [2, 6, 3.3]])
    tri6 = Atoms('CCOHHH', positions=cart,
                 cell=[[8, 0, 0], [.5, 7.5, 0], [.3, .4, 9]], pbc=True)
    write(HERE / 'tri6.xyz', tri6, format='extxyz')


# name, structure file, functional, rc (A).  Non-periodic cases: the oracle IGNORES -cutoff (all
# pairs are summed; measured 2026-11-03), so rc must exceed the cluster extent there.
CASES = [
    ('dimer', 'dimer.xyz', 'pbe', 12.0),
    ('benzene_dimer', 'benzene_dimer.xyz', 'pbe', 8.0),
    ('all86', 'all86.xyz', 'pbe', 20.0),
    ('graphite', 'graphite.xyz', 'pbe', 12.0),
    ('water27', 'water27.xyz', 'pbe', 12.0),
    ('water27_tri', 'water27_tri.xyz', 'b97-d', 12.0),
    ('tri6', 'tri6.xyz', 'tpss', 20.0),
    ('water_large', 'water_large.xyz', 'pbe', 12.0),
]


def write_oracle_input(atoms, path):
    """xyz (non-periodic) or POSCAR (periodic), positions with 17 significant digits."""
    if not atoms.pbc.any():
        lines = [str(len(atoms)), 'D2 oracle input']
        for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
            lines.append('%s %.17g %.17g %.17g' % (s, *p))
        path.write_text('\n'.join(lines) + '\n')
        return 'xyz'
    # the POSCAR reader of dftd3 accepts at most 10 species entries: group atoms by element
    # (stable sort, the caller permutes the forces back)
    atoms = atoms[np.argsort(atoms.numbers, kind='stable')]
    syms = atoms.get_chemical_symbols()
    runs = []
    for s in syms:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    lines = ['D2 oracle input', '1.0']
    for v in atoms.cell.array:
        lines.append('%.17g %.17g %.17g' % tuple(v))
    lines.append(' '.join(r[0] for r in runs))
    lines.append(' '.join(str(r[1]) for r in runs))
    lines.append('Cartesian')
    for p in atoms.positions:
        lines.append('%.17g %.17g %.17g' % tuple(p))
    path.write_text('\n'.join(lines) + '\n')
    return 'POSCAR'


def run_case(oracle, atoms, func, rc, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    kind = write_oracle_input(atoms, workdir / ('POSCAR' if atoms.pbc.any() else 'input.xyz'))
    fname = 'POSCAR' if kind == 'POSCAR' else 'input.xyz'
    cmd = [str(oracle), fname, '-func', func, '-old', '-grad', '-cutoff', '%.17g' % (rc / BOHR)]
    if kind == 'POSCAR':
        cmd.append('-pbc')
    out = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=True).stdout
    assert 'normal termination' in out, out[-500:]
    energy_eh = float((workdir / '.EDISP').read_text().split()[0].lower().replace('d', 'e'))
    grad = np.array([[float(x.lower().replace('d', 'e')) for x in ln.split()]
                     for ln in (workdir / 'dftd3_gradient').read_text().splitlines() if ln.strip()])
    assert grad.shape == (len(atoms), 3)
    if kind == 'POSCAR':      # undo the sort by element done in write_oracle_input
        order = np.argsort(atoms.numbers, kind='stable')
        unsorted = np.empty_like(grad)
        unsorted[order] = grad
        grad = unsorted
    result = dict(energy_eV=energy_eh * EV_PER_EH,
                  forces_eV_A=(-grad * EV_PER_EH / BOHR).tolist())
    if kind == 'POSCAR':
        # dftd3_cellgradient = dE/d(lattice vector components) at fixed fractional coordinates
        # (Eh/bohr, row i = lattice vector i).  Strain derivative S_ab = sum_i cg[i,a] lat[i,b];
        # GPUMD virial W = -S (this agrees with the analytic strain derivative
        # of the NumPy reference to 1e-13; the earlier "8e-4 mismatch" was finite-difference
        # noise of the sharply truncated energy, not an oracle error).
        cg = np.array([[float(x.lower().replace('d', 'e')) for x in ln.split()]
                       for ln in (workdir / 'dftd3_cellgradient').read_text().splitlines()
                       if ln.strip()]) * EV_PER_EH / BOHR              # eV/A
        result['cellgradient_eV_A'] = cg.tolist()
        result['virial_eV'] = (-(cg.T @ atoms.cell.array)).tolist()
    return result


def main(oracle, scratch=None):
    make_structures()
    ref = {'_provenance': dict(
        reference='dftd3-lib prg/dftd3 (-old -grad), commit ' + COMMIT,
        mode='unshifted D2, sharp cutoff rc', units='eV, eV/A (forces = -gradient)',
        bohr_A=BOHR, Eh_eV=EV_PER_EH,
        nonperiodic_cutoff='the oracle ignores -cutoff for non-periodic input (all pairs summed); '
                           'non-periodic cases therefore use rc larger than the cluster extent',
        virial='virial_eV = -(cellgradient^T @ cell), unshifted, periodic cases only')}
    scratch = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix='d2oracle_'))
    for name, fname, func, rc in CASES:
        atoms = read(HERE / fname)
        r = run_case(oracle, atoms, func, rc, scratch / name)
        r.update(file=fname, functional=func, s6=FUNC_S6[func], rc_A=rc,
                 natoms=len(atoms), pbc=[bool(b) for b in atoms.pbc])
        ref[name] = r
        print(name, len(atoms), 'E(eV) = %.15e' % r['energy_eV'])
    (HERE / 'oracle_reference.json').write_text(json.dumps(ref, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
