"""GPU DFT-D2 (`dftd2` keyword): force / virial consistency with finite differences of the GPU energy.

Requirement DR-002 AC-4 (force = -dE/dx, virial = -dE/d(strain)); results in
specs/verification/VER-0006-dftd2-consistency-and-performance.

The D2 part is isolated as (run with dftd2) - (same run without), so the NEP part cancels exactly
(the difference is bit-reproducible, VER-0006).  s6 = 75 (100 times the usual value) makes the D2
signal large compared with the float rounding of the pair terms.  Derivatives are 5-point central
differences of the double-precision energy of the dump header.

Tolerances (fixed BEFORE the first GPU measurement, never to be loosened to make a test pass):
- force: max |F_fd - F_gpu| <= 2e-4 * max|F_D2|, step 0.01 A;
- virial: max |W_fd - W_gpu| <= 2e-4 * max|W_D2|, step 1e-3 in the deformation x -> (1 + eps) x.
The largest observed errors are listed in the VER-0006 record.  The tests fail when the sign of the
force or of the virial is flipped (checked with scratch builds, VER-0006).

Environment: DFTD2_TEST_GPUMD overrides the gpumd executable (used for the mutation checks).
"""
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd2'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
EXE = Path(os.environ.get('DFTD2_TEST_GPUMD', GPUMD_EXECUTABLE))
S6, RC = 75.0, 12.0
H_FORCE, H_STRAIN = 0.01, 1e-3
TOL = 2e-4

pytestmark = [pytest.mark.fast,
              pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')]


class _Runner:
    def __init__(self, root):
        self.root, self.count = Path(root), 0

    def run(self, atoms, d2):
        self.count += 1
        d = self.root / f'r{self.count}'
        d.mkdir(parents=True)
        atoms.write(d / 'model.xyz', format='extxyz')
        lines = [f'potential {NEP_MODEL}'] + ([f'dftd2 {S6} {RC}'] if d2 else []) + \
            ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
        (d / 'run.in').write_text('\n'.join(lines) + '\n')
        r = subprocess.run([str(EXE)], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
        header = (d / 'dump.xyz').read_text().splitlines()[1]
        energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
        virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float).reshape(3, 3)
        return energy, read(d / 'dump.xyz', format='extxyz').get_forces(), virial

    def d2(self, atoms):
        e1, f1, w1 = self.run(atoms, True)
        e0, f0, w0 = self.run(atoms, False)
        return e1 - e0, f1 - f0, w1 - w0


def _displaced(atoms, k, a, step):
    at = atoms.copy()
    p = at.positions
    p[k, a] += step
    at.set_positions(p)
    return at


def _strained(atoms, a, b, step):
    F = np.eye(3)
    F[a, b] += step
    at = atoms.copy()
    pos, cell = atoms.positions.copy(), atoms.cell.array.copy()
    at.set_cell(cell @ F.T, scale_atoms=False)
    at.set_positions(pos @ F.T)
    return at


def _five_point(energies, h):
    em2, em1, ep1, ep2 = energies
    return (em2 - 8 * em1 + 8 * ep1 - ep2) / (12 * h)


def _systems():
    w = read(FIXTURE_DIR / 'water27.xyz')
    open_cluster = w.copy()
    open_cluster.pbc = False
    return {'cubic': w, 'triclinic': read(FIXTURE_DIR / 'water27_tri.xyz'), 'open': open_cluster}


def _derivative_jobs(runner, jobs):
    """jobs: list of (atoms_builder, h); returns 5-point derivatives, computed concurrently."""
    def one(job):
        build, h = job
        return _five_point([runner.d2(build(s))[0] for s in (-2 * h, -h, h, 2 * h)], h)
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, jobs))


@pytest.mark.parametrize('name', ['cubic', 'open'])
def test_dftd2_force_is_minus_energy_gradient(name, tmp_path):
    atoms = _systems()[name]
    runner = _Runner(tmp_path)
    _, F, _ = runner.d2(atoms)
    comps = [(k, a) for k in (0, 1, 40) for a in range(3)]   # O, H and an O from the middle of the cell
    jobs = [((lambda s, k=k, a=a: _displaced(atoms, k, a, s)), H_FORCE) for k, a in comps]
    fd = [-d for d in _derivative_jobs(runner, jobs)]
    err = max(abs(f - F[k, a]) for f, (k, a) in zip(fd, comps))
    tol = TOL * np.abs(F).max()
    assert err <= tol, f'force error {err:.3e} eV/A > {tol:.3e}'


@pytest.mark.parametrize('name, comps', [('cubic', [(0, 0), (1, 2)]), ('triclinic', [(1, 1), (0, 1)])])
def test_dftd2_virial_is_strain_derivative(name, comps, tmp_path):
    atoms = _systems()[name]
    runner = _Runner(tmp_path)
    _, _, W = runner.d2(atoms)
    jobs = [((lambda s, a=a, b=b: _strained(atoms, a, b, s)), H_STRAIN) for a, b in comps]
    fd = [-d for d in _derivative_jobs(runner, jobs)]
    err = max(abs(w - W[a, b]) for w, (a, b) in zip(fd, comps))
    tol = TOL * np.abs(W).max()
    assert err <= tol, f'virial error {err:.3e} eV > {tol:.3e}'
