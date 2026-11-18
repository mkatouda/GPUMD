"""DFT-D3 (`dftd3` keyword, BJ damping) on dense clusters with very large coordination numbers.

Specification: ADR-0008 (specs/decisions/ADR-0008-dftd3-c6-weight-normalization.md), acceptance
criteria AC-1 and AC-2; requirements DR-001, DR-003. Before ADR-0008 the factorized C6 weights
exp(-4 (CN - CNref)^2) underflowed to zero for CN >> CNref, the C6 fell back to the last reference
value and the derivative dC6/dCN was dropped (energy error 7-8 eV, force error ~0.1 eV/A on
H70). Structures: tests_pytest/fixtures/dftd3/{h70,cuh70}_dense_cluster.xyz (non-periodic
clusters, `dftd3 pbe 12 6`), from specs/verification/VER-0017-dftd3-stage3a/scripts/extreme_cn.py.

The D3 contribution is obtained with an all-zero NEP (fixtures/dftd3_zero/zero_nep_H_C_O_Cu_mn100.txt,
neighbour capacity 100 because the 70-atom clusters have up to 69 neighbours; the capacity-20 model
of test_dftd3_zero.py overflows on them), so energy, force and virial of a run are the D3
contribution alone and are bit-reproducible. (With NEP89 the total energy and virial differ by
~5e-4 eV between identical runs on these clusters, which would hide the 2e-6 relative cap.)
Double precision output, static frame (time_step 0). Each run uses its own directory (gpumd appends
to dump.xyz).

test_dftd3_force_finite_difference (AC-2): central difference of the D3 energy for atoms
0, 17, 35, 52, 69 along x, h = 0.02 and 0.04 A, against the analytic D3 force; tolerance
max(1e-4 eV/A, 1e-3 max|F_D3|) (fixed by the human, not to be loosened).

test_dftd3_vs_simple_dftd3 (AC-1): reference = simple-dftd3 (c6_underflow_reference.json, values
as generated, not altered). Forces: max|dF| <= 1e-5 eV/A; energy and virial: relative 2e-6
(DR-001 AC-1; energy relative to |E_D3|, virial relative to max|V_ref|).
"""
import json
import re
import subprocess

import numpy as np
import pytest
from ase.io import read, write

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd3'
NEP_MODEL = FIXTURE_DIR.parent / 'dftd3_zero' / 'zero_nep_H_C_O_Cu_mn100.txt'
D3 = 'dftd3 pbe 12 6'
CLUSTERS = {'H70_dense_cluster': 'h70_dense_cluster.xyz',
            'CuH70_dense_cluster': 'cuh70_dense_cluster.xyz'}
FD_ATOMS = [0, 17, 35, 52, 69]
FD_STEPS = [0.02, 0.04]
FD_ABS, FD_REL = 1e-4, 1e-3
FORCE_CAP = 1e-5
REL_CAP = 2e-6

pytestmark = [pytest.mark.fast,
              pytest.mark.skipif(not NEP_MODEL.exists(), reason='zero NEP model file not found')]


def _run(workdir, atoms):
    workdir.mkdir(parents=True, exist_ok=False)
    write(workdir / 'model.xyz', atoms, format='extxyz')
    lines = [f'potential {NEP_MODEL}', D3,
        'ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-1500:]
    header = (workdir / 'dump.xyz').read_text().splitlines()[1]
    energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float)
    forces = read(workdir / 'dump.xyz', format='extxyz').get_forces()
    return energy, forces, virial.reshape(3, 3)


@pytest.mark.parametrize('name', list(CLUSTERS))
def test_dftd3_force_finite_difference(tmp_path, name):
    atoms = read(FIXTURE_DIR / CLUSTERS[name])
    f_d3 = _run(tmp_path / 'f', atoms)[1]
    tol = max(FD_ABS, FD_REL * np.abs(f_d3).max())
    worst = 0.0
    for i in FD_ATOMS:
        for h in FD_STEPS:
            e = {}
            for sign, sname in ((1, 'p'), (-1, 'm')):
                moved = atoms.copy()
                moved.positions[i, 0] += sign * h
                e[sign] = _run(tmp_path / f'a{i}_h{h}_{sname}', moved)[0]
            fd = -(e[1] - e[-1]) / (2 * h)
            worst = max(worst, abs(fd - f_d3[i, 0]))
    assert worst <= tol, (name, worst, tol)


@pytest.mark.parametrize('name', list(CLUSTERS))
def test_dftd3_vs_simple_dftd3(tmp_path, name):
    ref = json.loads((FIXTURE_DIR / 'c6_underflow_reference.json').read_text())[name]
    atoms = read(FIXTURE_DIR / CLUSTERS[name])
    e, f, v = _run(tmp_path / 'd3', atoms)
    assert np.abs(f - np.array(ref['forces'])).max() <= FORCE_CAP
    assert abs(e - ref['energy']) <= REL_CAP * abs(ref['energy']), (name, e, ref['energy'])
    rv = np.array(ref['virial'])
    assert np.abs(v - rv).max() <= REL_CAP * np.abs(rv).max(), (name, np.abs(v - rv).max())
