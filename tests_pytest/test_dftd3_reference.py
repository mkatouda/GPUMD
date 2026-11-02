"""DFT-D3 (`dftd3` keyword) against stored values from an independent D3 implementation.

Requirement DR-001 (specs/requirements/DR-001-dftd3-gpu-performance.md), acceptance criterion
AC-1. The reference values in fixtures/dftd3/reference.json come from simple-dftd3 (PBE, BJ
damping, two-body only, same cutoffs as GPUMD); see fixtures/dftd3/generate_reference.py for the
provenance. They are never regenerated to make this test pass.

The D3 contribution is isolated as (run with `dftd3`) - (same run without it), using NEP89 for
the underlying potential; both runs are static (`time_step 0`, one step). Cases cover both code
paths of src/force/dftd3.cu: the "small box" path (a periodic thickness <= 2.5 x the pair
cutoff, plus the non-periodic case) and the "large box" path (cell lists).

Tolerance for each quantity is min(cap, max(2 x BASELINE, floor)): the caps are the limits of
DR-001 AC-1 (relative 2e-6 for energy and virial, absolute 1e-5 eV/A for forces), BASELINE is
the error measured on the unmodified code (2026-11-03, RTX A6000, sm_86, max of 3 runs), and
the floors keep the check above float run-to-run noise.
"""
import json
import re
import subprocess

import numpy as np
import pytest
from ase.io import read

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd3'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
REFERENCE = json.load(open(FIXTURE_DIR / 'reference.json'))
CASES = [name for name in REFERENCE if not name.startswith('_')]

REL_CAP = 2e-6        # energy and virial: fraction of |E_D3| / max|Virial_ref| (DR-001 AC-1)
FORCE_CAP = 1e-5      # eV/A
FLOORS = dict(energy=2e-6, force=1e-6, virial=2e-5)   # eV, eV/A, eV (float run-to-run noise)
# Measured on the unmodified code (see module docstring): |dE| (eV), max|dF| (eV/A), max|dVirial| (eV)
BASELINE = {
    'benzene_dimer': (3.72e-07, 2.82e-08, 3.40e-07),
    'graphite':      (3.94e-06, 4.57e-07, 5.27e-06),
    'water27':       (1.65e-06, 1.34e-08, 1.80e-06),
    'water27_rc25':  (1.73e-06, 1.28e-07, 1.21e-06),
    'water27_tri':   (2.16e-06, 2.68e-08, 2.22e-06),
    'water_large':   (4.82e-05, 1.94e-07, 4.61e-05),
    'cu_large':      (6.15e-04, 1.33e-07, 5.66e-04),
}


def _run(workdir, structure, d3_cutoffs):
    workdir.mkdir(parents=True, exist_ok=True)
    atoms = read(structure)
    atoms.write(workdir / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP_MODEL}']
    if d3_cutoffs:
        lines.append(f'dftd3 pbe {d3_cutoffs[0]} {d3_cutoffs[1]}')
    lines += ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
    header = (workdir / 'dump.xyz').read_text().splitlines()[1]
    energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float).reshape(3, 3)
    forces = read(workdir / 'dump.xyz', format='extxyz').get_forces()
    return energy, forces, virial


def d3_errors(name, tmp_path):
    """Return (|dE|, max|dF|, max|dVirial|) of GPUMD's D3 contribution vs the stored reference."""
    ref = REFERENCE[name]
    cutoffs = (ref['potential_cutoff'], ref['cn_cutoff'])
    structure = FIXTURE_DIR / f'{name}.xyz'
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    e1, f1, v1 = _run(tmp_path / 'D3', structure, cutoffs)
    return (abs((e1 - e0) - ref['energy']),
            float(np.abs((f1 - f0) - np.array(ref['forces'])).max()),
            float(np.abs((v1 - v0) - np.array(ref['virial'])).max()))


def tolerance(name, quantity):
    ref = REFERENCE[name]
    cap = {'energy': REL_CAP * abs(ref['energy']),
           'force': FORCE_CAP,
           'virial': REL_CAP * float(np.abs(np.array(ref['virial'])).max())}[quantity]
    baseline = BASELINE[name][['energy', 'force', 'virial'].index(quantity)]
    return min(cap, max(2 * baseline, FLOORS[quantity]))


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', CASES)
def test_dftd3_matches_reference(name, tmp_path):
    d_e, d_f, d_v = d3_errors(name, tmp_path)
    assert d_e <= tolerance(name, 'energy'), f'energy error {d_e:.3e} eV'
    assert d_f <= tolerance(name, 'force'), f'force error {d_f:.3e} eV/A'
    assert d_v <= tolerance(name, 'virial'), f'virial error {d_v:.3e} eV'
