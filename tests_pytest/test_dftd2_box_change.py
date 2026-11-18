"""DFT-D2 (`dftd2` keyword) when the box changes size during a run (growth and shrinkage).

Counterpart of test_dftd3_cell_buffers.py (DR-006). `dftd3` has a small-box and a large-box path
and once aborted when a run crossed the threshold; `dftd2` has a single cell-list path whose
buffers and cell counts are sized on every call (`ensure_size`, `choose_cell_counts` in
src/force/dftd2.cu), so no such crash is expected. This test checks it by running, not by reading:
water27 (81 atoms, 9.3 A cube) with `dftd2 0.75 8`; `time_step 0` plus `deform` moves only the box
(atoms scale affinely), so runs with and without D2 follow the same trajectory. After each segment
(200 steps of 0.02 A per axis = 4 A) the in-run D2 energy (difference of the two runs) is compared
with the D2 energy of a static single-call run of the dumped frame (cap: relative 2e-6 of the D2
energy, DR-001 AC-1 level, not loosened; the total energy is not compared: float NEP summation
noise). The sequences cover growth (the cell counts increase), shrinkage (they decrease), and
round trips, over box lengths 9.3 .. 21.3 A, i.e. from a box smaller than the cutoff diameter
(periodic images) to a box several cutoffs thick.
"""
import re
import subprocess

import numpy as np
import pytest
from ase.io import read, write

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd3'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
D2 = 'dftd2 0.75 8'
REL_CAP = 2e-6                # DR-001 AC-1, fraction of |E_D2|
RATE = 0.02                   # A/step per axis
STEPS = 200                   # 4.0 A per segment

pytestmark = [pytest.mark.fast,
              pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')]


def _structure(workdir, length):
    atoms = read(FIXTURE_DIR / 'water27.xyz')
    atoms.set_cell(np.eye(3) * length, scale_atoms=True)
    workdir.mkdir(parents=True, exist_ok=True)
    write(workdir / 'model.xyz', atoms, format='extxyz')
    return atoms


def _gpumd(workdir, lines):
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    return subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)


def _deform_segments(workdir, length0, rates, with_d2):
    """One `run` per entry of `rates`; returns (frames, in-run energies) at the segment ends."""
    _structure(workdir, length0)
    lines = [f'potential {NEP_MODEL}'] + ([D2] if with_d2 else []) + ['time_step 0']
    for rate in rates:
        lines += ['ensemble nve', f'deform xx {rate} yy {rate} zz {rate}',
                  f'dump_xyz {STEPS} dump.xyz force precision double', f'run {STEPS}']
    result = _gpumd(workdir, lines)
    assert result.returncode == 0, result.stdout[-1500:]
    frames = read(workdir / 'dump.xyz', index=':', format='extxyz')
    headers = [line for line in (workdir / 'dump.xyz').read_text().splitlines() if 'Lattice' in line]
    assert len(frames) == len(rates) == len(headers), (len(frames), len(headers))
    energies = [float(re.search(r'energy=([-0-9.eE+]+)', h).group(1)) for h in headers]
    return frames, energies


def _static_energy(workdir, atoms, with_d2):
    workdir.mkdir(parents=True, exist_ok=True)
    write(workdir / 'model.xyz', atoms, format='extxyz')
    lines = [f'potential {NEP_MODEL}'] + ([D2] if with_d2 else []) + [
        'ensemble nve', 'time_step 0', 'dump_xyz 1 d.xyz precision double', 'run 1']
    result = _gpumd(workdir, lines)
    assert result.returncode == 0, result.stdout[-1500:]
    header = (workdir / 'd.xyz').read_text().splitlines()[1]
    return float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))


@pytest.mark.parametrize('length0, rates', [
    (9.3, [RATE, RATE, RATE]),            # growth 9.3 -> 21.3
    (21.3, [-RATE, -RATE, -RATE]),        # shrinkage 21.3 -> 9.3
    (9.3, [RATE, RATE, -RATE, -RATE]),    # up and back down (9.3 -> 17.3 -> 9.3)
    (21.3, [-RATE, -RATE, RATE, RATE]),   # down and back up (21.3 -> 13.3 -> 21.3)
], ids=['grow', 'shrink', 'grow_shrink', 'shrink_grow'])
def test_dftd2_box_size_changes(tmp_path, length0, rates):
    frames, e_d2_run = _deform_segments(tmp_path / 'd2', length0, rates, True)
    _, e_nod2_run = _deform_segments(tmp_path / 'nod2', length0, rates, False)
    length = length0
    for k, (rate, frame) in enumerate(zip(rates, frames)):
        length += rate * STEPS
        assert abs(frame.cell[0, 0] - length) < 1e-6
        e_in = e_d2_run[k] - e_nod2_run[k]
        e_ref = (_static_energy(tmp_path / f'static_d2_{k}', frame, True)
                 - _static_energy(tmp_path / f'static_nod2_{k}', frame, False))
        assert abs(e_ref) > 1e-3
        assert abs(e_in - e_ref) <= REL_CAP * abs(e_ref), (k, e_in, e_ref)
