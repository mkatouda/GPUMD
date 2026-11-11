"""DFT-D3 (`dftd3` keyword) when the box crosses the small-box / large-box threshold during a run.

Requirement DR-006 (specs/requirements/DR-006-dftd3-cell-buffers.md), acceptance criteria AC-1 and
AC-2; record: specs/upstream-issues/UPSTREAM-0003-dftd3-cell-buffers.md.

`DFTD3::compute_small_box` is used while a periodic thickness is <= 2.5 x the radial cutoff and
`compute_large_box` (cell lists) above it. A run that starts in the small-box path and later
crosses the threshold (`deform`, NPT) used to abort at force/neighbor.cu:193 ("invalid argument")
because the cell-list buffers were allocated only together with the CN buffers of the large-box
path. Here `dftd3 pbe 4 2` gives a threshold of 10.0 A; water27 (9.3 A) is scaled to start below
or above it and `deform` (0.005 A/step, 200 steps) pushes the box across it. With `time_step 0` only the box moves (atoms scale affinely), so runs with and without D3 follow
the same trajectory; at each segment end the in-run D3 energy (difference of the two runs) is
compared with the D3 energy of a static single-path run of the dumped frame (cap: DR-001 AC-1,
relative 2e-6 of the D3 energy; not loosened; observed agreement ~5e-9 eV after the fix).
"""
import re
import subprocess

import numpy as np
import pytest
from ase.io import read, write

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd3'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
D3 = 'dftd3 pbe 4 2'          # radial cutoff 4 A: small-box path while thickness <= 10.0 A
THRESHOLD = 10.0
REL_CAP = 2e-6                # DR-001 AC-1, fraction of |E_D3|
RATE = 0.005                  # A/step per axis
STEPS = 200                   # 1.0 A per segment

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
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    return result


def _deform_segments(workdir, length0, rates, with_d3):
    """time_step 0 + deform (atoms scale affinely with the box, so runs with and without D3 follow
    the same trajectory); one `run` per entry of `rates`. Returns (frames, in-run energies)."""
    _structure(workdir, length0)
    lines = [f'potential {NEP_MODEL}'] + ([D3] if with_d3 else []) + ['time_step 0']
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


def _static_energy(workdir, atoms, with_d3):
    workdir.mkdir(parents=True, exist_ok=True)
    write(workdir / 'model.xyz', atoms, format='extxyz')
    lines = [f'potential {NEP_MODEL}'] + ([D3] if with_d3 else []) + [
        'ensemble nve', 'time_step 0', 'dump_xyz 1 d.xyz precision double', 'run 1']
    result = _gpumd(workdir, lines)
    assert result.returncode == 0, result.stdout[-1500:]
    header = (workdir / 'd.xyz').read_text().splitlines()[1]
    return float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))


def _check(tmp_path, length0, rates):
    # the run with D3 first: on the unmodified binary it aborts when the path changes
    frames, e_d3_run = _deform_segments(tmp_path / 'd3', length0, rates, True)
    _, e_nod3_run = _deform_segments(tmp_path / 'nod3', length0, rates, False)
    length = length0
    for k, (rate, frame) in enumerate(zip(rates, frames)):
        length += rate * STEPS
        assert abs(frame.cell[0, 0] - length) < 1e-6
        # D3 energy of the in-run (transition) path vs the single-path static run of the same frame
        # (the total energy is not compared: float NEP summation noise is ~1e-5 eV)
        e_in = e_d3_run[k] - e_nod3_run[k]
        e_ref = (_static_energy(tmp_path / f'static_d3_{k}', frame, True)
                 - _static_energy(tmp_path / f'static_nod3_{k}', frame, False))
        assert abs(e_ref) > 1e-3
        assert abs(e_in - e_ref) <= REL_CAP * abs(e_ref), (k, e_in, e_ref)


# small -> large (the DR-006 bug), large -> small, and the two round trips
@pytest.mark.parametrize('length0, rates', [
    (9.3, [RATE]),                       # small -> large   (9.3 -> 10.3)
    (10.6, [-RATE]),                     # large -> small   (10.6 -> 9.6)
    (9.3, [RATE, -RATE]),                # small -> large -> small
    (10.6, [-RATE, RATE]),               # large -> small -> large
], ids=['small_to_large', 'large_to_small', 'small_large_small', 'large_small_large'])
def test_dftd3_box_path_transition(tmp_path, length0, rates):
    lengths = np.cumsum([length0] + [r * STEPS for r in rates])
    sides = [length > THRESHOLD for length in lengths]
    assert all(a != b for a, b in zip(sides[:-1], sides[1:])), lengths   # every segment crosses
    _check(tmp_path, length0, rates)
