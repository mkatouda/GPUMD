"""DFT-D3 (`dftd3` keyword) against stored values from an independent D3 implementation.

Requirement DR-001 (specs/requirements/DR-001-dftd3-gpu-performance.md), acceptance criterion
AC-1. The reference values in fixtures/dftd3/reference.json come from simple-dftd3 (PBE, BJ
damping, two-body only, same cutoffs as GPUMD); see fixtures/dftd3/generate_reference.py for the
provenance. They are never regenerated to make this test pass.

The D3 contribution is isolated as (run with `dftd3`) - (same run without it), using NEP89 for
the underlying potential; both runs are static (`time_step 0`, one step). Cases cover both code
paths of src/force/dftd3/dftd3.cu: the "small box" path (a periodic thickness <= 2.5 x the pair
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
from ase.io import read, write

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
    _run.last_stdout = result.stdout
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


# GPUMD_DFTD3_SMALL_BOX_NUM_SUB (a test aid, see dftd3.cu) fixes the number of threads per atom of
# the small-box force kernels. Every instantiation (1, 2, ..., 128 threads per atom, with the
# fixed-order shared-memory reduction for more than one) must agree with one thread per atom within
# the AC-1 caps (a different summation order only), must itself meet the AC-1 tolerance against the
# stored reference, and must be bit-identical from run to run. The large-box cases (the two `_large`
# cases and the non-periodic benzene dimer) do not use the variable and print nothing.
LARGE_BOX_CASES = ['benzene_dimer', 'water_large', 'cu_large']
SMALL_BOX_CASES = [name for name in CASES if name not in LARGE_BOX_CASES]
NUM_SUB_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', SMALL_BOX_CASES)
def test_dftd3_small_box_all_threads_per_atom_agree_and_are_reproducible(name, tmp_path, monkeypatch):
    ref = REFERENCE[name]
    cutoffs = (ref['potential_cutoff'], ref['cn_cutoff'])
    structure = FIXTURE_DIR / f'{name}.xyz'
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    results = {}
    for num_sub in NUM_SUB_VALUES:
        monkeypatch.setenv('GPUMD_DFTD3_SMALL_BOX_NUM_SUB', str(num_sub))
        runs = []
        for k in range(2):
            runs.append(_run(tmp_path / f's{num_sub}_{k}', structure, cutoffs))
            assert f'threads per atom = {num_sub}' in _run.last_stdout
        (e1, f1, v1), (e2, f2, v2) = runs
        assert e1 == e2 and np.array_equal(f1, f2) and np.array_equal(v1, v2), num_sub
        results[num_sub] = (e1 - e0, f1 - f0, v1 - v0)
    e_ref, f_ref, v_ref = results[1]
    for num_sub, (e, f, v) in results.items():
        d_e = abs(e - ref['energy'])
        d_f = float(np.abs(f - np.array(ref['forces'])).max())
        d_v = float(np.abs(v - np.array(ref['virial'])).max())
        assert d_e <= tolerance(name, 'energy'), (num_sub, d_e)
        assert d_f <= tolerance(name, 'force'), (num_sub, d_f)
        assert d_v <= tolerance(name, 'virial'), (num_sub, d_v)
        assert abs(e - e_ref) <= REL_CAP * abs(ref['energy']), num_sub
        assert np.abs(f - f_ref).max() <= FORCE_CAP, num_sub
        assert np.abs(v - v_ref).max() <= REL_CAP * np.abs(np.array(ref['virial'])).max(), num_sub


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', LARGE_BOX_CASES)
def test_dftd3_large_box_ignores_small_box_num_sub(name, tmp_path, monkeypatch):
    monkeypatch.setenv('GPUMD_DFTD3_SMALL_BOX_NUM_SUB', '4')
    ref = REFERENCE[name]
    _run(tmp_path / 'D3', FIXTURE_DIR / f'{name}.xyz', (ref['potential_cutoff'], ref['cn_cutoff']))
    assert 'threads per atom' not in _run.last_stdout


# Large-box path: the CN kernel stores the partners it accepts in a per-thread list that the extra
# kernel reuses instead of repeating the cell traversal; a slot whose partners do not all fit is
# handled by the traversal. GPUMD_DFTD3_LARGE_BOX_LIST_CAPACITY (a test aid, see dftd3.cu) fixes the
# capacity per thread (0 switches the list off), GPUMD_DFTD3_LARGE_BOX_NUM_SUB fixes the number of
# threads per atom. The list is built in the order of the traversal, so list, overflow fallback and
# plain traversal must give identical numbers.
LIST_CAPACITIES = (None, '0', '1', '4')   # None = default rule


def _large_box_env(monkeypatch, num_sub, capacity):
    for key, value in (('GPUMD_DFTD3_LARGE_BOX_NUM_SUB', num_sub),
                       ('GPUMD_DFTD3_LARGE_BOX_LIST_CAPACITY', capacity)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd3_large_box_neighbor_list_is_exact_and_reproducible(tmp_path, monkeypatch):
    # 24 atoms in a 40 A periodic box: large-box path, one warp, so the order of the cell lists (and
    # with it the summation order) is the same in every run and the numbers can be compared bitwise.
    atoms = read(FIXTURE_DIR / 'water27.xyz')[:24]
    atoms.set_cell([40.0, 40.0, 40.0])
    atoms.set_pbc(True)
    atoms.center()
    structure = tmp_path / 'tiny24.xyz'
    write(structure, atoms, format='extxyz')
    _large_box_env(monkeypatch, None, None)
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    results = {}
    for num_sub in (1, 2, 4, 8, 16):
        reference = None
        for capacity in LIST_CAPACITIES:
            _large_box_env(monkeypatch, num_sub, capacity)
            runs = [_run(tmp_path / f's{num_sub}_c{capacity}_{k}', structure, (12.0, 6.0))
                    for k in range(2 if capacity is None else 1)]
            assert f'threads per atom = {num_sub}' in _run.last_stdout
            if capacity is not None:
                assert f'capacity per thread = {capacity}' in _run.last_stdout
            if len(runs) == 2:
                assert runs[0][0] == runs[1][0] and np.array_equal(runs[0][1], runs[1][1]) \
                    and np.array_equal(runs[0][2], runs[1][2]), (num_sub, 'not reproducible')
            e1, f1, v1 = runs[0]
            if reference is None:
                reference = (e1, f1, v1)
            assert e1 == reference[0] and np.array_equal(f1, reference[1]) \
                and np.array_equal(v1, reference[2]), (num_sub, capacity)
        results[num_sub] = (reference[0] - e0, reference[1] - f0, reference[2] - v0)
    e_ref, f_ref, v_ref = results[1]
    for num_sub, (e, f, v) in results.items():
        assert abs(e - e_ref) <= REL_CAP * abs(e_ref), num_sub
        assert np.abs(f - f_ref).max() <= FORCE_CAP, num_sub
        assert np.abs(v - v_ref).max() <= REL_CAP * np.abs(v_ref).max(), num_sub


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', LARGE_BOX_CASES)
def test_dftd3_large_box_neighbor_list_capacities_agree(name, tmp_path, monkeypatch):
    ref = REFERENCE[name]
    structure = FIXTURE_DIR / f'{name}.xyz'
    _large_box_env(monkeypatch, None, None)
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    results = {}
    for capacity in LIST_CAPACITIES:
        _large_box_env(monkeypatch, None, capacity)
        e1, f1, v1 = _run(tmp_path / f'c{capacity}', structure,
                          (ref['potential_cutoff'], ref['cn_cutoff']))
        results[capacity] = (e1 - e0, f1 - f0, v1 - v0)
    e_ref, f_ref, v_ref = results['0']
    for capacity, (e, f, v) in results.items():
        assert abs(e - ref['energy']) <= tolerance(name, 'energy'), capacity
        assert np.abs(f - np.array(ref['forces'])).max() <= tolerance(name, 'force'), capacity
        assert np.abs(v - np.array(ref['virial'])).max() <= tolerance(name, 'virial'), capacity
        assert abs(e - e_ref) <= REL_CAP * abs(ref['energy']), capacity
        assert np.abs(f - f_ref).max() <= FORCE_CAP, capacity
        assert np.abs(v - v_ref).max() <= REL_CAP * np.abs(np.array(ref['virial'])).max(), capacity


# Block size not dividing N, several blocks, many threads per atom: 130 atoms (a truncated 2 x 1 x 1
# tiling of water27, 18.6 x 9.3 x 9.3 A, small-box path) with 1, 64 and 128 threads per atom must
# agree with one thread per atom within the AC-1 caps and be bit-reproducible. No independent
# reference exists for this structure (the stored references cover the fixtures only).
@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd3_small_box_130_atoms_many_threads_per_atom(tmp_path, monkeypatch):
    atoms = read(FIXTURE_DIR / 'water27.xyz')
    atoms = atoms.repeat((2, 1, 1))[:130]
    structure = tmp_path / 'water130.xyz'
    write(structure, atoms, format='extxyz')
    assert len(atoms) == 130
    monkeypatch.delenv('GPUMD_DFTD3_SMALL_BOX_NUM_SUB', raising=False)
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    results = {}
    for num_sub in (1, 64, 128):
        monkeypatch.setenv('GPUMD_DFTD3_SMALL_BOX_NUM_SUB', str(num_sub))
        runs = [_run(tmp_path / f's{num_sub}_{k}', structure, (12.0, 6.0)) for k in range(2)]
        assert f'threads per atom = {num_sub}' in _run.last_stdout
        (e1, f1, v1), (e2, f2, v2) = runs
        assert e1 == e2 and np.array_equal(f1, f2) and np.array_equal(v1, v2), num_sub
        results[num_sub] = (e1 - e0, f1 - f0, v1 - v0)
    e_ref, f_ref, v_ref = results[1]
    for num_sub, (e, f, v) in results.items():
        assert abs(e - e_ref) <= REL_CAP * abs(e_ref), (num_sub, abs(e - e_ref))
        assert np.abs(f - f_ref).max() <= FORCE_CAP, (num_sub, np.abs(f - f_ref).max())
        assert np.abs(v - v_ref).max() <= REL_CAP * np.abs(v_ref).max(), num_sub


# Constant-pressure style run of the large-box path: `deform` with npt_ber (npt_scr has a stochastic
# cell rescaling, so two runs differ at 1e-3 eV after a few steps; without `velocity` the initial
# velocities are not reproducible either) changes the box strongly
# (compression raises the expected number of partners, so the CN list buffer must grow: water_large
# at -0.069 A/step grows it twice, 25% headroom included; expansion changes the number of cells). Energies must stay finite, and the per-step potential energy with
# the neighbor list (default rule) must agree with the run without the list (capacity 0) within the
# AC-1 energy cap (relative to the D3 energy) over the first steps; the whole run must complete.
def _run_deform(workdir, structure, rate, cutoffs, steps):
    workdir.mkdir(parents=True, exist_ok=True)
    read(structure).write(workdir / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP_MODEL}', f'dftd3 pbe {cutoffs[0]} {cutoffs[1]}' if cutoffs else '',
             'velocity 300 seed 12345', 'time_step 0.1',
             'ensemble npt_ber 300 300 100 0 0 0 100 100 100 1000',
             f'deform xx {rate} yy {rate} zz {rate}', 'dump_thermo 1', f'run {steps}']
    (workdir / 'run.in').write_text('\n'.join(line for line in lines if line) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
    return np.loadtxt(workdir / 'thermo.out')


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name, rate', [('water_large', -0.069), ('water_large', 0.2),
                                        ('cu_large', -0.0427), ('cu_large', 0.2)])
def test_dftd3_large_box_changing_box_list_matches_no_list(name, rate, tmp_path, monkeypatch):
    ref = REFERENCE[name]
    cutoffs = (ref['potential_cutoff'], ref['cn_cutoff'])
    structure = FIXTURE_DIR / f'{name}.xyz'
    steps = 100
    _large_box_env(monkeypatch, None, None)
    base = _run_deform(tmp_path / 'noD3', structure, rate, None, 5)
    with_list = _run_deform(tmp_path / 'list', structure, rate, cutoffs, steps)
    _large_box_env(monkeypatch, None, 0)
    no_list = _run_deform(tmp_path / 'nolist', structure, rate, cutoffs, steps)
    assert with_list.shape[0] == steps and np.isfinite(with_list).all()
    assert np.isfinite(no_list).all()
    u_list, u_none = with_list[:, 2], no_list[:, 2]       # thermo.out: T, K, U, ...
    e_d3 = abs(u_list[0] - base[0, 2])
    assert e_d3 > 0
    assert np.abs(u_list[:5] - u_none[:5]).max() <= REL_CAP * e_d3
    # the box really changed a lot (column 10 of thermo.out: ax)
    assert abs(with_list[-1, 9] / with_list[0, 9] - 1.0) > 0.1


# The expected number of partners per thread (list capacity rule) is huge for a degenerate cell of a
# non-periodic system (volume 1e-6 A^3); the capacity must be clamped, not converted from an
# out-of-range double. The result must still meet the reference of the same molecule.
@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd3_non_periodic_degenerate_cell_matches_reference(tmp_path, monkeypatch):
    name = 'benzene_dimer'
    ref = REFERENCE[name]
    atoms = read(FIXTURE_DIR / f'{name}.xyz')
    atoms.set_cell([0.01, 0.01, 0.01])
    atoms.set_pbc(False)
    structure = tmp_path / 'dimer_tiny_cell.xyz'
    write(structure, atoms, format='extxyz')
    _large_box_env(monkeypatch, None, None)
    e0, f0, v0 = _run(tmp_path / 'noD3', structure, None)
    e1, f1, v1 = _run(tmp_path / 'D3', structure, (ref['potential_cutoff'], ref['cn_cutoff']))
    assert abs((e1 - e0) - ref['energy']) <= tolerance(name, 'energy')
    assert np.abs((f1 - f0) - np.array(ref['forces'])).max() <= tolerance(name, 'force')
