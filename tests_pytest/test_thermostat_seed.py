"""DR-009: optional trailing `seed <n>` of `ensemble nvt_bdp` and `ensemble npt_scr`.

AC-1: with an all-zero NEP there are no forces, so the trajectory depends on the thermostat
random stream only (velocity creation is fixed by `velocity ... seed`). Same seed: bit-identical
dump and thermo; different seeds: different; no seed: the clock seed, two runs differ.
(The last check needs a binary built without -DDEBUG, which fixes the seed of the old behaviour.)
AC-3: refusals of a zero, negative, non-integer, missing or repeated seed.
The seed fixes only the thermostat stream; it does not make trajectories of interacting systems
reproducible (GPU force accumulation), see doc/gpumd/input_parameters/ensemble.rst.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from ase.io import read, write

from conftest import GPUMD_EXECUTABLE

FIXTURES = Path(__file__).parent / 'fixtures' / 'dftd3_zero'
ZERO_NEP = FIXTURES / 'zero_nep_H_C_O_Cu.txt'
STRUCTURE = FIXTURES / 'tiny24.xyz'

NVT = 'ensemble nvt_bdp 300 300 20'
NPT = 'ensemble npt_scr 300 300 20 0 0 0 50 50 50 100'


def run(workdir, ensemble, seed=None, steps=200):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'model.xyz').write_text(STRUCTURE.read_text())
    ens = ensemble if seed is None else f'{ensemble} {seed}'
    lines = [f'potential {ZERO_NEP}', 'velocity 300 seed 7', ens, 'time_step 1',
             'dump_thermo 10', f'dump_xyz {steps} dump.xyz velocity precision double',
             f'run {steps}']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    return result


def outputs(workdir):
    thermo = (workdir / 'thermo.out').read_text()
    # the last frame: positions and velocities after `steps` steps
    atoms = read(workdir / 'dump.xyz', format='extxyz', index=-1)
    return thermo, atoms.get_positions().tobytes() + atoms.get_velocities().tobytes()


def ok_run(workdir, ensemble, seed=None):
    result = run(workdir, ensemble, seed)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    return outputs(workdir)


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
def test_same_seed_identical_different_seed_differs(tmp_path, ensemble):
    a = ok_run(tmp_path / 'a', ensemble, 'seed 12345')
    time.sleep(1.1)
    b = ok_run(tmp_path / 'b', ensemble, 'seed 12345')
    c = ok_run(tmp_path / 'c', ensemble, 'seed 54321')
    assert a == b
    assert a[0] != c[0] and a[1] != c[1]


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
def test_no_seed_keeps_clock_seed(tmp_path, ensemble):
    a = ok_run(tmp_path / 'a', ensemble)
    time.sleep(1.1)
    b = ok_run(tmp_path / 'b', ensemble)
    if a == b:
        pytest.skip('identical runs without seed: binary built with -DDEBUG (fixed seed)?')
    assert a != b


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
@pytest.mark.parametrize('seed', ['seed 0', 'seed -1', 'seed 1.5', 'seed abc', 'seed',
                                  'seed 3 seed 4'])
def test_invalid_seed_refused(tmp_path, ensemble, seed):
    result = run(tmp_path, ensemble, seed, steps=1)
    assert result.returncode != 0
    assert 'seed' in result.stdout + result.stderr


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
def test_valid_seed_accepted(tmp_path, ensemble):
    result = run(tmp_path, ensemble, 'seed 1', steps=1)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1000:]
    assert 'thermostat random seed is 1' in result.stdout


# ---- several runs in one script: an ensemble line applies to one run, so every seeded ensemble
# line starts its own stream from its seed (the generator and its Gaussian cache are reset)

def run_script(workdir, body, dumps=('dump.xyz',), natoms=24, cell_scale=1.0):
    workdir.mkdir(parents=True, exist_ok=True)
    atoms = read(STRUCTURE, format='extxyz')[:natoms]
    atoms.set_cell(atoms.cell * cell_scale, scale_atoms=True)
    write(workdir / 'model.xyz', atoms, format='extxyz')
    lines = [f'potential {ZERO_NEP}', 'velocity 300 seed 7'] + body
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    frames = [read(workdir / name, format='extxyz', index=-1) for name in dumps]
    return [(f.get_positions(), f.arrays['vel']) for f in frames]


def dump(name):
    return f'dump_xyz 50 {name} velocity precision double'


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
def test_two_seeded_runs_reproducible_and_seed_dependent(tmp_path, ensemble):
    def script(seed2):
        return ['time_step 1', f'{ensemble} seed 5', dump('d1.xyz'), 'run 50',
                f'{ensemble} seed {seed2}', dump('d2.xyz'), 'run 50']
    names = ('d1.xyz', 'd2.xyz')
    a = run_script(tmp_path / 'a', script(5), names)
    b = run_script(tmp_path / 'b', script(5), names)
    c = run_script(tmp_path / 'c', script(6), names)
    # the same script gives identical output in two separate processes
    assert all(np.array_equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q))
    # the second run follows its own seed: the first run is common, the second differs
    assert all(np.array_equal(x, y) for x, y in zip(a[0], c[0]))
    assert not np.array_equal(a[1][1], c[1][1])
    assert not np.array_equal(a[0][1], a[1][1])


@pytest.mark.parametrize('ensemble', [NVT, NPT], ids=['nvt_bdp', 'npt_scr'])
@pytest.mark.parametrize('warmup', [7, 8])
def test_seeded_ensemble_after_unseeded_run_is_fresh(tmp_path, ensemble, warmup):
    """The Gaussian cache of the generator is reset when a seeded stream starts: the seeded part
    gives the same result as in a fresh process. The unseeded warm-up run (zero time step, so the
    positions stay) uses the generator and, for an odd number of draws, leaves a cached Gaussian
    number; `velocity 300 seed 7` afterwards restores the initial state. Both parities of the
    warm-up length are tested. For npt_scr the warm-up also changes the box and the positions;
    with the zero NEP the velocities depend only on the kinetic energy and the generator (see
    test_npt_scr_velocities_nearly_independent_of_cell), so only the velocities are compared there."""
    seeded = [f'{ensemble} seed 5', dump('dump.xyz'), 'run 50']
    # the number of Gaussian draws per step depends on the parity of the degrees of freedom:
    # 24 and 23 atoms cover both cases (one draw per step leaves a cached number after an odd
    # number of steps)
    for natoms in (24, 23):
        check_fresh(tmp_path / f'n{natoms}', ensemble, warmup, seeded, natoms)


def check_fresh(tmp_path, ensemble, warmup, seeded, natoms):
    fresh = run_script(tmp_path / 'fresh', ['time_step 1'] + seeded, natoms=natoms)
    after = run_script(
        tmp_path / 'after',
        ['time_step 0', ensemble, f'run {warmup}', 'velocity 300 seed 7', 'time_step 1'] + seeded,
        natoms=natoms)
    # velocities (the thermostat result) are bit-identical; positions agree to rounding (the
    # zero-time-step warm-up re-wraps them at the 1e-15 level)
    if ensemble == NVT:
        assert np.array_equal(after[0][1], fresh[0][1])
        assert np.allclose(after[0][0], fresh[0][0], rtol=0, atol=1e-10)
    else:
        # npt_scr: the velocities depend on the cell only at the 1e-11 level (see the premise
        # test); a different Gaussian sample would change them by orders of magnitude more
        assert np.abs(after[0][1] - fresh[0][1]).max() < 1e-9


@pytest.mark.parametrize('natoms', [24, 23])
def test_npt_scr_velocities_nearly_independent_of_cell(tmp_path, natoms):
    """Premise of the npt_scr check above: with a zero NEP (no forces, no virial) the final
    velocities of a seeded npt_scr run depend on the starting cell or positions only at the
    1e-11 level (not bit for bit: measured 1e-11 for a 10 % change of the cell, against a
    velocity scale of 2e-2)."""
    body = ['time_step 1', f'{NPT} seed 5', dump('dump.xyz'), 'run 50']
    a = run_script(tmp_path / 'a', body, natoms=natoms)
    b = run_script(tmp_path / 'b', body, natoms=natoms, cell_scale=1.1)
    assert np.abs(a[0][1] - b[0][1]).max() < 1e-9
    assert not np.allclose(a[0][0], b[0][0], atol=1e-3)  # the cells differ
