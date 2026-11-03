"""DFT-D2 (`dftd2` keyword): parameter table, independent reference, and GPUMD agreement.

Three groups of tests:

1. CPU only (must pass without the feature): the generated C6/R0 table against the dftd3-lib
   source, the NumPy reference `dftd2_reference.py` against the dftd3-lib D2-mode oracle (stored
   values in fixtures/dftd2/oracle_reference.json, unshifted limit, plus the shift terms added
   analytically), and self-consistency of the reference (finite-difference forces, strain-
   derivative virial, translation / permutation / periodic-image invariance, continuity at rc).
2. GPU (`gpumd dftd2 <s6> <rc>`): the D2 contribution, isolated as (run with `dftd2`) - (same run
   without it) on NEP89, against the reference with the same shifted-force definition.
3. GPU input errors.  The expected message patterns below may be adapted if the wording changes.  Every pattern also requires that the message is NOT the generic "invalid keyword"
   error GPUMD prints for an unknown keyword today.

Tolerances (fixed BEFORE any GPU measurement; never to be loosened to make a test pass):
- oracle (unshifted, stored dftd3-lib values): energy relative 1e-9, force 5e-9 of the largest
  force component, virial 5e-9 of the largest virial component.  Observed on 2026-11-03: <= 3.2e-11,
  1.2e-9, 3.8e-11 (periodic cases show a ~1e-9 force residual of the oracle itself, cause not found).
- reference self-consistency: finite-difference force / strain virial relative 1e-7.
- GPU vs reference: energy and virial relative 2e-6 of |E_ref| and
  max|W_ref|; force 1e-5 of the largest reference force component), with no absolute floors.  The D2 part
  is obtained as a difference of two float NEP runs, so s6 is chosen large (75) to make the D2 signal much
  larger than the float run-to-run noise of NEP.  Observed (s6 = 75, all 17 GPU cases including the
  degenerate ones, 2026-11-03): relative errors <= 3.3e-7 (energy), <= 5.9e-7 (virial), <= 1.7e-6
  (force, of the largest component).
"""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

import dftd2_reference as ref
from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd2'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
DFTD3_LIB_SOURCE = Path(os.environ.get(
    'DFTD3_LIB_EXTRAS_F90', '/home/katouda/work/dielectric-response_pj/dftd3-lib/prg/extras.f90'))
ORACLE = json.load(open(FIXTURE_DIR / 'oracle_reference.json'))
ORACLE_CASES = [k for k in ORACLE if not k.startswith('_')]

ORACLE_E_REL = 1e-9
ORACLE_F_REL = 5e-9
ORACLE_W_REL = 5e-9
SELF_REL = 1e-7
BOHR = 0.52917726


def structure(name):
    return read(FIXTURE_DIR / f'{name}.xyz')


# ---------------------------------------------------------------------------------------------
# 1a. parameter table
# ---------------------------------------------------------------------------------------------
def _independent_parse(path):
    """Second, deliberately different parser of loadoldpar (line based, no regex over blocks)."""
    inside, name, vals = False, None, {'r0': [], 'c6': []}
    for line in Path(path).read_text().splitlines():
        line = line.split('!')[0]
        if 'subroutine loadoldpar' in line and 'end' not in line:
            inside = True
        if 'end subroutine loadoldpar' in line:
            break
        if not inside:
            continue
        m = re.match(r'\s*(r0|c6)\(1:86\)\s*=', line)
        if m:
            name = m.group(1)
        if name:
            vals[name] += [float(x.replace('d0', '')) for x in re.findall(r'\d+\.\d*d0', line)]
        if '/)' in line:
            name = None
    return vals['r0'], vals['c6']


def test_table_has_86_elements_and_units():
    p = ref.PARAMS
    assert len(p['symbol']) == 87 and p['symbol'][1] == 'H' and p['symbol'][86] == 'Rn'
    assert np.all(np.isfinite(p['c6'][1:])) and np.all(np.isfinite(p['R0'][1:]))
    # c6 in eV A^6 recomputed from J nm^6/mol with the documented constants
    rows = [ln.split(',') for ln in (FIXTURE_DIR / 'dftd2_params.csv').read_text().splitlines()
            if ln and not ln.startswith('#')][1:]
    c6conv = 1e-3 / 2625.4999 / (0.52917726 / 10) ** 6          # Eh bohr^6 per J nm^6 mol^-1
    for row in rows:
        z, c6_raw = int(row[0]), float(row[2])
        assert p['c6'][z] == pytest.approx(c6_raw * c6conv * 27.21138505 * 0.52917726 ** 6, rel=1e-14)
        assert p['R0'][z] == pytest.approx(1.1 * float(row[3]), rel=1e-14)
    assert len(rows) == 86


@pytest.mark.skipif(not DFTD3_LIB_SOURCE.exists(), reason='dftd3-lib source not available')
def test_table_equals_dftd3_lib_all_86():
    r0, c6 = _independent_parse(DFTD3_LIB_SOURCE)
    assert len(r0) == 86 and len(c6) == 86
    rows = [ln.split(',') for ln in (FIXTURE_DIR / 'dftd2_params.csv').read_text().splitlines()
            if ln and not ln.startswith('#')][1:]
    for row in rows:
        z = int(row[0])
        assert float(row[2]) == c6[z - 1], f'C6 mismatch for Z={z}'
        assert float(row[3]) == r0[z - 1], f'r0 mismatch for Z={z}'


@pytest.mark.skipif(not DFTD3_LIB_SOURCE.exists(), reason='dftd3-lib source not available')
def test_csv_is_exactly_the_generator_output(tmp_path):
    """The CSV is generated, not typed: regenerating it from the source reproduces it byte for byte."""
    spec = importlib.util.spec_from_file_location('generate_params', FIXTURE_DIR / 'generate_params.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    commit = re.search(r'commit (\w+)', (FIXTURE_DIR / 'dftd2_params.csv').read_text()).group(1)
    mod.main(DFTD3_LIB_SOURCE, commit, tmp_path / 'regenerated.csv')
    assert (tmp_path / 'regenerated.csv').read_text() == (FIXTURE_DIR / 'dftd2_params.csv').read_text()


def test_group_fills_are_single_source():
    """Z = 55-86 are single-source group fills (Ce-Lu one value, Ta-Au one value)."""
    c6 = ref.PARAMS['c6']
    R0 = ref.PARAMS['R0']
    for lo, hi in ((58, 71), (73, 79), (21, 30)):          # Ce-Lu, Ta-Au, Sc-Zn
        assert np.ptp(c6[lo:hi + 1]) == 0 and np.ptp(R0[lo:hi + 1]) == 0
    assert ref.PARAMS['symbol'][58] == 'Ce' and ref.PARAMS['symbol'][71] == 'Lu'


def test_hxe_matches_published_table():
    """H-Xe agree with Table 6.3 of the Turbomole manual: C6 exactly, R0 to the
    3 printed decimals."""
    lines = [ln for ln in (FIXTURE_DIR / 'turbomole_table63_hxe.csv').read_text().splitlines()
             if ln and not ln.startswith('#')][1:]
    ours = {}
    for ln in (FIXTURE_DIR / 'dftd2_params.csv').read_text().splitlines():
        if ln and not ln.startswith('#') and not ln.startswith('Z,'):
            f = ln.split(',')
            ours[f[1]] = (float(f[2]), float(f[4]))
    assert len(lines) == 54
    for ln in lines:
        sym, c6, r0 = ln.split(',')
        assert abs(ours[sym][0] - float(c6)) < 1e-9, sym
        assert abs(ours[sym][1] - float(r0)) < 6e-4, sym


# ---------------------------------------------------------------------------------------------
# 1b. reference vs dftd3-lib oracle
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name', ORACLE_CASES)
def test_reference_matches_oracle_unshifted(name):
    o = ORACLE[name]
    atoms = structure(name)
    E, F, W = ref.energy_forces_virial(atoms, o['s6'], o['rc_A'], shifted=False)
    Fo = np.array(o['forces_eV_A'])
    assert abs(E - o['energy_eV']) <= ORACLE_E_REL * abs(o['energy_eV'])
    assert np.abs(F - Fo).max() <= ORACLE_F_REL * np.abs(Fo).max()
    if 'virial_eV' in o:
        Wo = np.array(o['virial_eV'])
        assert np.abs(W - Wo).max() <= ORACLE_W_REL * np.abs(Wo).max()


@pytest.mark.parametrize('name', [n for n in ORACLE_CASES if ORACLE[n]['natoms'] <= 200])
def test_reference_shifted_matches_oracle_plus_shift_terms(name):
    """E_sf = E_oracle(rc) + shift terms enumerated by brute force (independent pair search)."""
    o = ORACLE[name]
    atoms = structure(name)
    E, F, W = ref.energy_forces_virial(atoms, o['s6'], o['rc_A'], shifted=True)
    dE, dF, dW = ref.shift_terms_bruteforce(atoms, o['s6'], o['rc_A'])
    E_exp = o['energy_eV'] + dE
    F_exp = np.array(o['forces_eV_A']) + dF
    assert abs(E - E_exp) <= ORACLE_E_REL * abs(E_exp)
    assert np.abs(F - F_exp).max() <= ORACLE_F_REL * np.abs(F_exp).max()
    if 'virial_eV' in o:
        W_exp = np.array(o['virial_eV']) + dW
        assert np.abs(W - W_exp).max() <= ORACLE_W_REL * np.abs(W_exp).max()


# ---------------------------------------------------------------------------------------------
# 1c. self-consistency of the reference (shifted force)
# ---------------------------------------------------------------------------------------------
SELF_CASES = {   # name: (s6, rc)
    'dimer': (0.75, 12.0), 'benzene_dimer': (0.75, 9.0), 'graphite': (0.75, 12.0),
    'water27_tri': (1.25, 12.0), 'tri6': (1.0, 20.0),
}


@pytest.mark.parametrize('name', list(SELF_CASES))
def test_forces_equal_minus_energy_gradient(name):
    s6, rc = SELF_CASES[name]
    atoms = structure(name)
    _, F, _ = ref.energy_forces_virial(atoms, s6, rc)
    idx = sorted({0, len(atoms) // 2, len(atoms) - 1})
    Fn = ref.numerical_forces(atoms, s6, rc, indices=idx)
    assert np.abs(Fn[idx] - F[idx]).max() <= SELF_REL * np.abs(F).max()


@pytest.mark.parametrize('name', list(SELF_CASES))
def test_virial_equals_numerical_strain_derivative(name):
    s6, rc = SELF_CASES[name]
    atoms = structure(name)
    _, _, W = ref.energy_forces_virial(atoms, s6, rc)
    Wn = ref.numerical_virial(atoms, s6, rc)
    assert np.abs(Wn - W).max() <= SELF_REL * np.abs(W).max()


@pytest.mark.parametrize('name', ['benzene_dimer', 'graphite', 'water27_tri'])
def test_invariances(name):
    s6, rc = SELF_CASES[name]
    atoms = structure(name)
    E, F, W = ref.energy_forces_virial(atoms, s6, rc)
    scale_E, scale_F = abs(E), np.abs(F).max()
    # (a) rigid translation (positions leave the cell: periodic wrapping is not applied)
    shifted = atoms.copy()
    shifted.translate([1.7, -2.3, 0.9])
    E2, F2, W2 = ref.energy_forces_virial(shifted, s6, rc)
    assert abs(E2 - E) <= 1e-11 * scale_E
    assert np.abs(F2 - F).max() <= 1e-10 * scale_F and np.abs(W2 - W).max() <= 1e-10 * np.abs(W).max()
    # (b) atom order
    perm = np.random.default_rng(7).permutation(len(atoms))
    E3, F3, W3 = ref.energy_forces_virial(atoms[perm], s6, rc)
    assert abs(E3 - E) <= 1e-11 * scale_E
    assert np.abs(F3 - F[perm]).max() <= 1e-10 * scale_F and np.abs(W3 - W).max() <= 1e-10 * np.abs(W).max()
    # (c) periodic image: move one atom by a lattice vector (only for periodic cells)
    if atoms.pbc.any():
        moved = atoms.copy()
        p = moved.positions
        p[1] += moved.cell[0] - 2 * moved.cell[2]
        moved.set_positions(p)
        E4, F4, W4 = ref.energy_forces_virial(moved, s6, rc)
        assert abs(E4 - E) <= 1e-11 * scale_E
        assert np.abs(F4 - F).max() <= 1e-10 * scale_F and np.abs(W4 - W).max() <= 1e-10 * np.abs(W).max()


def test_shifted_force_goes_to_zero_continuously_at_rc():
    rc, s6 = 12.0, 0.75
    for r_over in (1.0 - 1e-4 / rc, 1.0, 1.0 + 1e-9):
        atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, r_over * rc]], pbc=False)
        E, F, W = ref.energy_forces_virial(atoms, s6, rc)
        c6, R0 = ref.pair_parameters(6, 8)
        phi_rc, dphi_rc = ref.phi_dphi(rc, c6, R0, s6)
        if r_over >= 1.0:
            assert E == 0.0 and np.all(F == 0.0) and np.all(W == 0.0)
        else:        # quadratic approach: |E_sf| ~ 0.5 phi'' d^2, |F_sf| ~ phi'' d, with d = 1e-4 A
            assert abs(E) <= 1e-7 * abs(phi_rc)
            assert np.abs(F).max() <= 1e-3 * abs(dphi_rc)
    # the plain truncation is discontinuous (what makes the shift necessary)
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, rc - 1e-6]], pbc=False)
    assert ref.energy_forces_virial(atoms, s6, rc, shifted=False)[0] == pytest.approx(phi_rc, rel=1e-5)


def test_reference_rejects_elements_beyond_z86():
    atoms = Atoms('CUAc', positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0]])
    with pytest.raises(ValueError, match=r'U.*Ac|Ac.*U'):
        ref.energy_forces_virial(atoms, 0.75, 12.0)


# ---------------------------------------------------------------------------------------------
# 2. GPU: D2 contribution vs the reference
# ---------------------------------------------------------------------------------------------
S6_GPU = 75.0      # large: D2 signal >> float noise of the NEP difference, so no absolute floors needed
GPU_CASES = {   # name: (s6, rc, description)
    'dimer': (S6_GPU, 12.0, 'non-periodic dimer'),
    'benzene_dimer': (S6_GPU, 12.0, 'non-periodic benzene dimer'),
    'graphite': (S6_GPU, 12.0, 'hexagonal cell'),
    'water27': (S6_GPU, 12.0, 'cubic cell, rc > box'),
    'water27_tri': (S6_GPU, 12.0, 'triclinic cell'),
    'water_large': (S6_GPU, 12.0, '2187 atoms'),
}
REL_CAP_EV = 2e-6
REL_CAP_F = 1e-5


def _run(workdir, atoms, dftd2_line):
    workdir.mkdir(parents=True, exist_ok=True)
    atoms.write(workdir / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP_MODEL}']
    if dftd2_line:
        lines.append(dftd2_line)
    lines += ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1500:]
    header = (workdir / 'dump.xyz').read_text().splitlines()[1]
    energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float).reshape(3, 3)
    forces = read(workdir / 'dump.xyz', format='extxyz').get_forces()
    return energy, forces, virial


def _check_against_reference(tmp_path, atoms, s6, rc):
    """(run with dftd2) - (run without) equals the reference; relative caps only, no floors."""
    E_ref, F_ref, W_ref = ref.energy_forces_virial(atoms, s6, rc)
    e0, f0, v0 = _run(tmp_path / 'noD2', atoms, None)
    e1, f1, v1 = _run(tmp_path / 'D2', atoms, f'dftd2 {s6} {rc}')
    tol_e = REL_CAP_EV * abs(E_ref)
    tol_f = REL_CAP_F * np.abs(F_ref).max()
    tol_v = REL_CAP_EV * np.abs(W_ref).max()
    assert abs((e1 - e0) - E_ref) <= tol_e, f'energy error {abs((e1 - e0) - E_ref):.3e} eV (tol {tol_e:.1e})'
    err_f = np.abs((f1 - f0) - F_ref).max()
    assert err_f <= tol_f, 'force error %.3e eV/A (tol %.1e)' % (err_f, tol_f)
    err_v = np.abs((v1 - v0) - W_ref).max()
    assert err_v <= tol_v, 'virial error %.3e eV (tol %.1e)' % (err_v, tol_v)


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', list(GPU_CASES))
def test_dftd2_matches_reference(name, tmp_path):
    s6, rc, _ = GPU_CASES[name]
    _check_against_reference(tmp_path, structure(name), s6, rc)


def _tri_many_cells():      # triclinic, 3 cells per direction (n >= 2 for rc = 12)
    return structure('water27_tri').repeat((2, 2, 2))


def _open_cluster():        # non-periodic, extent (~19 A) larger than rc/2 -> several cells (n > 1)
    atoms = structure('water27').repeat((2, 2, 2))
    atoms.pbc = False
    return atoms


def _slab():                # mixed boundary conditions: periodic in x, y; open in z
    atoms = structure('water27').repeat((2, 2, 2))
    atoms.pbc = [True, True, False]
    return atoms


def _permuted():
    atoms = structure('water27_tri')
    perm = np.random.default_rng(11).permutation(len(atoms))
    return atoms[perm]


def _huge_coordinates():    # integer multiples of the lattice vectors added to all positions
    atoms = structure('water27_tri')
    cell = atoms.cell.array
    atoms.positions += 100 * cell[0] - 75 * cell[1] + 40 * cell[2]
    return atoms


GPU_DERIVED_CASES = {   # name: (builder, rc, description)
    'tri_many_cells': (_tri_many_cells, 12.0, 'triclinic cell with n >= 2 cells in each direction'),
    'open_cluster': (_open_cluster, 12.0, 'non-periodic cluster larger than rc/2'),
    'slab_ttf': (_slab, 12.0, 'mixed PBC (T T F)'),
    'permuted_order': (_permuted, 12.0, 'atom order permuted'),
    'huge_coordinates': (_huge_coordinates, 12.0, 'positions shifted by large integer lattice multiples'),
}


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', list(GPU_DERIVED_CASES))
def test_dftd2_matches_reference_geometry_variants(name, tmp_path):
    builder, rc, _ = GPU_DERIVED_CASES[name]
    _check_against_reference(tmp_path, builder(), S6_GPU, rc)


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_goes_to_zero_at_the_cutoff(tmp_path):
    """The pair distance is evaluated in single precision, and so is the cutoff (rc = 15.3 is not
    exactly representable).  At distances 4 float ulps (about 3.8e-6 A, resolved by the float
    arithmetic) below the cutoff, energy and force are ~0 (shifted force, continuous in E and F)
    and 4 ulps above it they are exactly 0."""
    rc, s6 = 15.3, 0.75
    rc_f = float(np.float32(rc))
    ulp = float(np.spacing(np.float32(rc)))
    for k, (dist, bound) in enumerate(((rc_f - 4 * ulp, 1e-9), (rc_f + 4 * ulp, 0.0))):
        atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, dist]], cell=[40, 40, 40], pbc=False)
        e0, f0, _ = _run(tmp_path / f'a{k}', atoms, None)
        e1, f1, _ = _run(tmp_path / f'b{k}', atoms, f'dftd2 {s6} {rc}')
        # for scale: the unshifted pair energy at rc is ~1e-5 eV
        assert abs(e1 - e0) <= max(bound, 1e-12), f'E = {e1 - e0:.3e} at r = {dist}'
        assert np.abs(f1 - f0).max() <= max(bound, 1e-12), f'|F| = {np.abs(f1 - f0).max():.3e} at r = {dist}'


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_repeated_runs_agree(tmp_path):
    """Run-to-run reproducibility (limited by the atomic placement into cells): H atoms on a sparse
    jittered lattice (spacing 9 A, beyond the NEP cutoff, so NEP contributes nothing and the whole
    result is the D2 part), 1000 atoms, rc = 12 A, 5 runs; relative difference <= 1e-12."""
    rng = np.random.default_rng(5)
    n = 10
    grid = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float) * 9.0
    atoms = Atoms('H%d' % len(grid), positions=grid + rng.uniform(-0.8, 0.8, grid.shape),
                  cell=[9.0 * n] * 3, pbc=True)
    results = [_run(tmp_path / f'run{i}', atoms, 'dftd2 0.75 12') for i in range(5)]
    e0, f0, v0 = results[0]
    assert abs(e0) > 0
    for e, f, v in results[1:]:
        assert abs(e - e0) <= 1e-12 * abs(e0)
        assert np.abs(f - f0).max() <= 1e-12 * np.abs(f0).max()
        assert np.abs(v - v0).max() <= 1e-12 * np.abs(v0).max()


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_dense_water_is_bitwise_reproducible(tmp_path):
    """The order of the atoms inside a cell is deterministic (increasing atom index), so repeated
    runs give bit-identical energy, forces and virial.  Dense water (2187 atoms, rc = 30 A, so that the
    whole system is one or a few cells), 5 runs with dftd2, NEP-only run for the D2 part."""
    atoms = structure('water_large')
    runs = [_run(tmp_path / f'on{i}', atoms, 'dftd2 0.75 30') for i in range(5)]
    e0, f0, w0 = runs[0]
    for e, f, w in runs[1:]:
        assert e == e0 and np.array_equal(f, f0) and np.array_equal(w, w0)
    e_off, f_off, _ = _run(tmp_path / 'off', atoms, None)
    assert abs(e0 - e_off) > 1.0 and np.abs(f0 - f_off).max() > 1e-3     # D2 is really active


# gpumd itself requires at least 2 atoms ("Number of atoms should >= 2"), so N = 1 cannot be run.
def _two_atoms_periodic():       # smallest periodic system: self-images and the pair, rc = 3 cells
    return Atoms('CO', positions=[[1.0, 1.0, 1.0], [2.7, 1.4, 0.6]], cell=[4.0, 4.0, 4.0], pbc=True)


def _two_atoms_open_beyond_cutoff():     # non-periodic, separated by more than rc: D2 is exactly zero
    return Atoms('CO', positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 21.0]], cell=[30.0, 30.0, 30.0], pbc=False)


def _tiny_box():             # box (2.4 A) much smaller than rc = 12 A: neighbor range k = 5 cells
    return Atoms('C2O2', positions=[[0.1, 0.2, 0.3], [1.3, 1.4, 0.2], [0.2, 1.2, 1.3], [1.4, 0.1, 1.2]],
                 cell=[2.4, 2.4, 2.4], pbc=True)


def _sparse_large_box():     # 1000 atoms in a 300 A box: 50^3 cells requested, halved to the cell limit
    rng = np.random.default_rng(3)
    return Atoms('C1000', positions=rng.uniform(0.0, 300.0, (1000, 3)), cell=[300.0] * 3, pbc=True)


def _triclinic_mixed_pbc():  # triclinic, periodic in x and z, open in y
    atoms = structure('water27_tri').copy()
    atoms.pbc = [True, False, True]
    return atoms


def _far_separated_open_direction():     # two groups 150 A apart along the open z direction
    atoms = structure('water27').repeat((1, 1, 2))
    atoms.pbc = [True, True, False]
    pos = atoms.positions
    pos[len(atoms) // 2:, 2] += 150.0
    atoms.set_positions(pos)
    return atoms


# Coincident atoms (r = 0) are not tested: the NEP model itself returns NaN for them, so the D2
# contribution cannot be isolated; D2 is documented as unsupported for them.
DEGENERATE_CASES = {    # name: (builder, rc)
    'two_atoms_periodic': (_two_atoms_periodic, 12.0),
    'two_atoms_open_beyond_cutoff': (_two_atoms_open_beyond_cutoff, 12.0),
    'tiny_box': (_tiny_box, 12.0),
    'sparse_large_box': (_sparse_large_box, 12.0),
    'triclinic_mixed_pbc': (_triclinic_mixed_pbc, 12.0),
    'far_separated_open_direction': (_far_separated_open_direction, 12.0),
}


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('name', list(DEGENERATE_CASES))
def test_dftd2_matches_reference_degenerate_cases(name, tmp_path):
    builder, rc = DEGENERATE_CASES[name]
    _check_against_reference(tmp_path, builder(), S6_GPU, rc)


# ---------------------------------------------------------------------------------------------
# 3. GPU input errors
# ---------------------------------------------------------------------------------------------
def _expect_input_error(tmp_path, dftd_lines, patterns, atoms=None):
    atoms = atoms if atoms is not None else structure('dimer')
    atoms.write(tmp_path / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP_MODEL}'] + dftd_lines + ['ensemble nve', 'time_step 0', 'run 1']
    (tmp_path / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=tmp_path, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode != 0, 'run.in was accepted'
    assert 'invalid keyword' not in text.lower(), 'dftd2 is not implemented (generic unknown-keyword error)'
    for pat in patterns:
        assert re.search(pat, text, re.I), f'error message lacks /{pat}/:\n{text[-600:]}'


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('lines', [['dftd2 0.75 12', 'dftd3 pbe 12 6'], ['dftd3 pbe 12 6', 'dftd2 0.75 12']])
def test_dftd2_and_dftd3_are_mutually_exclusive(tmp_path, lines):
    _expect_input_error(tmp_path, lines, [r'dftd2', r'dftd3'])


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_can_only_appear_once(tmp_path):
    _expect_input_error(tmp_path, ['dftd2 0.75 12', 'dftd2 0.75 12'], [r'dftd2.*once'])


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('s6', ['0', '-0.75'])
def test_dftd2_s6_must_be_positive(tmp_path, s6):
    _expect_input_error(tmp_path, [f'dftd2 {s6} 12'], [r's6'])


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('rc', ['0', '-12'])
def test_dftd2_cutoff_must_be_positive(tmp_path, rc):
    _expect_input_error(tmp_path, [f'dftd2 0.75 {rc}'], [r'cutoff'])


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_rejects_elements_beyond_z86_naming_them(tmp_path):
    atoms = Atoms('CUAcH', positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 3]], cell=[20, 20, 20])
    _expect_input_error(tmp_path, ['dftd2 0.75 12'], [r'\bU\b', r'\bAc\b'], atoms=atoms)


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
@pytest.mark.parametrize('symbol', ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'])
def test_dftd2_names_each_element_beyond_z86(tmp_path, symbol):
    atoms = Atoms(f'H{symbol}', positions=[[0, 0, 0], [3, 0, 0]], cell=[20, 20, 20])
    _expect_input_error(tmp_path, ['dftd2 0.75 12'], [rf'\b{symbol}\b'], atoms=atoms)


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_runs_with_nep89_when_structure_has_no_element_beyond_z86(tmp_path):
    """NEP89 itself contains Ac..Pu, but only the elements present in the structure matter."""
    first_line = NEP_MODEL.read_text().splitlines()[0].split()
    assert {'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'} <= set(first_line)
    e1, f1, _ = _run(tmp_path / 'D2', structure('dimer'), 'dftd2 0.75 12')
    assert np.isfinite(e1) and np.all(np.isfinite(f1))


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_on_unsupported_potential_is_an_error(tmp_path):
    tersoff = REPO_ROOT / 'potentials' / 'tersoff' / 'Si_Fan_2019.txt'
    if not tersoff.exists():
        pytest.skip('tersoff potential file not found')
    atoms = Atoms('Si2', positions=[[0, 0, 0], [2.4, 0, 0]], cell=[20, 20, 20], pbc=True)
    atoms.write(tmp_path / 'model.xyz', format='extxyz')
    (tmp_path / 'run.in').write_text(f'potential {tersoff}\ndftd2 0.75 12\nensemble nve\n'
                                     'time_step 0\nrun 1\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=tmp_path, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode != 0
    assert 'invalid keyword' not in text.lower(), 'dftd2 is not implemented (generic unknown-keyword error)'
    assert re.search(r'dftd2.*(NEP|support)', text, re.I), text[-600:]


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_without_potential_gives_a_specific_message(tmp_path):
    structure('dimer').write(tmp_path / 'model.xyz', format='extxyz')
    (tmp_path / 'run.in').write_text('dftd2 0.75 12\nensemble nve\ntime_step 0\nrun 1\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=tmp_path, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode != 0
    assert re.search(r"no 'potential' keyword", text), text[-600:]
    assert 'only supported for NEP' not in text and 'not supported' not in text


@pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
def test_dftd2_large_cutoff_prints_a_warning_and_still_runs(tmp_path):
    """A cutoff above 200 A only triggers a printed warning (open dimer, so the run is cheap)."""
    atoms = structure('dimer')
    atoms.pbc = False
    workdir = tmp_path / 'big'
    workdir.mkdir()
    atoms.write(workdir / 'model.xyz', format='extxyz')
    (workdir / 'run.in').write_text(f'potential {NEP_MODEL}\ndftd2 0.75 250\nensemble nve\ntime_step 0\nrun 1\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
    assert re.search(r'Warning: the dftd2 cutoff \(250 A\) is larger than 200 A', result.stdout)
    small = _run(tmp_path / 'small', atoms, 'dftd2 0.75 12')
    assert np.isfinite(small[0])
