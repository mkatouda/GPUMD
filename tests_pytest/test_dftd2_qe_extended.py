"""Additional tests of `dftd2 ... table qe` / `trunc plain` after the DR-004 implementation review.

1. Table check (CPU only): all 3741 C6 (first reference point) and r0ab entries of
   src/force/dftd2qe_table.cuh against the VER-0025 CSVs (fixtures/dftd2_qe/) and, when the QE 7.2
   sources are readable, against an independent re-parse of the Fortran data statements (flat scan
   of the numbers, D-exponent handling; no block indices, no limit() emulation: different from
   tools/Miscellaneous/dftd2/generate_dftd2qe_table.py and from the VER-0025 extraction script).
   QE sources: env DFTD2_QE72_SRC (directory with pars.f90 and core.f90) or the git checkout
   /home/katouda/build/q-e with tag qe-7.2.
2. Mixed-element end to end (H C N O Si Cl Rn, pairs Z = 1 / 86 present), non-periodic and periodic,
   `table qe`, `trunc plain` and `sf`, against dftd2_qe_oracle.py (full table from the CSVs).
3. Boundary at the cutoff: pairs at exactly rc are excluded (r < rc).
4. Measured margins against the DR-002 AC-1 caps (energy 2e-6, virial 2e-6, force 1e-5) for the
   heaviest C6 (Rn-Rn) and the smallest-R0 (H-H) pairs: the values are printed (pytest -s) and
   recorded (record_property) so that an erosion of the margin is visible.
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import dftd2_qe_oracle as oracle
from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd2_qe'
TABLE_CUH = REPO_ROOT / 'src' / 'force' / 'dftd2qe_table.cuh'
QE_TAG = 'qe-7.2'
QE_DEFAULT = Path('/home/katouda/build/q-e')
fast = pytest.mark.fast
REL_CAP_E, REL_CAP_W, REL_CAP_F = 2e-6, 2e-6, 1e-5
S6 = 0.75


# ---------------------------------------------------------------------------------------------
# 1. tables (CPU only)
# ---------------------------------------------------------------------------------------------
def read_cuh_arrays():
    text = TABLE_CUH.read_text()
    out = {}
    for name in ('DFTD2QE_C6_EH_BOHR6', 'DFTD2QE_R0AB_A'):
        body = re.search(name + r'\[3741\]\s*=\s*\{(.*?)\};', text, re.S).group(1)
        out[name] = np.array([float(t) for t in body.replace('\n', ' ').split(',') if t.strip()])
        assert out[name].size == 3741
    return out['DFTD2QE_C6_EH_BOHR6'], out['DFTD2QE_R0AB_A']


def packed(matrix):
    """lower triangle in the packing index zl * (zl + 1) / 2 + zs (0-based Z, zs <= zl)"""
    return np.array([matrix[zl, zs] for zl in range(86) for zs in range(zl + 1)])


@fast
def test_table_header_equals_ver0025_csv():
    c6, r0 = read_cuh_arrays()
    c6_csv, r0_csv = oracle.load_tables()
    assert np.array_equal(c6_csv, c6_csv.T) and np.array_equal(r0_csv, r0_csv.T)
    # the CSV holds up to 9 significant digits, the data have <= 8: agreement to the last digit
    np.testing.assert_allclose(c6, packed(c6_csv), rtol=1e-9, atol=0)
    np.testing.assert_allclose(r0, packed(r0_csv), rtol=1e-9, atol=0)


def fortran_numbers(text):
    """all numbers of the `(/ ... /)` data statements in file order (comments removed; `D`/`d`
    exponents; `_wp` kinds); returns the list of (statement name, [values])."""
    text = '\n'.join(ln.split('!')[0] for ln in text.splitlines()).replace('&', ' ')
    result = []
    for m in re.finditer(r'(\w+)\s*\(\s*\d+\s*:\s*\d+\s*\)\s*=\s*\(/(.*?)/\)', text, re.S):
        vals = [float(re.sub(r'[dD]', 'e', t.strip().replace('_wp', ''))) for t in m.group(2).split(',') if t.strip()]
        result.append((m.group(1), vals))
    return result


def qe_sources():
    env = os.environ.get('DFTD2_QE72_SRC')
    if env:
        d = Path(env)
        if (d / 'pars.f90').exists() and (d / 'core.f90').exists():
            return (d / 'pars.f90').read_text(), (d / 'core.f90').read_text(), str(d)
        return None
    if (QE_DEFAULT / '.git').exists():
        try:
            get = lambda f: subprocess.run(['git', '-C', str(QE_DEFAULT), 'show', f'{QE_TAG}:dft-d3/{f}'],
                                           capture_output=True, text=True, check=True).stdout
            return get('pars.f90'), get('core.f90'), f'{QE_DEFAULT}@{QE_TAG}'
        except (subprocess.CalledProcessError, OSError):
            return None
    return None


@fast
def test_table_header_equals_independent_reparse_of_qe72_sources():
    src = qe_sources()
    if src is None:
        pytest.skip('QE 7.2 sources not available (set DFTD2_QE72_SRC to a directory with pars.f90 and '
                    f'core.f90, or provide {QE_DEFAULT} with tag {QE_TAG})')
    pars_text, core_text, where = src
    # pars: flat list of 5-number records (C6, Zi + 100 (iref_i - 1), Zj + 100 (iref_j - 1), CN_i, CN_j)
    pars = [v for name, vals in fortran_numbers(pars_text) if name == 'pars' for v in vals]
    assert len(pars) == 161925, where
    rec = np.array(pars).reshape(-1, 5)
    c6 = np.full((86, 86), np.nan)
    for c, a, b, _, _ in rec:
        a, b = int(round(a)), int(round(b))
        if a <= 86 and b <= 86:                       # both atoms: first reference point (Z < 100)
            assert np.isnan(c6[a - 1, b - 1]) and np.isnan(c6[b - 1, a - 1]) or a == b
            c6[a - 1, b - 1] = c6[b - 1, a - 1] = c
    assert not np.isnan(c6).any()
    # setr0ab: flat 4465 values, row-major lower triangle (i = 1..94, j = 1..i)
    r0flat = [v for name, vals in fortran_numbers(core_text) if name == 'r0ab' for v in vals]
    assert len(r0flat) == 4465, where
    r0 = np.zeros((86, 86))
    k = 0
    for i in range(94):
        for j in range(i + 1):
            if i < 86:
                r0[i, j] = r0[j, i] = r0flat[k]
            k += 1
    hc6, hr0 = read_cuh_arrays()
    # exact: the header repr()s the doubles parsed from the same literals
    assert np.array_equal(hc6, packed(c6)), where
    assert np.array_equal(hr0, packed(r0)), where


# ---------------------------------------------------------------------------------------------
# GPU helpers (zero NEP with the elements H C N O Si Cl Rn: the D2 term alone)
# ---------------------------------------------------------------------------------------------
SPECIES = ['H', 'C', 'N', 'O', 'Si', 'Cl', 'Rn']


def write_zero_nep(path, species=SPECIES):
    t = len(species)
    npar = 16 * t + 1 + 8 * t * t          # ann (dim 6, 2 neurons) + radial and angular coefficients
    lines = [f'nep {t} ' + ' '.join(species), 'format_version 1', 'model nep', 'cutoff 3 3 20 20',
             'n_max 1 1', 'basis_size 1 1', 'l_max 2 0 0', 'ann 2 0', 'energy_reference ' + ' '.join(['0'] * t),
             f'@ parameters {npar}'] + ['0'] * npar + ['@ q_scaler 6'] + ['1'] * 6
    Path(path).write_text('\n'.join(lines) + '\n')


def run_d2(workdir, atoms, d2_line):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    write_zero_nep(workdir / 'zero.txt')
    atoms.write(workdir / 'model.xyz', format='extxyz')
    (workdir / 'run.in').write_text('\n'.join(
        ['potential zero.txt', d2_line, 'ensemble nve', 'time_step 0',
         'dump_xyz 1 dump.xyz force precision double', 'run 1']) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout + result.stderr)[-800:]
    lines = (workdir / 'dump.xyz').read_text().splitlines()
    n = int(lines[0])
    energy = float(re.search(r'energy=([-0-9.eE+]+)', lines[1]).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', lines[1]).group(1).split(), float).reshape(3, 3)
    forces = np.array([ln.split()[4:7] for ln in lines[2:2 + n]], float)
    return energy, forces, virial


def atomic_numbers(symbols):
    return [Atoms(s).get_atomic_numbers()[0] for s in symbols]


def random_cluster(symbols, box, seed, dmin):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < len(symbols):
        p = rng.uniform(0.05, 0.95, 3) * box
        if all(np.linalg.norm(p - q) > dmin for q in pos):
            pos.append(p)
    return np.array(pos)


def errors(e, f, w, ref):
    e0, f0, w0 = ref
    return (abs(e - e0) / abs(e0), np.abs(w - w0).max() / np.abs(w0).max(), np.abs(f - f0).max() / np.abs(f0).max())


def check_caps(err):
    ee, ew, ef = err
    assert ee <= REL_CAP_E, f'energy rel. error {ee:.2e} > {REL_CAP_E:.0e}'
    assert ew <= REL_CAP_W, f'virial rel. error {ew:.2e} > {REL_CAP_W:.0e}'
    assert ef <= REL_CAP_F, f'force rel. error {ef:.2e} > {REL_CAP_F:.0e}'


# ---------------------------------------------------------------------------------------------
# 2. mixed elements end to end
# ---------------------------------------------------------------------------------------------
MIXED = ['H', 'C', 'N', 'O', 'Si', 'Cl', 'Rn', 'H', 'Rn', 'O', 'C', 'Cl', 'Si', 'H', 'N', 'Rn']


@pytest.mark.parametrize('trunc', ['plain', 'sf'])
@pytest.mark.parametrize('periodic', [False, True])
def test_mixed_elements_match_the_full_table_oracle(periodic, trunc, tmp_path):
    box = 11.0 if periodic else 14.0
    pos = random_cluster(MIXED, box, seed=7, dmin=1.8)
    rc = 12.0 if periodic else 8.0
    cell = np.diag([box, box, box]) if not periodic else np.array([[box, 0, 0], [1.3, box, 0], [0.7, -1.1, box]])
    if periodic:                                   # triclinic periodic cell, atoms inside the cell
        pos = pos @ np.linalg.inv(np.diag([box] * 3)) @ cell * 0.999
    atoms = Atoms(MIXED, positions=pos, cell=cell if periodic else np.diag([40.0] * 3), pbc=periodic)
    if not periodic:
        atoms.positions += 13.0
    ref = oracle.d2(atomic_numbers(MIXED), atoms.positions, atoms.cell.array, periodic, trunc, rc, S6)
    assert 1 in atomic_numbers(MIXED) and 86 in atomic_numbers(MIXED)
    e, f, w = run_d2(tmp_path, atoms, f'dftd2 {S6} {rc:g} table qe trunc {trunc}')
    check_caps(errors(e, f, w, ref))


# ---------------------------------------------------------------------------------------------
# 3. boundary at the cutoff
# ---------------------------------------------------------------------------------------------
def single_atom_cell(symbol, d):
    """simple cubic lattice (spacing d) in a 2 x 1 x 1 cell with two atoms (GPUMD needs >= 2 atoms)"""
    return Atoms([symbol] * 2, positions=[[0.5 * d] * 3, [1.5 * d, 0.5 * d, 0.5 * d]],
                 cell=np.diag([2 * d, d, d]), pbc=True)


@pytest.mark.parametrize('trunc', ['plain', 'sf'])
def test_pairs_at_exactly_rc_are_excluded(trunc, tmp_path):
    """Simple cubic lattice: the first shell (6 neighbors per atom) is at d = 4 (exact in float).
    r < rc: rc = d excludes it (E = 0 exactly), rc = d + 1e-4 includes it."""
    d = 4.0
    atoms = single_atom_cell('Rn', d)
    e, f, w = run_d2(tmp_path / 'at', atoms, f'dftd2 {S6} {d:g} table qe trunc {trunc}')
    assert e == 0.0 and not f.any() and not w.any(), f'shell at exactly rc was included: E = {e}'
    rc = d + 1e-4
    ref = oracle.d2([86] * 2, atoms.positions, atoms.cell.array, True, trunc, rc, S6)
    e, f, w = run_d2(tmp_path / 'above', atoms, f'dftd2 {S6} {rc:.4f} table qe trunc {trunc}')
    if trunc == 'plain':
        assert ref[0] != 0.0 and e != 0.0
        assert abs(e - ref[0]) <= REL_CAP_E * abs(ref[0])
        assert np.abs(w - ref[2]).max() <= REL_CAP_W * np.abs(ref[2]).max()


@pytest.mark.parametrize('d', [4.3, 5.1234])
def test_shell_just_inside_and_outside_rc_with_gap(d, tmp_path):
    """Non-representable d: a gap of 1e-3 A (>> float rounding of the pair vector) decides."""
    atoms = single_atom_cell('Rn', d)
    e_out, *_ = run_d2(tmp_path / 'out', atoms, f'dftd2 {S6} {d - 1e-3:.4f} table qe trunc plain')
    assert e_out == 0.0
    e_in, f, w = run_d2(tmp_path / 'in', atoms, f'dftd2 {S6} {d + 1e-3:.4f} table qe trunc plain')
    ref = oracle.d2([86] * 2, atoms.positions, atoms.cell.array, True, 'plain', d + 1e-3, S6)
    assert ref[0] != 0.0 and abs(e_in - ref[0]) <= REL_CAP_E * abs(ref[0])


# ---------------------------------------------------------------------------------------------
# 4. measured margins at the heaviest C6 (Rn-Rn) and the smallest R0 (H-H) pairs
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('trunc', ['plain', 'sf'])
@pytest.mark.parametrize('symbol, dmin', [('Rn', 3.4), ('H', 1.6)])
def test_margin_to_the_caps_heavy_and_light_pairs(symbol, dmin, trunc, tmp_path, record_property):
    n, box, rc = 24, 14.0, 12.0
    pos = random_cluster([symbol] * n, box, seed=3, dmin=dmin)
    atoms = Atoms([symbol] * n, positions=pos, cell=np.diag([box] * 3), pbc=True)
    ref = oracle.d2([atoms.get_atomic_numbers()[0]] * n, pos, atoms.cell.array, True, trunc, rc, S6)
    e, f, w = run_d2(tmp_path, atoms, f'dftd2 {S6} {rc:g} table qe trunc {trunc}')
    err = errors(e, f, w, ref)
    for key, val, cap in zip(('energy', 'virial', 'force'), err, (REL_CAP_E, REL_CAP_W, REL_CAP_F)):
        record_property(f'{symbol}_{trunc}_{key}_rel_error', f'{val:.3e}')
        record_property(f'{symbol}_{trunc}_{key}_cap_fraction', f'{val / cap:.3f}')
    print(f'\nMARGIN {symbol}-{symbol} {trunc}: energy {err[0]:.2e} ({err[0] / REL_CAP_E:.1%} of cap), '
          f'virial {err[1]:.2e} ({err[1] / REL_CAP_W:.1%}), force {err[2]:.2e} ({err[2] / REL_CAP_F:.1%})')
    check_caps(err)
