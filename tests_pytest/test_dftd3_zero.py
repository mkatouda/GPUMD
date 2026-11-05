"""DFT-D3 zero damping, D3(0) (`dftd3 <functional> <rc> <rc_cn> [zero|bj]`): tests written BEFORE the
implementation (tests-first). They define the contract of the optional damping token:

1. Reference (AC-1): energy, forces and virial against stored simple-dftd3 values (zero damping,
   two-body only, same cutoffs; fixtures/dftd3_zero/reference.json, provenance in
   fixtures/dftd3_zero/generate_reference.py and README.md): PBE on all structures of the BJ test
   plus a heavy-element cell, revPBE/BLYP/B3LYP/TPSS on three. The D3 contribution is isolated as
   (run with `dftd3`) - (same run without), NEP89 underneath, as in test_dftd3_reference.py.
2. Parser and error behaviour (ADR-0003 items 1-3, DR-003 D1, D4, D5, AC-4, AC-4b).
3. BJ compatibility (AC-3): `dftd3 pbe 12 6` and `... bj` give output bit-identical to each other and
   to the frozen output of the pre-change binary (fixtures/dftd3_zero/bj_frozen.json). These use an
   all-zero NEP (output = D3 alone, bit-reproducible); NEP89 itself is not, run to run.
4. Force and virial consistency with finite differences of the energy (AC-2).
5. Robustness (AC-5): small-box and large-box path agreement, and no NaN/inf for r << R0.

Tests marked "BJ compat" must pass on the pre-change binary as well; all others need the zero
damping feature.

Tolerances (reference tests): min(cap, max(2 x BASELINE, floor)), caps of ADR-0002 (relative 2e-6 of
the D3 energy and of the largest virial component, 1e-5 eV/A for forces). BASELINE for zero damping
is NOT yet measured: it is provisionally taken from the BJ baselines of the same structure
(test_dftd3_reference.py, same float summation), and is to be replaced by the measured values of the
zero path in DES-0003 Phase 4. For structures without a BJ baseline the floors apply.
"""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURES = REPO_ROOT / 'tests_pytest' / 'fixtures'
ZERO_DIR = FIXTURES / 'dftd3_zero'
BJ_DIR = FIXTURES / 'dftd3'                       # structures shared with test_dftd3_reference.py
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
EXE = Path(os.environ.get('DFTD3_ZERO_TEST_GPUMD', GPUMD_EXECUTABLE))
REFERENCE = json.load(open(ZERO_DIR / 'reference.json'))
CASES = [name for name in REFERENCE if not name.startswith('_')]
FROZEN = json.load(open(ZERO_DIR / 'bj_frozen.json'))

def fast_model(func):
    return pytest.mark.fast(pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')(func))


def slow_model(func):
    return pytest.mark.slow(pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')(func))

REL_CAP = 2e-6        # energy and virial: fraction of |E_D3| / max|Virial_ref| (ADR-0002)
FORCE_CAP = 1e-5      # eV/A
FLOORS = dict(energy=2e-6, force=1e-6, virial=2e-5)
# PROVISIONAL baselines (|dE| eV, max|dF| eV/A, max|dVirial| eV): the BJ values of test_dftd3_reference.py
BJ_BASELINE = {
    'benzene_dimer': (3.72e-07, 2.82e-08, 3.40e-07),
    'graphite':      (3.94e-06, 4.57e-07, 5.27e-06),
    'water27':       (1.65e-06, 1.34e-08, 1.80e-06),
    'water27_rc25':  (1.73e-06, 1.28e-07, 1.21e-06),
    'water27_tri':   (2.16e-06, 2.68e-08, 2.22e-06),
    'water_large':   (4.82e-05, 1.94e-07, 4.61e-05),
    'cu_large':      (6.15e-04, 1.33e-07, 5.66e-04),
}


ZERO_NEP = ZERO_DIR / 'zero_nep_H_C_O_Cu.txt'     # all-zero NEP: output = D3 alone, bit-reproducible


def run_gpumd(workdir, atoms, d3_line, extra_lines=(), check=True, model=NEP_MODEL):
    """Static run (time_step 0, one step). Returns (CompletedProcess, (energy, forces, virial) or None)."""
    workdir.mkdir(parents=True, exist_ok=True)
    atoms.write(workdir / 'model.xyz', format='extxyz')
    lines = [f'potential {model}'] + list(extra_lines) + ([d3_line] if d3_line else []) + \
        ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(EXE)], cwd=workdir, capture_output=True, text=True)
    if check:
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    if result.returncode != 0:
        return result, None
    header = (workdir / 'dump.xyz').read_text().splitlines()[1]
    energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float).reshape(3, 3)
    return result, (energy, read(workdir / 'dump.xyz', format='extxyz').get_forces(), virial)


def d3_part(workdir, atoms, d3_line):
    """(E, F, virial) of the D3 contribution only: run with minus run without."""
    _, (e1, f1, v1) = run_gpumd(workdir / 'D3', atoms, d3_line)
    _, (e0, f0, v0) = run_gpumd(workdir / 'noD3', atoms, None)
    return e1 - e0, f1 - f0, v1 - v0


def combined_output(result):
    """Last lines of stdout + stderr: the error message is printed last. (The whole output is not
    searched: GPUMD echoes the NEP89 element list, which contains U and Pu.)"""
    return '\n'.join((result.stdout + result.stderr).strip().splitlines()[-6:])


# ---------------------------------------------------------------------------------------------
# 1. Reference (AC-1)
# ---------------------------------------------------------------------------------------------
def structure_path(ref):
    name = ref['structure']
    return ZERO_DIR / name if (ZERO_DIR / name).exists() else BJ_DIR / name


def tolerance(key, quantity):
    ref = REFERENCE[key]
    cap = {'energy': REL_CAP * abs(ref['energy']),
           'force': FORCE_CAP,
           'virial': REL_CAP * float(np.abs(np.array(ref['virial'])).max())}[quantity]
    index = ['energy', 'force', 'virial'].index(quantity)
    baseline = BJ_BASELINE.get(ref['case'], (0.0, 0.0, 0.0))[index]
    return min(cap, max(2 * baseline, FLOORS[quantity]))


@fast_model
@pytest.mark.parametrize('key', CASES)
def test_dftd3_zero_matches_reference(key, tmp_path):
    ref = REFERENCE[key]
    atoms = read(structure_path(ref))
    line = f"dftd3 {ref['functional']} {ref['potential_cutoff']} {ref['cn_cutoff']} zero"
    e, f, v = d3_part(tmp_path, atoms, line)
    d_e = abs(e - ref['energy'])
    d_f = float(np.abs(f - np.array(ref['forces'])).max())
    d_v = float(np.abs(v - np.array(ref['virial'])).max())
    assert d_e <= tolerance(key, 'energy'), f'energy error {d_e:.3e} eV'
    assert d_f <= tolerance(key, 'force'), f'force error {d_f:.3e} eV/A'
    assert d_v <= tolerance(key, 'virial'), f'virial error {d_v:.3e} eV'


# ---------------------------------------------------------------------------------------------
# 2. Parser and error behaviour (ADR-0003, DR-003 D1/D4/D5, AC-4, AC-4b)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def water():
    return Atoms('OH2', positions=[[5, 5, 5], [5.96, 5, 5], [4.76, 5.93, 5]], cell=[12.0] * 3, pbc=True)


def run_input(tmp_path, atoms, d3_line, extra_lines=()):
    return run_gpumd(tmp_path, atoms, d3_line, extra_lines, check=False)[0]


@fast_model
@pytest.mark.parametrize('line', ['dftd3 pbe 12 6', 'dftd3 pbe 12 6 bj'])
def test_bj_syntax_accepted(line, tmp_path, water):
    """`bj` is the default damping; the explicit token is accepted (ADR-0003 item 1)."""
    result = run_input(tmp_path, water, line)
    assert result.returncode == 0, combined_output(result)[-1000:]


@fast_model
def test_zero_syntax_accepted(tmp_path, water):
    result = run_input(tmp_path, water, 'dftd3 pbe 12 6 zero')
    assert result.returncode == 0, combined_output(result)[-1000:]


@fast_model
@pytest.mark.parametrize('damping', ['zero', 'bj'])
def test_header_names_the_damping(damping, tmp_path, water):
    """ADR-0003 item 4: the printed header gains a damping line (existing lines keep their text)."""
    result = run_input(tmp_path, water, f'dftd3 pbe 12 6 {damping}')
    assert result.returncode == 0, combined_output(result)[-1000:]
    block = result.stdout.split('Add DFT-D3:')[1].split('\n\n')[0].lower()
    assert 'damping' in block and damping in block, block


@fast_model
@pytest.mark.parametrize('line', [
    'dftd3 pbe 12 6 foo',          # unknown selector
    'dftd3 pbe 12 6 ZERO',         # case sensitive
    'dftd3 pbe 12 6 BJ',
    'dftd3 pbe 12 6 zero extra',   # more than 5 tokens
    'dftd3 pbe 12 6 bj extra',
    'dftd3 pbe 12 6 zero bj',
])
def test_invalid_damping_token_or_extra_tokens_rejected_with_valid_values(line, tmp_path, water):
    result = run_input(tmp_path, water, line)
    out = combined_output(result)
    assert result.returncode != 0, out[-500:]
    assert 'zero' in out and 'bj' in out, f'message must list the valid values zero|bj: {out[-500:]}'


@fast_model
@pytest.mark.parametrize('line', ['dftd3', 'dftd3 pbe', 'dftd3 pbe 12'])
def test_too_few_tokens_rejected(line, tmp_path, water):
    """BJ compat: the existing message for fewer than 3 parameters stays (in meaning)."""
    result = run_input(tmp_path, water, line)
    out = combined_output(result)
    assert result.returncode != 0
    assert 'dftd3' in out and 'parameters' in out, out[-500:]


@fast_model
@pytest.mark.parametrize('first, second', [('dftd3 pbe 12 6', 'dftd3 pbe 12 6'),
                                           ('dftd3 pbe 12 6', 'dftd3 pbe 12 6 zero'),
                                           ('dftd3 pbe 12 6 zero', 'dftd3 pbe 12 6 zero')])
def test_dftd3_twice_rejected(first, second, tmp_path, water):
    result = run_input(tmp_path, water, first, [second])
    assert result.returncode != 0
    assert 'can only appear once' in combined_output(result)


@fast_model
@pytest.mark.parametrize('functional', ['wb97m', 'foo'])
def test_functional_unsupported_for_zero_names_damping_and_functional(functional, tmp_path, water):
    """AC-4: `wb97m` is tabulated for BJ but not for D3(0); `foo` for neither. No fallback to BJ."""
    result = run_input(tmp_path, water, f'dftd3 {functional} 12 6 zero')
    out = combined_output(result)
    assert result.returncode != 0, 'must not silently fall back to BJ damping'
    assert functional in out and 'zero' in out.lower() and 'not supported' in out, out[-500:]


@fast_model
def test_functional_unsupported_for_bj_default_names_damping_and_functional(tmp_path, water):
    """BJ compat: existing message 'The foo functional is not supported for DFT-D3 with BJ damping.'"""
    result = run_input(tmp_path, water, 'dftd3 foo 12 6')
    out = combined_output(result)
    assert result.returncode != 0
    assert 'foo' in out and 'BJ' in out and 'not supported' in out, out[-500:]


@fast_model
def test_functional_unsupported_for_explicit_bj_names_damping_and_functional(tmp_path, water):
    result = run_input(tmp_path, water, 'dftd3 foo 12 6 bj')
    out = combined_output(result)
    assert result.returncode != 0
    assert 'foo' in out and 'BJ' in out and 'not supported' in out, out[-500:]


@fast_model
def test_functional_supported_for_bj_only_still_works_with_bj(tmp_path, water):
    """BJ compat: wb97m is a BJ functional; adding zero damping must not remove it."""
    result = run_input(tmp_path, water, 'dftd3 wb97m 12 6')
    assert result.returncode == 0, combined_output(result)[-500:]


@fast_model
@pytest.mark.parametrize('symbols, named', [(['O', 'U'], ['U']), (['Pu', 'H'], ['Pu']),
                                           (['U', 'Pu', 'O'], ['U', 'Pu'])])
def test_zero_damping_rejects_atoms_with_z_above_86_naming_the_element(symbols, named, tmp_path):
    """DR-003 D5/AC-4b. NEP89 contains Ac-Pu (Z = 89-94). The element
    is named in the error. NOTE: the check must look at the atoms of the structure (the NEP89
    model itself lists U and Pu, and zero damping with NEP89 must work for light structures, see
    test_zero_damping_accepts_light_structure_with_nep89_model)."""
    pos = [[5.0, 5.0, 5.0], [7.5, 5.0, 5.0], [5.0, 8.0, 5.0]][:len(symbols)]
    atoms = Atoms(symbols, positions=pos, cell=[12.0] * 3, pbc=True)
    result = run_input(tmp_path, atoms, 'dftd3 pbe 12 6 zero')
    out = combined_output(result)
    assert result.returncode != 0, 'Z >= 87 with zero damping must be an error'
    for symbol in named:
        assert re.search(rf'\b{symbol}\b', out), f'element {symbol} not named: {out[-500:]}'


@fast_model
def test_zero_damping_accepts_light_structure_with_nep89_model(tmp_path, water):
    result = run_input(tmp_path, water, 'dftd3 pbe 12 6 zero')
    assert result.returncode == 0, combined_output(result)[-500:]


@fast_model
@pytest.mark.parametrize('line', ['dftd3 pbe 12 6', 'dftd3 pbe 12 6 bj'])
def test_bj_damping_still_accepts_z_above_86(line, tmp_path):
    """BJ compat (DR-003 D5 applies to zero damping only); the explicit `bj` needs the feature."""
    atoms = Atoms(['O', 'U'], positions=[[5, 5, 5], [7.5, 5, 5]], cell=[12.0] * 3, pbc=True)
    result = run_input(tmp_path, atoms, line)
    assert result.returncode == 0, combined_output(result)[-500:]


@fast_model
@pytest.mark.parametrize('d3_line', ['dftd3 pbe 12 6', 'dftd3 pbe 12 6 zero'])
@pytest.mark.parametrize('d2_first', [True, False])
def test_dftd3_and_dftd2_mutually_exclusive(d3_line, d2_first, tmp_path, water):
    lines = ['dftd2 1.0 12', d3_line] if d2_first else [d3_line, 'dftd2 1.0 12']
    water.write(tmp_path / 'model.xyz', format='extxyz')
    (tmp_path / 'run.in').write_text(f'potential {NEP_MODEL}\n' + '\n'.join(lines) +
                                     '\nensemble nve\ntime_step 0\nrun 1\n')
    result = subprocess.run([str(EXE)], cwd=tmp_path, capture_output=True, text=True)
    out = combined_output(result)
    assert result.returncode != 0
    assert 'cannot be used together' in out, out[-500:]


# ---------------------------------------------------------------------------------------------
# 3. BJ compatibility (AC-3): bit-identity with the pre-change binary
# ---------------------------------------------------------------------------------------------
def dump_lines(workdir):
    """Lines 2.. of dump.xyz (header with energy and virial, then species/positions/forces), without
    the `stress` field of the header (it varies from run to run with a zero potential)."""
    lines = (workdir / 'dump.xyz').read_text().splitlines()[1:]
    lines[0] = re.sub(r' stress="[^"]*"', '', lines[0])
    return lines


def d3_block(stdout):
    """Lines of the 'Add DFT-D3:' block of the standard output."""
    lines = stdout.splitlines()
    start = lines.index('    Add DFT-D3:')
    end = start
    while end < len(lines) and lines[end].strip():
        end += 1
    return lines[start:end]


FROZEN_CASES = [k for k in FROZEN if not k.startswith('_')]


def frozen_structure(case):
    return read(ZERO_DIR / case['structure'] if (ZERO_DIR / case['structure']).exists()
                else BJ_DIR / case['structure'])


@pytest.mark.fast
@pytest.mark.parametrize('name', FROZEN_CASES)
def test_bj_output_bit_identical_to_pre_change_binary(name, tmp_path):
    """`dftd3 pbe rc rcn` (no token) reproduces the stored output of the pre-change binary exactly
    (energy, forces, virial text of the dump with `precision double`; all-zero NEP, so the output is
    the D3 contribution). The header lines of the pre-change binary stay in the same order (extra
    lines allowed)."""
    case = FROZEN[name]
    cut = case['cutoffs']
    result, _ = run_gpumd(tmp_path, frozen_structure(case), f'dftd3 pbe {cut[0]} {cut[1]}',
                          model=ZERO_NEP)
    assert dump_lines(tmp_path) == case['D3']
    block = d3_block(result.stdout)
    it = iter(block)
    assert all(line in it for line in case['stdout_block']), block   # ordered subsequence


@pytest.mark.fast
@pytest.mark.parametrize('name', FROZEN_CASES)
def test_explicit_bj_token_bit_identical_to_default(name, tmp_path):
    case = FROZEN[name]
    atoms, cut = frozen_structure(case), case['cutoffs']
    for sub, line in (('default', f'dftd3 pbe {cut[0]} {cut[1]}'), ('bj', f'dftd3 pbe {cut[0]} {cut[1]} bj')):
        run_gpumd(tmp_path / sub, atoms, line, model=ZERO_NEP)
    assert dump_lines(tmp_path / 'default') == dump_lines(tmp_path / 'bj')


@fast_model
def test_zero_and_bj_differ(tmp_path):
    """The token must actually select a different damping."""
    atoms = read(BJ_DIR / 'water27.xyz')
    _, (e_bj, _, _) = run_gpumd(tmp_path / 'bj', atoms, 'dftd3 pbe 12 6 bj')
    _, (e_zero, _, _) = run_gpumd(tmp_path / 'zero', atoms, 'dftd3 pbe 12 6 zero')
    assert abs(e_zero - e_bj) > 0.1, (e_zero, e_bj)    # about 0.4 eV for these 81 atoms


# ---------------------------------------------------------------------------------------------
# 4. Finite differences (AC-2)
# ---------------------------------------------------------------------------------------------
# Force = -dE/dx and virial = -dE/d(strain) of the D3 part (run with minus run without), 5-point
# central differences of the double-precision energy of the dump header, rc = 25 A / CN 15 A.
# Tolerance relative to the largest D3 force / virial component; step 0.01 A, strain step 1e-3.
# Fixed from the BJ path (harness check on the pre-change binary, 2026-11-06, RTX A6000): measured
# relative errors force 6.3e-4 (cubic) / 1.6e-4 (open), virial 7.4e-4 (cubic) / 1.7e-3 (triclinic);
# the tolerances are about 3 x the worst values. The error is dominated by pairs crossing the hard
# cutoff and float energy noise, not by the force expressions. Do not loosen for the zero path.
H_FORCE, H_STRAIN = 0.01, 1e-3
FD_TOL_FORCE, FD_TOL_VIRIAL = 2e-3, 5e-3
# `bj` = the default (no token): the BJ parametrizations run on the pre-change binary (BJ compat)
TOKEN = {'zero': ' zero', 'bj': ''}
# Large cutoffs: the D3 energy has a hard cutoff, and pairs crossing it make FD derivatives noisy
# (a jump ~ C6/rc^6 per pair; with 12 A the virial FD error was 0.7% of the virial, measured on BJ).
DAMPING_LINES = {d: f'dftd3 pbe 25 15{TOKEN[d]}' for d in TOKEN}


class _Runner:
    def __init__(self, root, line):
        self.root, self.line, self.count = Path(root), line, 0

    def run(self, atoms, with_d3):
        self.count += 1
        _, (e, f, v) = run_gpumd(self.root / f'r{self.count}', atoms, self.line if with_d3 else None)
        return e, f, v

    def d3(self, atoms):
        e1, f1, v1 = self.run(atoms, True)
        e0, f0, v0 = self.run(atoms, False)
        return e1 - e0, f1 - f0, v1 - v0


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


def _derivatives(runner, jobs):
    def one(job):
        build, h = job
        em2, em1, ep1, ep2 = [runner.d3(build(s))[0] for s in (-2 * h, -h, h, 2 * h)]
        return (em2 - 8 * em1 + 8 * ep1 - ep2) / (12 * h)
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, jobs))


def _fd_systems():
    w = read(BJ_DIR / 'water27.xyz')
    open_cluster = w.copy()
    open_cluster.pbc = False
    return {'cubic': w, 'triclinic': read(BJ_DIR / 'water27_tri.xyz'), 'open': open_cluster}


@slow_model
@pytest.mark.parametrize('damping', ['zero', 'bj'])
@pytest.mark.parametrize('name', ['cubic', 'open'])
def test_force_is_minus_energy_gradient(damping, name, tmp_path):
    """The `bj` parametrization checks the harness on the pre-change binary (BJ compat)."""
    atoms = _fd_systems()[name]
    runner = _Runner(tmp_path, DAMPING_LINES[damping])
    _, F, _ = runner.d3(atoms)
    comps = [(k, a) for k in (0, 40) for a in range(3)]
    jobs = [((lambda s, k=k, a=a: _displaced(atoms, k, a, s)), H_FORCE) for k, a in comps]
    fd = [-d for d in _derivatives(runner, jobs)]
    err = max(abs(f - F[k, a]) for f, (k, a) in zip(fd, comps))
    tol = FD_TOL_FORCE * np.abs(F).max()
    assert err <= tol, f'force error {err:.3e} eV/A > {tol:.3e}'


@slow_model
@pytest.mark.parametrize('damping', ['zero', 'bj'])
@pytest.mark.parametrize('name, comps', [('cubic', [(0, 0), (1, 2)]), ('triclinic', [(1, 1), (0, 1)])])
def test_virial_is_strain_derivative(damping, name, comps, tmp_path):
    atoms = _fd_systems()[name]
    runner = _Runner(tmp_path, DAMPING_LINES[damping])
    _, _, W = runner.d3(atoms)
    jobs = [((lambda s, a=a, b=b: _strained(atoms, a, b, s)), H_STRAIN) for a, b in comps]
    fd = [-d for d in _derivatives(runner, jobs)]
    err = max(abs(w - W[a, b]) for w, (a, b) in zip(fd, comps))
    tol = FD_TOL_VIRIAL * np.abs(W).max()
    assert err <= tol, f'virial error {err:.3e} eV > {tol:.3e}'


# ---------------------------------------------------------------------------------------------
# 5. Robustness (AC-5)
# ---------------------------------------------------------------------------------------------
def rattled_fcc_cu(n=7):
    from ase.build import bulk
    cu = bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((n, n, n))
    cu.rattle(0.005, seed=11)
    return cu


@fast_model
@pytest.mark.parametrize('damping', ['zero', 'bj'])
def test_small_box_and_large_box_paths_agree(damping, tmp_path):
    """The path is chosen by the smallest thickness against 2.5 x potential cutoff (dftd3.cu):
    a 7x7x7 fcc Cu cell (25.27 A, 1372 atoms, 0.005 A rattle) is a large box for rc = 10.0 A
    (2.5 rc = 25.0) and a small box for rc = 10.15 A (25.375). The fcc shells nearest to the
    cutoffs lie at 9.886 and 10.211 A, so both cutoffs select the same pairs. The D3 energy, forces
    and virial must agree within the ADR-0002 caps (summation order only)."""
    cu = rattled_fcc_cu()
    results = {}
    for name, rc in (('large', 10.0), ('small', 10.15)):
        results[name] = d3_part(tmp_path / name, cu, f'dftd3 pbe {rc} 5{TOKEN[damping]}')
    (e_l, f_l, v_l), (e_s, f_s, v_s) = results['large'], results['small']
    assert abs(e_l) > 1.0
    assert abs(e_l - e_s) <= REL_CAP * abs(e_l), (e_l, e_s)
    assert np.abs(f_l - f_s).max() <= FORCE_CAP
    assert np.abs(v_l - v_s).max() <= REL_CAP * np.abs(v_l).max()


def overlapping_pair(periodic):
    """H-H pair at r = 0.002 A (r/R0 ~ 5e-4) plus a few atoms at normal distances."""
    pos = [[5.0, 5.0, 5.0], [5.002, 5.0, 5.0], [7.0, 5.0, 5.0], [5.0, 7.5, 5.0], [8.0, 8.0, 5.5]]
    return Atoms(['H', 'H', 'O', 'C', 'N'], positions=pos, cell=[12.0] * 3, pbc=periodic)


@fast_model
@pytest.mark.parametrize('damping', ['zero', 'bj'])
@pytest.mark.parametrize('periodic', [True, False], ids=['small_box', 'open_large_box_path'])
@pytest.mark.parametrize('distance', [0.05, 0.002, 1e-4])
def test_extreme_short_distance_is_finite(damping, periodic, distance, tmp_path):
    """r << R0 (R0(H-H) = 2.18 A): (r/R0)^-14 and ^-16 overflow float below r/R0 ~ 1e-2.. no NaN/inf
    may appear, and the D3 energy stays of the order of the normal pairs (zero damping: the close
    pair contributes ~ r^8 -> 0)."""
    atoms = overlapping_pair(periodic)
    p = atoms.positions
    p[1, 0] = p[0, 0] + distance
    atoms.set_positions(p)
    _, (e0, f0, v0) = run_gpumd(tmp_path / 'noD3', atoms, None)
    assert np.isfinite(e0) and np.isfinite(f0).all(), 'NEP itself is not finite at this distance'
    _, (e1, f1, v1) = run_gpumd(tmp_path / 'D3', atoms, f'dftd3 pbe 12 6{TOKEN[damping]}')
    assert np.isfinite(e1) and np.isfinite(f1).all() and np.isfinite(v1).all(), (e1, f1, v1)
    if damping == 'zero':
        assert abs(e1 - e0) < 1.0, f'D3(0) energy {e1 - e0:.3e} eV is not small'
