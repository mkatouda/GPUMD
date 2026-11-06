"""`dftd2` QE-compatible table and plain cutoff (DR-004; ADR-0004): `dftd2 <s6> [<rc>] [table grimme2006|qe] [trunc sf|plain]`.

Written BEFORE the implementation (DES-0004 Phase 2): the tests of the new options must fail on the
pre-change binary because the feature is absent (the parser rejects the extra tokens), the tests of
the default path must pass on it.

Groups
1. Reference (AC-1): D2 contribution = (run with dftd2) - (same run without), NEP89, 4 PTFE crystal
   structures, rc = 12, s6 = 0.75, against the stored output of the requester's double-precision
   numpy reference (fixtures/dftd2_qe/ref_*.xyz; they are the oracle, never regenerated). The
   `grimme2006/sf` form is the harness check (passes on the pre-change binary).
   CPU only: the stored files equal `d2_reference.py` (the documented oracle) evaluated now.
2. Default compatibility (AC-2): bitwise comparison with the frozen output of the binary of the R3
   commit (fixtures/dftd2_qe/default_frozen.json; ADR-0005 item 5: regenerated once because the
   accumulation precision changed the last digits), header block text, no `table`/`trunc` header
   lines; and a tolerance comparison (DR-002 AC-1 caps) with the frozen output of the pre-R3 binary
   (default_frozen_pre_r3.json, develop af269060).
3. Parser and input errors (ADR-0004 items 1, 2, 4, 6; AC-6); header lines of non-default options (item 7).
4. Replication (AC-4); 5. Finite-difference force and virial consistency (AC-5).

Tolerances (DR-002 AC-1, as in test_dftd2_reference.py; fixed before any measurement, never to be
loosened to make a test pass; no absolute floors): energy |dE| <= 2e-6 |E_ref|; virial
max|dW| <= 2e-6 max|W_ref|; force max|dF| <= 1e-5 max|F_ref|.
Finite differences: force 2e-4 max|F_D2| (step 0.01 A), virial 2e-4 max|W_D2| (strain step 1e-3), as in
test_dftd2_consistency.py. For `trunc plain` the energy jumps when a pair crosses rc, so the
finite-difference systems are sparse and rc is chosen in a gap of the pair-distance spectrum
(>= 2 x the largest distance change during the stencil); the derivative of a hard cutoff is not
defined at the cutoff itself.
"""
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

FIXTURE_DIR = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd2_qe'
NEP_MODEL = REPO_ROOT / 'potentials' / 'nep' / 'nep89_20250409' / 'nep89_20250409.txt'
ZERO_NEP = FIXTURE_DIR / 'zero_nep_H_C_O_Cu.txt'
D2_REFERENCE_SHA256 = '2c0351f03180fcb1a1b23af5ab38d7991877aff7114fb4efd4d328c9024ca4e1'

needs_model = pytest.mark.skipif(not NEP_MODEL.exists(), reason='NEP89 model file not found')
fast = pytest.mark.fast
slow = pytest.mark.slow

STRUCTURES = ['vol000', 'volm05', 'volm10', 'volm15']
S6, RC = 0.75, 12.0
REL_CAP_EV, REL_CAP_F = 2e-6, 1e-5


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def read_ref(path):
    """Energy (eV), forces (eV/A), virial (3x3, eV) of a requester reference file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    energy = float(re.search(r'energy=([-0-9.eE+]+)', lines[1]).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', lines[1]).group(1).split(), float).reshape(3, 3)
    forces = np.array([ln.split()[4:7] for ln in lines[2:2 + n]], float)
    return energy, forces, virial


def ptfe(name):
    atoms = read(FIXTURE_DIR / f'ptfe_cryst157_{name}.xyz')
    return Atoms(atoms.get_chemical_symbols(), positions=atoms.positions, cell=atoms.cell.array, pbc=True)


def ref_path(name, table, trunc):
    return FIXTURE_DIR / f'ref_cryst157_{name}_{table}_{trunc}_rc12.xyz'


def run_gpumd(workdir, atoms, d2_line, model=NEP_MODEL, check=True):
    """Static run; returns (CompletedProcess, energy, forces, virial) (None x3 when the run failed)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    atoms.write(workdir / 'model.xyz', format='extxyz')
    lines = [f'potential {model}'] + ([d2_line] if d2_line else []) + \
        ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (workdir / 'run.in').write_text('\n'.join(lines) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    if not check and result.returncode != 0:
        return result, None, None, None
    assert result.returncode == 0, (result.stdout + result.stderr)[-1000:]
    header = (workdir / 'dump.xyz').read_text().splitlines()[1]
    energy = float(re.search(r'energy=([-0-9.eE+]+)', header).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', header).group(1).split(), float).reshape(3, 3)
    return result, energy, read(workdir / 'dump.xyz', format='extxyz').get_forces(), virial


def d2_part(tmp_path, atoms, d2_line):
    """(run with dftd2) - (same run without): energy, forces, virial of the D2 contribution."""
    _, e0, f0, w0 = run_gpumd(tmp_path / 'noD2', atoms, None)
    _, e1, f1, w1 = run_gpumd(tmp_path / 'D2', atoms, d2_line)
    return e1 - e0, f1 - f0, w1 - w0


def dump_lines(workdir):
    """dump.xyz lines 2.. without the `stress` field (varies run to run with the zero potential)."""
    lines = (Path(workdir) / 'dump.xyz').read_text().splitlines()[1:]
    lines[0] = re.sub(r' stress="[^"]*"', '', lines[0])
    return lines


def d2_header(stdout):
    return [ln for ln in stdout.splitlines() if re.search(r'dft-?d2', ln, re.I)]


OPTION_LINE = re.compile(r'\b(table|trunc)\s*=', re.I)


def expect_error(tmp_path, d2_line, patterns, atoms=None, forbidden=(r'requires 1 or 2 parameters',)):
    """The run is rejected, the message matches every pattern, and it is not the legacy
    one-or-two-parameters message (the grammar changes, so that text would be wrong)."""
    atoms = atoms if atoms is not None else ptfe('vol000')
    result, *_ = run_gpumd(tmp_path, atoms, d2_line, check=False)
    text = result.stdout + result.stderr
    assert result.returncode != 0, f'`{d2_line}` was accepted'
    assert 'invalid keyword' not in text.lower()
    for pat in patterns:
        assert re.search(pat, text, re.I), f'message for `{d2_line}` lacks /{pat}/:\n{text[-500:]}'
    for pat in forbidden:
        assert not re.search(pat, text, re.I), f'message for `{d2_line}` is the legacy one:\n{text[-500:]}'


# ---------------------------------------------------------------------------------------------
# 1. Reference (AC-1)
# ---------------------------------------------------------------------------------------------
@fast
def test_d2_reference_script_is_the_recorded_one():
    assert hashlib.sha256((FIXTURE_DIR / 'd2_reference.py').read_bytes()).hexdigest() == D2_REFERENCE_SHA256


@fast
@pytest.mark.parametrize('table, trunc', [('qe', 'plain'), ('qe', 'sf'), ('grimme2006', 'plain'),
                                          ('grimme2006', 'sf')])
@pytest.mark.parametrize('name', STRUCTURES)
def test_stored_reference_equals_the_documented_oracle(name, table, trunc):
    """CPU only: the stored reference file equals d2_reference.py evaluated now (10-digit files)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('d2_reference_fixture', FIXTURE_DIR / 'd2_reference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sp, X, L = mod.read_xyz(FIXTURE_DIR / f'ptfe_cryst157_{name}.xyz')
    E, F, W = mod.d2(sp, X, L, table, trunc, 12.0)
    E0, F0, W0 = read_ref(ref_path(name, table, trunc))
    assert abs(E - E0) <= 1e-9 * abs(E0)
    assert np.abs(F - F0).max() <= 1e-9 * np.abs(F0).max()      # files are printed with 11 significant digits
    assert np.abs(W - W0).max() <= 1e-9 * np.abs(W0).max()


REF_MODES = {   # id: (options appended to `dftd2 0.75 12`, reference file table/trunc)
    'qe_plain': ('table qe trunc plain', ('qe', 'plain')),
    'qe_sf': ('table qe trunc sf', ('qe', 'sf')),
    'grimme2006_plain': ('table grimme2006 trunc plain', ('grimme2006', 'plain')),
    'grimme2006_sf_default_form': ('', ('grimme2006', 'sf')),            # harness check
    'grimme2006_sf_explicit': ('table grimme2006 trunc sf', ('grimme2006', 'sf')),
}


@needs_model
@fast
@pytest.mark.parametrize('mode', list(REF_MODES))
@pytest.mark.parametrize('name', STRUCTURES)
def test_dftd2_matches_requester_reference(name, mode, tmp_path):
    opts, (table, trunc) = REF_MODES[mode]
    E_ref, F_ref, W_ref = read_ref(ref_path(name, table, trunc))
    e, f, w = d2_part(tmp_path, ptfe(name), f'dftd2 {S6} {RC:g} {opts}'.strip())
    tol_e, tol_f, tol_w = REL_CAP_EV * abs(E_ref), REL_CAP_F * np.abs(F_ref).max(), REL_CAP_EV * np.abs(W_ref).max()
    assert abs(e - E_ref) <= tol_e, f'energy error {abs(e - E_ref):.3e} eV (tol {tol_e:.1e})'
    assert np.abs(f - F_ref).max() <= tol_f, f'force error {np.abs(f - F_ref).max():.3e} eV/A (tol {tol_f:.1e})'
    assert np.abs(w - W_ref).max() <= tol_w, f'virial error {np.abs(w - W_ref).max():.3e} eV (tol {tol_w:.1e})'


# ---------------------------------------------------------------------------------------------
# 2. Default compatibility (AC-2)
# ---------------------------------------------------------------------------------------------
FROZEN = json.load(open(FIXTURE_DIR / 'default_frozen.json'))
FROZEN_PRE_R3 = json.load(open(FIXTURE_DIR / 'default_frozen_pre_r3.json'))
FROZEN_CASES = [k for k in FROZEN if not k.startswith('_')]
FROZEN_INPUTS = ['dftd2 0.75', 'dftd2 0.75 12']


@fast
@pytest.mark.parametrize('d2_line', FROZEN_INPUTS)
@pytest.mark.parametrize('name', FROZEN_CASES)
def test_default_form_bit_identical_to_frozen_output(name, d2_line, tmp_path):
    """Dump text (energy, virial, forces; `precision double`) and the DFT-D2 header lines of the
    default-form inputs equal the frozen output of the R3 binary (run-to-run reproducibility), byte
    for byte; the default output has no `table`/`trunc` lines (ADR-0004 item 7)."""
    case = FROZEN[name]
    result, *_ = run_gpumd(tmp_path, read(FIXTURE_DIR / case['structure']), d2_line, model=ZERO_NEP)
    assert dump_lines(tmp_path) == case['inputs'][d2_line]['D2']
    assert d2_header(result.stdout) == case['inputs'][d2_line]['header']
    assert not [ln for ln in result.stdout.splitlines() if OPTION_LINE.search(ln)]


def parse_dump_lines(lines):
    """energy, forces (n x 3), virial (3 x 3) from stored dump.xyz lines 2.. (header, then atoms)."""
    energy = float(re.search(r'energy=([-0-9.eE+]+)', lines[0]).group(1))
    virial = np.array(re.search(r'virial="([^"]+)"', lines[0]).group(1).split(), float).reshape(3, 3)
    forces = np.array([ln.split()[4:7] for ln in lines[1:]], float)
    return energy, forces, virial


@fast
@pytest.mark.parametrize('d2_line', FROZEN_INPUTS)
@pytest.mark.parametrize('name', [k for k in FROZEN_PRE_R3 if not k.startswith('_')])
def test_default_form_matches_pre_r3_frozen_output_within_ac1_caps(name, d2_line, tmp_path, record_property):
    """ADR-0005 item 5 (a): the R3 binary (compensated float accumulation) against the frozen
    output of the pre-R3 binary (develop af269060, double accumulation), at the DR-002 AC-1 level:
    energy and virial 2e-6 of |E_old| and max|W_old|, forces 1e-5 of max|F_old|. The measured
    differences are recorded (pytest -s prints them)."""
    case = FROZEN_PRE_R3[name]
    result, *_ = run_gpumd(tmp_path, read(FIXTURE_DIR / case['structure']), d2_line, model=ZERO_NEP)
    e0, f0, w0 = parse_dump_lines(case['inputs'][d2_line]['D2'])
    e1, f1, w1 = parse_dump_lines(dump_lines(tmp_path))
    de, dw, df = abs(e1 - e0) / abs(e0), np.abs(w1 - w0).max() / np.abs(w0).max(), np.abs(f1 - f0).max() / np.abs(f0).max()
    for key, val, cap in (('energy', de, REL_CAP_EV), ('virial', dw, REL_CAP_EV), ('force', df, REL_CAP_F)):
        record_property(f'{key}_diff_cap_fraction', f'{val / cap:.3e}')
    print(f'\nPRE-R3 {name} / {d2_line}: dE {de:.2e} ({de / REL_CAP_EV:.1%} of cap), '
          f'dW {dw:.2e} ({dw / REL_CAP_EV:.1%}), dF {df:.2e} ({df / REL_CAP_F:.1%})')
    assert d2_header(result.stdout) == case['inputs'][d2_line]['header']
    assert de <= REL_CAP_EV and dw <= REL_CAP_EV and df <= REL_CAP_F


@fast
@pytest.mark.parametrize('name', FROZEN_CASES)
def test_explicit_default_options_bit_identical_to_frozen_output(name, tmp_path):
    """`table grimme2006 trunc sf` (explicit defaults), with and without a cutoff, reproduces the
    frozen default-form output bit for bit, header block included."""
    case = FROZEN[name]
    atoms = read(FIXTURE_DIR / case['structure'])
    for base, line in (('dftd2 0.75', 'dftd2 0.75 table grimme2006 trunc sf'),
                       ('dftd2 0.75 12', 'dftd2 0.75 12 trunc sf table grimme2006')):
        sub = tmp_path / base.replace(' ', '_')
        result, *_ = run_gpumd(sub, atoms, line, model=ZERO_NEP)
        assert dump_lines(sub) == case['inputs'][base]['D2'], line
        assert d2_header(result.stdout) == case['inputs'][base]['header'], line
        assert not [ln for ln in result.stdout.splitlines() if OPTION_LINE.search(ln)]


# ---------------------------------------------------------------------------------------------
# 3. Parser, errors, header lines
# ---------------------------------------------------------------------------------------------
@needs_model
@fast
@pytest.mark.parametrize('line', [
    'dftd2 0.75 12 table qe trunc plain',
    'dftd2 0.75 12 trunc plain table qe',
    'dftd2 0.75 12 table qe',
    'dftd2 0.75 12 trunc plain',
    'dftd2 0.75 12 trunc sf',
    'dftd2 0.75 12 table grimme2006',
    'dftd2 0.75 12 table grimme2006 trunc sf',
    'dftd2 0.75 12 trunc sf table grimme2006',
    'dftd2 0.75 table grimme2006 trunc sf',
    'dftd2 0.75 table grimme2006',
    'dftd2 0.75 trunc sf',
])
def test_valid_option_forms_run(line, tmp_path):
    _, e, f, w = run_gpumd(tmp_path, ptfe('vol000'), line)
    assert np.isfinite(e) and np.all(np.isfinite(f)) and np.all(np.isfinite(w))


@needs_model
@fast
def test_option_order_does_not_matter(tmp_path):
    outs = []
    for sub, line in (('a', 'dftd2 0.75 12 table qe trunc plain'), ('b', 'dftd2 0.75 12 trunc plain table qe')):
        outs.append(d2_part(tmp_path / sub, ptfe('vol000'), line))
    (ea, fa, wa), (eb, fb, wb) = outs
    assert abs(ea - eb) <= REL_CAP_EV * abs(ea)
    assert np.abs(fa - fb).max() <= REL_CAP_F * np.abs(fa).max()
    assert np.abs(wa - wb).max() <= REL_CAP_EV * np.abs(wa).max()


@needs_model
@fast
@pytest.mark.parametrize('line', [
    'dftd2 0.75 table qe',                      # item 6: table qe needs an explicit cutoff
    'dftd2 0.75 table qe trunc sf',
    'dftd2 0.75 trunc plain',                   # item 6: trunc plain needs an explicit cutoff
    'dftd2 0.75 table qe trunc plain',
])
def test_new_options_require_an_explicit_cutoff(line, tmp_path):
    expect_error(tmp_path, line, [r'cutoff', r'explicit'])


@needs_model
@fast
@pytest.mark.parametrize('line, patterns', [
    ('dftd2 0.75 12 table foo', [r'table', r'foo', r'grimme2006', r'\bqe\b']),
    ('dftd2 0.75 12 trunc foo', [r'trunc', r'foo', r'\bplain\b', r'\bsf\b']),
    ('dftd2 0.75 12 table QE', [r'table', r'QE', r'grimme2006', r'\bqe\b']),      # case sensitive (item 2)
    ('dftd2 0.75 12 trunc PLAIN', [r'trunc', r'PLAIN', r'\bplain\b', r'\bsf\b']),
    ('dftd2 0.75 12 table', [r'table', r'grimme2006', r'\bqe\b']),                # missing value
    ('dftd2 0.75 12 trunc', [r'trunc', r'\bplain\b', r'\bsf\b']),
    ('dftd2 0.75 12 table qe trunc', [r'trunc', r'\bplain\b', r'\bsf\b']),
    ('dftd2 0.75 12 trunc plain table', [r'table', r'grimme2006', r'\bqe\b']),
    ('dftd2 0.75 12 table qe table qe', [r'table', r'(once|more than|repeat|duplicate|twice)']),
    ('dftd2 0.75 12 table qe table grimme2006', [r'table', r'(once|more than|repeat|duplicate|twice)']),
    ('dftd2 0.75 12 trunc plain trunc sf', [r'trunc', r'(once|more than|repeat|duplicate|twice)']),
    ('dftd2 0.75 12 foo', [r'foo']),                                              # unknown token
    ('dftd2 0.75 12 foo bar', [r'foo']),
    ('dftd2 0.75 12 table qe foo', [r'foo']),
    ('dftd2 0.75 12 table qe trunc plain extra', [r'extra']),
    ('dftd2 0.75 table qe 12', [r'(\b12\b|cutoff.*(before|first|directly|immediately|follow|precede))']),
    ('dftd2 0.75 12 table qe 12', [r'(\b12\b|cutoff.*(before|first|directly|immediately|follow|precede))']),
    ('dftd2 0.75 12 12', [r'\b12\b|cutoff']),                                     # two numbers
])
def test_option_errors_are_informative(line, patterns, tmp_path):
    expect_error(tmp_path, line, patterns)


@needs_model
@fast
@pytest.mark.parametrize('line, patterns', [
    ('dftd2', [r'dftd2']),
    ('dftd2 abc 12', [r's6|number|numeric']),
    ('dftd2 0 12', [r's6']),
    ('dftd2 -0.75 12', [r's6']),
    ('dftd2 0.75 0', [r'cutoff']),
    ('dftd2 0.75 -12', [r'cutoff']),
    ('dftd2 0.75 0 table qe', [r'cutoff']),
    ('dftd2 0.75 12 table qe\ndftd2 0.75 12', [r'dftd2.*once']),
    ('dftd2 0.75 12\ndftd2 0.75 12 table qe', [r'dftd2.*once']),
    ('dftd2 0.75 12 table qe\ndftd3 pbe 12 6', [r'dftd2', r'dftd3']),
    ('dftd3 pbe 12 6\ndftd2 0.75 12 table qe trunc plain', [r'dftd2', r'dftd3']),
])
def test_legacy_and_exclusion_errors_stay(line, patterns, tmp_path):
    """Legacy messages keep their meaning, and the new forms obey the same exclusion / once-only rules."""
    expect_error(tmp_path, line, patterns, forbidden=())


@needs_model
@fast
@pytest.mark.parametrize('line', ['dftd2 0.75 12', 'dftd2 0.75 12 table grimme2006', 'dftd2 0.75 12 trunc plain'])
def test_beyond_z86_error_names_elements_grimme_table(line, tmp_path):
    """`table grimme2006` keeps the existing element rule (Z <= 86; error naming the elements,
    checked at the first force evaluation); the legacy form passes on the pre-change binary."""
    atoms = Atoms('CUAcH', positions=[[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 3]],
                  cell=[20, 20, 20], pbc=True)
    expect_error(tmp_path, line, [r'\bU\b', r'\bAc\b', r'Z <= 86'], atoms=atoms)


@needs_model
@fast
@pytest.mark.parametrize('line', ['dftd2 0.75 12 table qe', 'dftd2 0.75 12 table qe trunc plain',
                                  'dftd2 0.75 12 trunc sf table qe'])
@pytest.mark.parametrize('symbol', ['U', 'Pu'])
def test_beyond_z86_with_table_qe_names_the_element(symbol, line, tmp_path):
    atoms = Atoms(f'CF{symbol}', positions=[[0, 0, 0], [2, 0, 0], [0, 3, 3]], cell=[20, 20, 20], pbc=True)
    expect_error(tmp_path, line, [rf'\b{symbol}\b'], atoms=atoms)


@needs_model
@fast
def test_table_qe_runs_when_no_element_beyond_z86_is_present(tmp_path):
    """NEP89 contains Ac..Pu, but only the elements present in the structure matter."""
    _, e, f, _ = run_gpumd(tmp_path, ptfe('vol000'), 'dftd2 0.75 12 table qe trunc plain')
    assert np.isfinite(e) and np.all(np.isfinite(f))


@needs_model
@fast
@pytest.mark.parametrize('line, expected', [
    ('dftd2 0.75 12 table qe trunc plain', [r'\btable\s*=\s*qe\b', r'\btrunc\s*=\s*plain\b']),
    ('dftd2 0.75 12 table qe', [r'\btable\s*=\s*qe\b']),
    ('dftd2 0.75 12 trunc plain', [r'\btrunc\s*=\s*plain\b']),
    ('dftd2 0.75 12 table grimme2006 trunc plain', [r'\btrunc\s*=\s*plain\b']),
])
def test_header_lines_for_non_default_options(line, expected, tmp_path):
    result, *_ = run_gpumd(tmp_path, ptfe('vol000'), line)
    opt_lines = [ln for ln in result.stdout.splitlines() if OPTION_LINE.search(ln)]
    for pat in expected:
        assert any(re.search(pat, ln) for ln in opt_lines), (pat, opt_lines)
    # only the non-default values are printed
    assert len(opt_lines) == len(expected), opt_lines
    assert not any(re.search(r'\btable\s*=\s*grimme2006\b|\btrunc\s*=\s*sf\b', ln) for ln in opt_lines)
    assert any('DFT-D2' in ln for ln in result.stdout.splitlines())      # the existing header line stays


@needs_model
@fast
@pytest.mark.parametrize('line', ['dftd2 0.75 12', 'dftd2 0.75', 'dftd2 0.75 12 table grimme2006 trunc sf'])
def test_no_option_header_lines_for_defaults(line, tmp_path):
    result, *_ = run_gpumd(tmp_path, ptfe('vol000'), line, check=False)
    if result.returncode != 0:      # `table`/`trunc` not implemented yet: the parser error is the failure
        pytest.fail((result.stdout + result.stderr)[-300:])
    assert not [ln for ln in result.stdout.splitlines() if OPTION_LINE.search(ln)]


# ---------------------------------------------------------------------------------------------
# 4. Replication (AC-4)
# ---------------------------------------------------------------------------------------------
@needs_model
@fast
@pytest.mark.parametrize('trunc', ['plain', 'sf'])
def test_replicated_supercell_gives_same_energy_per_atom_and_forces(trunc, tmp_path):
    """4x4x2 supercell of the 45-atom PTFE cell (the cell itself is thin: 4.9 A < rc in y, so the many-
    image path; the supercell widths 22.7 / 19.7 / 39.4 A are of the order of 2 rc), table qe, rc = 12:
    same energy per atom and the same per-atom forces, within the AC-1 tolerances."""
    unit = ptfe('vol000')
    rep = unit.repeat((4, 4, 2))
    line = f'dftd2 {S6} {RC:g} table qe trunc {trunc}'
    e1, f1, w1 = d2_part(tmp_path / 'unit', unit, line)
    e2, f2, w2 = d2_part(tmp_path / 'rep', rep, line)
    n = len(unit)
    nrep = len(rep) // n
    tol_e = REL_CAP_EV * abs(e1) / n
    tol_f = REL_CAP_F * np.abs(f1).max()
    assert abs(e2 / len(rep) - e1 / n) <= tol_e, (e2 / len(rep), e1 / n)
    f2_blocks = f2.reshape(nrep, n, 3)      # ase repeat: copy k holds atoms k*n .. (k+1)*n - 1
    assert np.abs(f2_blocks - f1[None]).max() <= tol_f
    tol_w = REL_CAP_EV * np.abs(w1).max()
    assert np.abs(w2 / nrep - w1).max() <= tol_w


@needs_model
@slow
def test_replicated_supercell_large_box_regime(tmp_path):
    """5x5x2 supercell: all perpendicular widths exceed 2 rc = 24 A (28.4 / 24.6 / 39.4 A)."""
    unit = ptfe('vol000')
    rep = unit.repeat((5, 5, 2))
    line = f'dftd2 {S6} {RC:g} table qe trunc plain'
    e1, f1, _ = d2_part(tmp_path / 'unit', unit, line)
    e2, f2, _ = d2_part(tmp_path / 'rep', rep, line)
    n = len(unit)
    assert abs(e2 / len(rep) - e1 / n) <= REL_CAP_EV * abs(e1) / n
    assert np.abs(f2.reshape(-1, n, 3) - f1[None]).max() <= REL_CAP_F * np.abs(f1).max()


# ---------------------------------------------------------------------------------------------
# 5. Finite differences (AC-5)
# ---------------------------------------------------------------------------------------------
S6_FD = 75.0
H_FORCE, H_STRAIN = 0.01, 1e-3
FD_TOL = 2e-4


def _sparse_system(periodic):
    """C/F atoms in a sparse cell (deterministic): pair distances stay >= 3 A and the distance
    spectrum near rc has a gap, found below."""
    rng = np.random.default_rng(20261106)
    if periodic:
        cell, pbc, sym = np.array([[9.0, 0, 0], [1.3, 8.6, 0], [-0.9, 1.1, 9.4]]), True, ['C', 'F', 'F']
    else:
        cell, pbc, sym = np.diag([16.0, 16.0, 16.0]), False, ['C', 'F', 'F', 'C', 'F', 'F']
    while True:
        atoms = Atoms(sym, positions=rng.uniform(0, 1, (len(sym), 3)) @ cell, cell=cell, pbc=pbc)
        if len(neighbor_list('d', atoms, 3.0)) == 0:
            return atoms


def _rc_in_gap(atoms, lo, hi):
    d = np.sort(neighbor_list('d', atoms, hi))
    d = np.concatenate([[lo], d[d >= lo], [hi]])
    i = int(np.argmax(np.diff(d)))
    return 0.5 * (d[i] + d[i + 1]), float(np.diff(d)[i])


class _Runner:
    def __init__(self, root):
        self.root, self.count = Path(root), 0

    def d2(self, atoms, line):
        self.count += 1
        base = self.root / f'r{self.count}'
        e1, f1, w1 = run_gpumd(base / 'D2', atoms, line)[1:]
        e0, f0, w0 = run_gpumd(base / 'noD2', atoms, None)[1:]
        return e1 - e0, f1 - f0, w1 - w0


def _five_point(energies, h):
    em2, em1, ep1, ep2 = energies
    return (em2 - 8 * em1 + 8 * ep1 - ep2) / (12 * h)


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


def _derivatives(runner, line, jobs):
    def one(job):
        build, h = job
        return _five_point([runner.d2(build(s), line)[0] for s in (-2 * h, -h, h, 2 * h)], h)
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, jobs))


FD_MODES = {    # id: options; `default_form` is the harness check on the pre-change binary
    'default_form': '',
    'qe_sf': 'table qe trunc sf',
    'qe_plain': 'table qe trunc plain',
    'grimme2006_plain': 'table grimme2006 trunc plain',
}


@needs_model
@slow
@pytest.mark.parametrize('mode', list(FD_MODES))
@pytest.mark.parametrize('periodic', [True, False], ids=['periodic', 'open'])
def test_force_is_minus_energy_gradient(mode, periodic, tmp_path):
    atoms = _sparse_system(periodic)
    rc, gap = _rc_in_gap(atoms, 7.0, 11.0)
    assert gap >= 0.3, gap          # >= 2 x (2h + strain shift): no pair crosses rc during the stencil
    line = f'dftd2 {S6_FD} {rc:.6f} {FD_MODES[mode]}'.strip()
    runner = _Runner(tmp_path)
    _, F, _ = runner.d2(atoms, line)
    comps = [(k, a) for k in range(len(atoms)) for a in range(3)][:4]
    jobs = [((lambda s, k=k, a=a: _displaced(atoms, k, a, s)), H_FORCE) for k, a in comps]
    fd = [-d for d in _derivatives(runner, line, jobs)]
    err = max(abs(f - F[k, a]) for f, (k, a) in zip(fd, comps))
    tol = FD_TOL * np.abs(F).max()
    assert err <= tol, f'force error {err:.3e} eV/A > {tol:.3e}'


@needs_model
@slow
@pytest.mark.parametrize('mode', list(FD_MODES))
def test_virial_is_strain_derivative(mode, tmp_path):
    atoms = _sparse_system(True)
    rc, gap = _rc_in_gap(atoms, 7.0, 11.0)
    assert gap >= 0.3, gap
    line = f'dftd2 {S6_FD} {rc:.6f} {FD_MODES[mode]}'.strip()
    runner = _Runner(tmp_path)
    _, _, W = runner.d2(atoms, line)
    comps = [(0, 0), (1, 2), (0, 1), (2, 2)]
    jobs = [((lambda s, a=a, b=b: _strained(atoms, a, b, s)), H_STRAIN) for a, b in comps]
    fd = [-d for d in _derivatives(runner, line, jobs)]
    err = max(abs(w - W[a, b]) for w, (a, b) in zip(fd, comps))
    tol = FD_TOL * np.abs(W).max()
    assert err <= tol, f'virial error {err:.3e} eV > {tol:.3e}'
