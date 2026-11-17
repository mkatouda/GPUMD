"""CPU-only tests of the charge_transfer extension of potentials/nep/nep_model_converter.py (DR-005).

Legacy layout (pre-merge fork readers, nep4_charge2/3): `charge_transfer <rc_phi>` line immediately after
`cutoff`; data lines: parameters, q_scaler, phi (num_pairs * (basis_size_radial + 1) values, pairs t1 <= t2
row-major). The converter must emit `charge_transfer <rc_phi>` and `@ phi_parameters N` in the new format
(merged reader: src/utilities/nep_io/nep_model_file_charge.cu). No gpumd or nep program is started.
"""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

pytestmark = pytest.mark.fast

CONVERTER = REPO_ROOT / 'potentials' / 'nep' / 'nep_model_converter.py'
MAKE_CT_MODEL = (
    REPO_ROOT / 'specs' / 'verification' / 'VER-0032-qnep-charge-transfer-gpu' / 'scripts' / 'make_ct_model.py'
)
if not MAKE_CT_MODEL.exists():
    pytest.skip('VER-0032 generator script not present', allow_module_level=True)

_spec = importlib.util.spec_from_file_location('make_ct_model', MAKE_CT_MODEL)
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


def convert(tmp_path, text, *extra):
    src = tmp_path / 'legacy.txt'
    dst = tmp_path / 'new.txt'
    src.write_text(text)
    r = subprocess.run([sys.executable, str(CONVERTER), str(src), str(dst), *extra],
                       capture_output=True, text=True)
    return r, dst


def parse_new(path):
    """Fields (name -> list of tokens) and blocks (name -> list of tokens) of a new-format file."""
    fields, blocks, current = {}, {}, None
    lines = path.read_text().splitlines()
    first = lines[0].split()
    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == '@':
            assert len(tokens) == 3
            current = tokens[1]
            assert current not in blocks
            blocks[current] = []
            assert int(tokens[2]) >= 0
            blocks[current + '#count'] = int(tokens[2])
        elif current is None:
            assert tokens[0] not in fields
            fields[tokens[0]] = tokens[1:]
        else:
            assert len(tokens) == 1
            blocks[current].append(tokens[0])
    return first, fields, blocks


def required_counts(nt):
    """Counts the merged C++ reader requires for the model of make_ct_model.py (independent formula)."""
    dim = (mk.N_MAX_RADIAL + 1) + (mk.N_MAX_ANGULAR + 1) * (mk.L_MAX + mk.HAS_Q_222 + mk.HAS_Q_1111)
    ann = (dim + 3) * mk.NEURONS * nt + 2
    desc = nt * nt * ((mk.N_MAX_RADIAL + 1) * (mk.BASIS_RADIAL + 1) + (mk.N_MAX_ANGULAR + 1) * (mk.BASIS_ANGULAR + 1))
    return ann + desc, dim, nt * (nt + 1) // 2 * (mk.BASIS_RADIAL + 1)


@pytest.mark.parametrize('charge_mode', [2, 3])
@pytest.mark.parametrize('nt, rc_phi', [(2, 4.0), (3, 3.25)])
def test_legacy_charge_transfer_conversion(tmp_path, charge_mode, nt, rc_phi):
    seed = 7
    legacy = mk.make_legacy_text(seed, 4.0, 3.5, rc_phi, True, nt=nt, charge_mode=charge_mode)
    r, dst = convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    first, fields, blocks = parse_new(dst)
    n_par, dim, n_phi = required_counts(nt)
    # (i) block names/counts required by the merged reader; (iv) model naming
    assert first == ['nep', str(nt)] + mk.SYMBOLS[:nt]
    assert fields['model'] == ['qnep%d' % charge_mode]
    assert fields['charge_transfer'] == ['%g' % rc_phi]
    assert [k for k in blocks if not k.endswith('#count')] == ['parameters', 'q_scaler', 'phi_parameters']
    assert (blocks['parameters#count'], blocks['q_scaler#count'], blocks['phi_parameters#count']) == (n_par, dim, n_phi)
    assert [len(blocks[k]) for k in ('parameters', 'q_scaler', 'phi_parameters')] == [n_par, dim, n_phi]
    # (ii) phi values are the last n_phi legacy data tokens, character for character; rc_phi token exact
    data = legacy.splitlines()
    data = data[next(i for i, l in enumerate(data) if l.startswith('ANN')) + 1:]
    assert blocks['phi_parameters'] == data[-n_phi:]
    assert blocks['q_scaler'] == data[n_par:n_par + dim]
    # (iii) all weights equal the direct new-format path of make_ct_model.py (same RNG) at %.9g precision
    direct = tmp_path / 'direct.txt'
    direct.write_text(mk.make_model_text(seed, 4.0, 3.5, rc_phi, True, nt=nt, charge_mode=charge_mode))
    _, dfields, dblocks = parse_new(direct)
    assert dfields == fields
    for name in ('parameters', 'q_scaler', 'phi_parameters'):
        a = np.array([float('%.9g' % float(v)) for v in blocks[name]])
        b = np.array([float('%.9g' % float(v)) for v in dblocks[name]])
        assert np.array_equal(a, b), name
        # sanity: not vacuous
        assert np.abs(a).max() > 0.05, name


def test_conversion_without_charge_transfer_has_no_phi(tmp_path):
    legacy = mk.make_legacy_text(1, 4.0, 3.5, 4.0, False, charge_mode=2)
    r, dst = convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    _, fields, blocks = parse_new(dst)
    assert 'charge_transfer' not in fields and 'phi_parameters' not in blocks


def test_zbl_charge2_with_charge_transfer(tmp_path):
    legacy = mk.make_legacy_text(1, 4.0, 3.5, 4.0, True, charge_mode=2)
    lines = legacy.splitlines()
    lines[0] = lines[0].replace('nep4_charge2', 'nep4_zbl_charge2')
    lines.insert(1, 'zbl 1 2')
    r, dst = convert(tmp_path, '\n'.join(lines) + '\n')
    assert r.returncode == 0, r.stderr
    _, fields, blocks = parse_new(dst)
    assert fields['zbl'] == ['1', '2'] and fields['charge_transfer'] == ['4']
    assert 'zbl_parameters' not in blocks and blocks['phi_parameters#count'] == 15


def _edit(mode, nt=2, rc_phi=4.0, charge_mode=2):
    text = mk.make_legacy_text(3, 4.0, 3.5, rc_phi, True, nt=nt, charge_mode=charge_mode)
    return mode(text.splitlines())


@pytest.mark.parametrize(
    'name, edit, message',
    [
        ('charge1', lambda L: [L[0].replace('charge2', 'charge1')] + L[1:], 'requires nep4'),
        ('plain', lambda L: [L[0].replace('nep4_charge2', 'nep4')] + L[1:], 'requires nep4'),
        ('zero', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer 0' for x in L],
         'rc_phi'),
        ('negative', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer -1' for x in L],
         'rc_phi'),
        ('above_cutoff', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer 4.5' for x in L],
         'rc_phi'),
        ('nan', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer nan' for x in L],
         'invalid number'),
        ('no_value', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer' for x in L],
         'malformed charge_transfer'),
        ('two_values', lambda L: [x if not x.startswith('charge_transfer') else 'charge_transfer 3 3' for x in L],
         'malformed charge_transfer'),
        ('duplicate', lambda L: L[:3] + ['charge_transfer 4'] + L[3:], 'malformed n_max'),
        ('missing_phi_value', lambda L: L[:-1], 'expected'),
        ('extra_value', lambda L: L + ['1.0'], 'unexpected data'),
        ('no_phi_block', lambda L: L[:-15], 'expected'),
    ],
)
def test_refusals(tmp_path, name, edit, message):
    r, dst = convert(tmp_path, '\n'.join(_edit(edit)) + '\n')
    assert r.returncode != 0
    assert message in r.stderr, r.stderr
    assert not dst.exists()


def test_flexible_zbl_with_charge_transfer_refused(tmp_path):
    L = _edit(lambda L: L)
    L[0] = L[0].replace('nep4_charge2', 'nep4_zbl_charge2')
    L.insert(1, 'zbl 0 0')
    r, dst = convert(tmp_path, '\n'.join(L) + '\n')
    assert r.returncode != 0 and 'flexible ZBL is only supported for model nep' in r.stderr, r.stderr
    assert not dst.exists()


def test_restart_with_charge_transfer_refused(tmp_path):
    legacy = mk.make_legacy_text(3, 4.0, 3.5, 4.0, True)
    (tmp_path / 'legacy.restart').write_text('0 0.1\n' * 10)
    r, dst = convert(tmp_path, legacy, '--restart', str(tmp_path / 'legacy.restart'),
                     '--restart-output', str(tmp_path / 'new.restart'))
    assert r.returncode != 0 and 'charge_transfer' in r.stderr, r.stderr
    assert not dst.exists() and not (tmp_path / 'new.restart').exists()
