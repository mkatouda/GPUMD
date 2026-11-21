"""Flexible ZBL (`zbl 0 0`) of charge models in potentials/nep/nep_model_converter.py (DR-007 R1, AC-3).

Legacy layout (v5.9.1 for nep4_zbl_charge1/2; the fork's pre-merge reader for charge_transfer): header
`nep4_zbl_charge<m> nt elements`, `zbl 0 0`, `cutoff`, [`charge_transfer rc_phi`], `n_max`, `basis_size`,
`l_max`, `ann`, then one value per line: parameters (descriptor part pair-interleaved), q_scaler,
zbl parameters (10 per type pair t1 <= t2), phi.  There is no new -> old converter, so the tests build
the legacy file from a hand-made new-format model with an inverse re-layout written here
(independently of the converter), convert it, and require that the result equals the new-format
model block by block, and that gpumd gives identical results for both.  CPU only except the gpumd
runs.  No import from specs/.
"""
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
import test_qnep_flexible_zbl as zt

pytestmark = pytest.mark.fast

CONVERTER = REPO_ROOT / 'potentials' / 'nep' / 'nep_model_converter.py'

# label, fixture, charge_transfer rc_phi, model number
CASES = [
    ('qnep1', 'qnep_mode1_BaTiO3.txt', None, 1),
    ('qnep2', 'qnep_mode2_BaTiO3.txt', None, 2),
    ('qnep2-ct', 'qnep_mode2_BaTiO3.txt', 4.0, 2),
    ('qnep3-ct', 'qnep_mode2_BaTiO3.txt', 4.0, 3),
]
IDS = [c[0] for c in CASES]


def new_model(fixture, ct, mode):
    return zt.model_text(
        zt.read_fixture(fixture), 'flex', ct, None if mode != 3 else 3, zt.pair_params(zt.PAIR_RC))


def parse_new(text):
    fields, blocks, current, header = {}, {}, None, None
    for k, line in enumerate(text.splitlines()):
        t = line.split()
        if not t:
            continue
        if k == 0:
            header = t
        elif t[0] == '@':
            current = t[1]
            blocks[current] = []
            assert int(t[2]) > 0
        elif current is None:
            fields[t[0]] = t[1:]
        else:
            blocks[current].append(t[0])
    return header, fields, blocks


def to_legacy(text, mode):
    """Legacy file text of a new-format model (inverse of the converter's re-layout, own code)."""
    header, fields, blocks = parse_new(text)
    nt = int(header[1])
    nr, na = (int(x) for x in fields['n_max'])
    br, ba = (int(x) for x in fields['basis_size'])
    par = blocks['parameters']
    sizes = [(nr + 1) * (br + 1), (na + 1) * (ba + 1)]
    pairs = nt * nt
    n_desc = pairs * sum(sizes)
    out = par[:len(par) - n_desc]
    off = len(par) - n_desc
    for size in sizes:
        chunk = par[off:off + pairs * size]
        # new layout: [pair][k]; legacy layout: [k][pair]
        out += [chunk[pair * size + k] for k in range(size) for pair in range(pairs)]
        off += pairs * size
    assert off == len(par)
    out += blocks['q_scaler'] + blocks['zbl_parameters'] + blocks.get('phi_parameters', [])
    lines = ['nep4_zbl_charge%d %d %s' % (mode, nt, ' '.join(header[2:]))]
    for name in ('zbl', 'cutoff', 'charge_transfer', 'n_max', 'basis_size', 'l_max', 'ann'):
        if name in fields:
            lines.append(name + ' ' + ' '.join(fields[name]))
    return '\n'.join(lines + out) + '\n', len(par)


def convert(tmp_path, text, *extra, name='legacy.txt'):
    src, dst = tmp_path / name, tmp_path / 'new.txt'
    src.write_text(text)
    r = subprocess.run([sys.executable, str(CONVERTER), str(src), str(dst), *extra],
                       capture_output=True, text=True)
    return r, dst


@pytest.mark.parametrize('label, fixture, ct, mode', CASES, ids=IDS)
def test_flexible_zbl_charge_model_converts_and_round_trips(tmp_path, label, fixture, ct, mode):
    new = new_model(fixture, ct, mode)
    legacy, _ = to_legacy(new, mode)
    r, dst = convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    h0, f0, b0 = parse_new(new)
    h1, f1, b1 = parse_new(dst.read_text())
    assert h1 == h0
    assert f1['model'] == ['qnep%d' % mode] and f1['zbl'] == ['0', '0']
    assert ('charge_transfer' in f1) == (ct is not None)
    # pair order and counts: 10 * T (T + 1) / 2 values (nt = 3: 60), blocks of the new format
    assert len(b1['zbl_parameters']) == 60
    assert list(b1) == ['parameters', 'q_scaler', 'zbl_parameters'] + (['phi_parameters'] if ct else [])
    # the converted model equals the hand-made one, token for token
    for key in ('model', 'zbl', 'cutoff', 'n_max', 'basis_size', 'l_max', 'ann'):
        assert f1[key] == f0[key], key
    if ct:
        assert f1['charge_transfer'] == f0['charge_transfer']
    assert set(b1) == set(b0)
    for key in b0:
        assert b1[key] == b0[key], key
    # not vacuous: pair-specific values (the six pairs differ)
    vals = np.array([float(v) for v in b1['zbl_parameters']]).reshape(6, 10)
    assert len({tuple(row[:2]) for row in vals}) == 6


@pytest.mark.parametrize('label, fixture, ct, mode', CASES, ids=IDS)
def test_gpumd_on_converted_model_equals_hand_made_model(tmp_path, label, fixture, ct, mode):
    new = new_model(fixture, ct, mode)
    legacy, _ = to_legacy(new, mode)
    r, dst = convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    sym, pos, lat = zt.batio3_cluster(None, a=zt.FD_A, cells=zt.FD_CELLS)
    ref = zt.run_static(tmp_path, new, sym, pos, lat)
    got = zt.run_static(tmp_path, dst.read_text(), sym, pos, lat)
    none = zt.run_static(tmp_path, zt.model_text(zt.read_fixture(fixture), 'none', ct, None if mode != 3 else 3),
                         sym, pos, lat)
    assert np.max(np.abs(ref['forces'] - none['forces'])) > 1.0, 'ZBL does not contribute'
    assert abs(got['energy'] - ref['energy']) <= 1e-10 * abs(ref['energy'])
    assert np.max(np.abs(got['forces'] - ref['forces'])) <= 1e-9
    assert np.max(np.abs(got['virial'] - ref['virial'])) <= 1e-9 * np.max(np.abs(ref['virial']))


def _legacy_lines(mode=2, ct=4.0):
    return to_legacy(new_model('qnep_mode2_BaTiO3.txt', ct, mode), mode)[0].splitlines()


def test_zbl_pair_order_is_upper_triangle_row_major(tmp_path):
    # the pair block of the converted file keeps the legacy order (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)
    legacy = '\n'.join(_legacy_lines(2, None)) + '\n'
    r, dst = convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    _, _, b = parse_new(dst.read_text())
    inner = [float(b['zbl_parameters'][10 * k]) for k in range(6)]
    assert inner == [rc[0] for rc in zt.PAIR_RC]


@pytest.mark.parametrize(
    'name, edit, message',
    [
        ('typewise_factor', lambda L: [x if x != 'zbl 0 0' else 'zbl 0 0 0.65' for x in L],
         'flexible ZBL cannot have a cutoff factor'),
        ('missing_value', lambda L: L[:-1], 'expected'),
        ('extra_value', lambda L: L + ['1.0'], 'unexpected data'),
        ('inner_not_below_outer', lambda L: L[:L.index('ann 40 0') + 1 + 6098 + 37] + ['2.0', '1.0'] + L[L.index('ann 40 0') + 1 + 6098 + 37 + 2:],
         'each flexible ZBL pair requires 0 <= inner < outer'),
    ],
)
def test_flexible_zbl_charge_refusals(tmp_path, name, edit, message):
    r, dst = convert(tmp_path, '\n'.join(edit(_legacy_lines(2, None))) + '\n')
    assert r.returncode != 0
    assert message in r.stderr, r.stderr
    assert not dst.exists()


def test_flexible_zbl_still_refused_for_tensor_models(tmp_path):
    L = _legacy_lines(2, None)
    L[0] = L[0].replace('nep4_zbl_charge2', 'nep4_zbl_dipole')
    r, dst = convert(tmp_path, '\n'.join(L) + '\n')
    assert r.returncode != 0 and 'ZBL is not supported for nep_dipole' in r.stderr, r.stderr
    assert not dst.exists()


def test_flexible_zbl_vdw_models_still_refused(tmp_path):
    L = _legacy_lines(2, None)
    L[0] = L[0].replace('nep4_zbl_charge2', 'nep4_zbl_vdw')
    r, dst = convert(tmp_path, '\n'.join(L) + '\n')
    assert r.returncode != 0 and 'flexible ZBL is only supported for models nep and qnep1/2/3' in r.stderr, r.stderr
    assert not dst.exists()


def test_charge3_flexible_zbl_without_charge_transfer_rejected_as_before(tmp_path):
    L = _legacy_lines(3, None)  # qnep3 text without charge_transfer
    r, dst = convert(tmp_path, '\n'.join(L) + '\n')
    assert r.returncode == 2
    assert "unsupported model 'nep4_zbl_charge3'" in r.stderr, r.stderr
    assert not dst.exists()


def test_restart_with_charge_transfer_and_flexible_zbl_refused(tmp_path):
    legacy, n_par = to_legacy(new_model('qnep_mode2_BaTiO3.txt', 4.0, 2), 2)
    (tmp_path / 'legacy.restart').write_text('0 0.1\n' * n_par)
    r, dst = convert(tmp_path, legacy, '--restart', str(tmp_path / 'legacy.restart'),
                     '--restart-output', str(tmp_path / 'new.restart'))
    assert r.returncode != 0 and 'charge_transfer' in r.stderr, r.stderr
    assert not dst.exists() and not (tmp_path / 'new.restart').exists()


def test_restart_with_flexible_zbl_without_charge_transfer_converts(tmp_path):
    legacy, n_par = to_legacy(new_model('qnep_mode2_BaTiO3.txt', None, 2), 2)
    (tmp_path / 'legacy.restart').write_text('0 0.1\n' * n_par)
    r, dst = convert(tmp_path, legacy, '--restart', str(tmp_path / 'legacy.restart'),
                     '--restart-output', str(tmp_path / 'new.restart'))
    assert r.returncode == 0, r.stderr
    lines = (tmp_path / 'new.restart').read_text().splitlines()
    assert lines[0] == 'nep_restart' and lines[2] == 'parameters %d' % n_par
    assert len(lines) == 3 + n_par
