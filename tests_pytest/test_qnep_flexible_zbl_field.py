"""Flexible ZBL together with charge_transfer: charges/BEC independence and field conservation
(DR-007 AC-4, the two parts not covered by the finite-difference tests of test_qnep_flexible_zbl.py).

a. the flexible ZBL term does not enter the charge or the BEC kernels: for the same structure and the
   same NEP weights, charges and BEC (dump_xyz, precision double) of the model with flexible ZBL equal
   those of the model without the `zbl` line.  The tolerance is the run-to-run difference of the
   no-ZBL model with itself, measured in the test (0.0 for these cases on RTX A6000, sm_86, 3 repeats:
   charges and BEC are bit-identical).  ZBL must be active: the forces must differ (by > 1 eV/A).
b. field conservation with flexible ZBL + charge_transfer, same gates as test_qnep_charge_transfer_field.py
   (NVE: ptp(K+U-E.P) <= 2e-3 eV and K+U alone worse by >= 20x; add_efield force is the gradient of
   sqrt(eps) E.P: RMS 3e-4, max 1e-3 eV/A, Z RMS 5e-4).  The ZBL term is a plain pair potential, so it
   is conserved by NVE but cancels in the field force difference.
   The NVE check runs on the large-box structure (thick_n45_L14) only.  The thin_n12_L7.5 small-box NVE
   case was removed: `velocity 300` is unseeded and ptp(K+U) of this structure fluctuates around the 2e-3 eV
   gate even without ZBL (10 runs: 0.62-1.03e-3 eV, one 2.38e-3), while with ZBL the velocity-Verlet error
   of the 1.6 A contacts scales as dt^2 (8.8e-3, 2.0e-3, 3.4e-4 eV for dt 0.5, 0.25, 0.1 fs).  The
   small-box dynamic check is replaced by the static small-vs-large agreement of
   test_qnep_flexible_zbl.py (AC-2d) and a measured comparison of 10 trajectory frames, small-box vs
   2x2x2 large-box path (max dF 1.3e-5 eV/A, max dE 2.3e-6 eV/atom, no growth along the trajectory).
"""
import sys

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPTS = REPO_ROOT / 'specs' / 'verification' / 'VER-0034-qnep-charge-transfer-forces-gpu' / 'scripts'
if not (SCRIPTS / 'ct_field_tests.py').exists():
    pytest.skip('VER-0034 scripts not present', allow_module_level=True)
sys.path.insert(0, str(SCRIPTS))
import ct_field_tests as ft  # noqa: E402
import ct_forces_fd as fd  # noqa: E402
from test_qnep_flexible_zbl import pair_params  # noqa: E402

EXE = str(GPUMD_EXECUTABLE)
E_FIELD = [0.3, -0.2, 0.5]
# pair-specific windows (inner, outer) for Na-Na, Na-Cl, Cl-Cl; the structures have a minimum
# distance of 1.6 A, so pairs sit inside the switching regions (ZBL contribution 4-21 eV)
ZBL_VALUES = pair_params([(1.0, 2.4), (1.2, 2.6), (1.0, 2.2)])


def add_flexible_zbl(text):
    values = ZBL_VALUES
    lines = text.split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith('@ '))
    lines.insert(i, 'zbl 0 0')
    out = '\n'.join(lines).rstrip('\n') + '\n'
    return out + '@ zbl_parameters %d\n' % len(values) + ''.join('%.9g\n' % v for v in values)


# (structure, rc, charge mode, charge_transfer, label)
IND_CASES = [
    ('thin_n12_L7.5', 4.0, 2, True, 'qnep2 ct small-box'),
    ('thick_n45_L14', 3.5, 2, True, 'qnep2 ct large-box'),
    ('thin_n12_L7.5', 4.0, 3, True, 'qnep3 ct small-box'),
    ('thick_n45_L14', 3.5, 3, True, 'qnep3 ct large-box'),
    ('thin_n12_L7.5', 4.0, 2, False, 'qnep2 no-ct small-box'),
]


@pytest.mark.parametrize('structure, rc, mode, ct, label', IND_CASES, ids=[c[4] for c in IND_CASES])
def test_charges_and_bec_independent_of_flexible_zbl(tmp_path, structure, rc, mode, ct, label):
    text, pos, types, lat, _ = fd.setup(structure, 1, not ct, rc, rc, mode)
    zbl = add_flexible_zbl(text)

    def run(t, tag):
        return fd.run_static(EXE, str(tmp_path), tag, t, pos, types, lat, props='charge force bec')

    n0, n1, n2 = run(text, 'n0'), run(text, 'n1'), run(text, 'n2')
    z0, z1 = run(zbl, 'z0'), run(zbl, 'z1')

    def diff(a, b, k):
        return float(np.max(np.abs(a[k] - b[k])))

    for k in ('charge', 'bec'):
        rep = max(diff(n0, n1, k), diff(n0, n2, k))  # run-to-run difference of the no-ZBL model
        for z in (z0, z1):
            for n in (n0, n1, n2):
                assert diff(z, n, k) <= rep, (label, k, diff(z, n, k), rep)
    # ZBL is active: forces and energy differ strongly
    assert diff(z0, n0, 'forces') > 1.0, label
    assert abs(z0['energy'] - n0['energy']) > 1.0, label


def _model(tmp_path, structure, rc, mode):
    m = ft.Model(EXE, str(tmp_path), 1, rc, rc, 2, mode)
    m.ct = add_flexible_zbl(m.ct)
    m.noct = add_flexible_zbl(m.noct)
    _, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc, mode)
    return m, pos, types, lat


FIELD_CASES = [
    ('thin_n12_L7.5', 4.0, 2, 'small-box'),
    ('thick_n45_L14', 3.5, 2, 'large-box'),
    ('thin_n12_L7.5', 4.0, 3, 'mode 3 small-box'),
]


@pytest.mark.parametrize('structure, rc, mode, label', FIELD_CASES, ids=[c[3] for c in FIELD_CASES])
def test_add_efield_force_is_gradient_of_E_dot_P_with_zbl(tmp_path, structure, rc, mode, label):
    m, pos, types, lat = _model(tmp_path, structure, rc, mode)
    atoms = [0, len(pos) // 2, len(pos) - 1]
    res = ft.field_force_fd(m, pos, types, lat, E_FIELD, atoms)
    assert res['rms_Ffield'] > 0.03, (label, res)
    assert res['rms_err_Ffield'] <= 3e-4, (label, res)
    assert res['max_err_Ffield'] <= 1e-3, (label, res)
    assert res['rms_err_Z'] <= 5e-4, (label, res)


@pytest.mark.parametrize('structure, rc, mode, label', FIELD_CASES[1:2], ids=[c[3] for c in FIELD_CASES[1:2]])
def test_nve_with_field_conserves_K_U_minus_E_dot_P_with_zbl(tmp_path, structure, rc, mode, label):
    m, pos, types, lat = _model(tmp_path, structure, rc, mode)
    e0, h0 = ft.nve_field(m, pos, types, lat, [0, 0, 0], steps=600, tag='nve0')
    e1, h1 = ft.nve_field(m, pos, types, lat, E_FIELD, steps=600, tag='nve1')
    assert np.ptp(e0) <= 2e-3, (label, np.ptp(e0))
    assert np.ptp(h1) <= 2e-3, (label, np.ptp(h1))
    assert np.ptp(e1) >= 20 * np.ptp(h1), (label, np.ptp(e1), np.ptp(h1))
