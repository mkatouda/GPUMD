"""charge_transfer qNEP: electric-field consistency, energy and neutrality checks (DR-005 stage 2c, VER-0034).

The BEC is dP/dr of P_direct = sum_a h_a V_a (VER-0029/0032); with `add_efield ... bec` the field force is
grad_r (sqrt(eps_inf) E.P_direct), so K + U - sqrt(eps_inf) E.P_direct(r_t) is conserved in NVE
(the thermo energy itself does not contain -E.P), and the work of the BEC over a closed loop vanishes.
P_direct and h come from the host / from gpumd's charge dump of the twin model without the charge_transfer line.
"""
import subprocess
import sys

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPTS = REPO_ROOT / 'specs' / 'verification' / 'VER-0034-qnep-charge-transfer-forces-gpu' / 'scripts'
if not (SCRIPTS / 'ct_field_tests.py').exists():
    pytest.skip('VER-0034 scripts not present', allow_module_level=True)
sys.path.insert(0, str(SCRIPTS))
import ct_checks as ck  # noqa: E402
import ct_field_tests as ft  # noqa: E402
import ct_forces_fd as fd  # noqa: E402
import make_ct_model as mk  # noqa: E402

EXE = str(GPUMD_EXECUTABLE)
E_FIELD = [0.3, -0.2, 0.5]
# (structure, rc_radial = rc_phi, setup kwargs, label)
CASES = [
    ('thin_n12_L7.5', 4.0, {}, 'small-box'),
    ('thick_n45_L14', 3.5, {}, 'large-box'),
    ('thin_n12_L7.5', 4.0, dict(triclinic=True), 'triclinic small-box'),
    ('thick_n45_L14', 3.5, dict(triclinic=True), 'triclinic large-box'),
    ('thin_n12_L7.5', 4.0, dict(charge_mode=3), 'mode 3 small-box'),
]


def _model(tmp_path, structure, rc, kw):
    text, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc, **kw)
    m = ft.Model(EXE, str(tmp_path), 1, rc, rc, 2, kw.get('charge_mode', 2))
    return m, pos, types, lat


@pytest.mark.parametrize('structure, rc, kw, label', CASES)
def test_add_efield_force_is_gradient_of_E_dot_P(tmp_path, structure, rc, kw, label):
    m, pos, types, lat = _model(tmp_path, structure, rc, kw)
    atoms = [0, len(pos) // 2, len(pos) - 1]
    res = ft.field_force_fd(m, pos, types, lat, E_FIELD, atoms)
    assert res['rms_Ffield'] > 0.03, (label, res)  # not vacuous
    assert res['rms_err_Ffield'] <= 3e-4, (label, res)  # eV/A (FD truncation at delta 0.02 dominates)
    assert res['max_err_Ffield'] <= 1e-3, (label, res)
    assert res['rms_err_Z'] <= 5e-4, (label, res)


@pytest.mark.parametrize('structure, rc, kw, label', CASES[:2])
def test_nve_with_field_conserves_K_U_minus_E_dot_P(tmp_path, structure, rc, kw, label):
    m, pos, types, lat = _model(tmp_path, structure, rc, kw)
    e0, h0 = ft.nve_field(m, pos, types, lat, [0, 0, 0], steps=600, tag='nve0')
    e1, h1 = ft.nve_field(m, pos, types, lat, E_FIELD, steps=600, tag='nve1')
    assert np.ptp(e0) <= 2e-3, (label, np.ptp(e0))  # field off: K + U conserved
    assert np.ptp(h1) <= 2e-3, (label, np.ptp(h1))  # field on: K + U - sqrt(eps) E.P conserved
    assert np.ptp(e1) >= 20 * np.ptp(h1), (label, np.ptp(e1), np.ptp(h1))  # K + U alone is NOT (non-vacuous)


def test_closed_loop_work_vanishes_for_ct_not_for_old(tmp_path):
    m, pos, types, lat = _model(tmp_path, 'thin_n12_L7.5', 4.0, {})
    old = mk.make_model_text(1, 4.0, 3.5, 4.0, False)
    res = ft.closed_loop(m, pos, types, lat, old, npts=24)
    I, S = res['ct']
    assert np.max(np.abs(I) / S) <= 3e-3, res['ct']
    Io, So = res['old']
    assert np.max(np.abs(Io) / So) >= 0.05, res['old']  # the old BEC is not a gradient: non-vacuous


@pytest.mark.parametrize('structure, rc', [('thin_n12_L7.5', 4.0), ('thick_n45_L14', 3.5)])
def test_static_energy_matches_host_reciprocal_sum(tmp_path, structure, rc):
    res = ck.energy_check(EXE, str(tmp_path), structure, rc)
    assert res['max_abs_q_zero_phi'] == 0.0
    assert abs(res['E_el_gpumd_ct']) > 0.1 or structure == 'thick_n8_L13', res
    assert res['abs_diff_ct'] <= 1e-5, res
    assert res['abs_diff_noct'] <= 1e-4, res
    assert res['max_dq_gpu_host'] <= 1e-6, res


def test_large_system_neutrality_and_bec_statistics(tmp_path):
    res = ck.large_n(EXE, str(tmp_path), natoms=3000, box=59.0)
    assert abs(res['sum_q_double_of_float_dump']) <= 1e-5, res
    assert res['max_abs_dq_run_to_run'] == 0.0, res
    assert res['max_abs_dZ_run_to_run'] <= 1e-5, res  # atomicAdd order of the float BEC kernels
    assert res['max_abs_sum_k_Z'] <= 1e-4, res  # translation invariance of P
    assert res['rms_Z'] > 0.05, res


def test_gpumd_replica_rejects_charge_transfer(tmp_path):
    replica = REPO_ROOT / 'src' / 'gpumd_replica'
    if not replica.exists():
        pytest.skip('gpumd_replica not built')
    name, natoms, box = mk.STRUCTURES[3]
    pos, types = mk.make_structure(natoms, box, 100 + 3)
    mk.write_extxyz(tmp_path / 'model.xyz', pos, types, box)
    (tmp_path / 'nep.txt').write_text(mk.make_model_text(1, 4.0, 3.5, 4.0, True))
    (tmp_path / 'run.in').write_text(
        'potential nep.txt\ntime_step 1\nensemble nvt_bdp 300 400 100\n'
        'multi_replica remd replicas 2 exchange 5 temp 300 400 equilibrate 0\nkspace ewald\nrun 10\n'
    )
    r = subprocess.run([str(replica)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert 'charge_transfer is not supported by gpumd_replica' in out, out[-500:]
