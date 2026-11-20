"""Flexible ZBL (`zbl 0 0` + `zbl_parameters`) for the charge models in replica dynamics (DR-007 R5, AC-5).

Replica dynamics (`multi_replica` in gpumd) has its own copy of the charge-model force code
(src/force/replica/); this file tests the flexible-ZBL branch of that copy.

  a. displacement-identity gate (the method, its derivation, and the self-calibration are those of
     test_replica_charge_transfer.py, whose helpers are reused): one velocity-Verlet step of the
     replica gives the force at x1 as D = 2 m C^2 (v1 - u) / dt; RMS(D - F_gpumd(x1)) <= 1e-4 eV/A
     (DR-007 AC-5), evaluated only if the self-calibrated floor of the method (ordinary gpumd, same
     model, same step) is <= 1e-5 eV/A.  No BEC and no virial gate (the replica computes neither).
     Cases: qnep1/qnep2 with flexible ZBL (pair-specific parameters), qnep2/qnep3 with
     charge_transfer + flexible ZBL, in the large-box and the small-box kernels of the replica.
     ZBL must contribute clearly: RMS of (F with ZBL) - (F without ZBL) at x1 from ordinary gpumd is
     required to be above 1 eV/A.
  b. equivalence: flexible parameters equal to the universal ones give the same replica run as the
     universal ZBL (energy of the restart file and positions after one step).

The structure is the finite-difference structure of test_qnep_flexible_zbl.py (a = 3.0 A BaTiO3, 10
atoms, every pair has neighbours in its switching window) with the NEP part scaled by 0.1 so that the
forces stay moderate; the models are built from the checked-in fixtures (no specs/ needed for the
models).  Skipped when the VER-0032/VER-0034 scripts imported by test_replica_charge_transfer.py are
missing.  Needs one GPU.
"""
import subprocess

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE
import test_qnep_flexible_zbl as zt
from test_replica_charge_transfer import (
    FLOOR_MARGIN, K_B, LACKS_RESOLUTION, TAU_STEPS, TAU_STEPS_OTHER, TIME_UNIT_CONVERSION, TOL_F,
    T_INITIAL, _displacement_force, _min_image, _parse_dump_columns, _read_restart_full, _rms,
)

pytestmark = pytest.mark.fast

GPUMD = str(GPUMD_EXECUTABLE)
DT_FS = 2.0  # fs; ZBL forces of a few eV/A move the atoms by ~1e-3 A in one step
MASS_AMU = {'Ba': 137.327, 'Ti': 47.867, 'O': 15.999}  # masses of the ordinary calibration runs
TOL_E_PER_ATOM = 1e-4  # eV/atom, DR-007 AC-5

CASES = [
    # label, fixture, ct, qnep, small-box
    ('qnep1 small-box', 'qnep_mode1_BaTiO3.txt', None, None, True),
    ('qnep1 large-box', 'qnep_mode1_BaTiO3.txt', None, None, False),
    ('qnep2 small-box', 'qnep_mode2_BaTiO3.txt', None, None, True),
    ('qnep2 large-box', 'qnep_mode2_BaTiO3.txt', None, None, False),
    ('qnep2 ct small-box', 'qnep_mode2_BaTiO3.txt', 4.0, None, True),
    ('qnep2 ct large-box', 'qnep_mode2_BaTiO3.txt', 4.0, None, False),
    ('qnep3 ct small-box', 'qnep_mode2_BaTiO3.txt', 4.0, 3, True),
    ('qnep3 ct large-box', 'qnep_mode2_BaTiO3.txt', 4.0, 3, False),
]
IDS = [c[0] for c in CASES]


def _model(fixture, zbl, ct, qnep, values=None):
    text = zt.model_text(zt.read_fixture(fixture), zbl, ct, qnep, values, zt.FD_CAPACITY)
    return zt.scale_nep(text, zt.FD_NEP_SCALE)


def _structure(small):
    return zt.batio3_cluster(None if small else zt.LARGE_L, a=zt.FD_A, cells=zt.FD_CELLS)


def _write_replica_input(d, text, sym, pos, lat, replicas=2, dt=DT_FS):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'nep.txt').write_text(text)
    zt.write_xyz(d / 'model.xyz', sym, pos, lat)
    # exchange 1000 > 1 step: no exchange, so no exchange rescaling of the velocities at step 1
    (d / 'run.in').write_text(
        f'potential nep.txt\ntime_step {dt:g}\nensemble nvt_bdp 1 2 {TAU_STEPS:g}\n'
        f'multi_replica remd replicas {replicas} exchange 1000 temp 1 2 equilibrate 0\n'
        'kspace ewald\nrun 1\n')


def _run_replica(d, expect_ok=True):
    r = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True, timeout=300)
    (d / 'run.log').write_text(r.stdout + r.stderr)
    if expect_ok:
        assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    return r


def _read_restart_energy(d, replica):
    line = (d / f'remd_replica_{replica}_restart.xyz').read_text().splitlines()[1]
    return float(line.split('potential_energy=')[1].split()[0])


def _write_model_with_velocity(path, sym, pos, lat, mass, vel):
    with open(path, 'w') as f:
        f.write('%d\n' % len(pos))
        f.write('Lattice="%s" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 pbc="T T T"\n'
                % ' '.join('%.15g' % x for x in np.asarray(lat).reshape(-1)))
        for s, p, m, v in zip(sym, pos, mass, vel):
            f.write('%s %.15f %.15f %.15f %.15g %.15g %.15g %.15g\n' % (s, *p, m, *v))


def _ordinary_step(d, text, sym, pos, lat, vel, mass, tau):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'nep.txt').write_text(text)
    _write_model_with_velocity(d / 'model.xyz', sym, pos, lat, mass, vel)
    (d / 'run.in').write_text(
        'potential nep.txt\nkspace ewald\n'
        f'ensemble nvt_bdp {T_INITIAL:g} {T_INITIAL:g} {tau:g}\n'
        f'time_step {DT_FS:g}\n'
        'dump_xyz 1 d.xyz precision double velocity force\nrun 1\n')
    r = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    cols = _parse_dump_columns(d / 'd.xyz', len(sym))
    return cols['pos'], cols['vel'], cols['forces']


def _floor(tmp_path, text, sym, pos, lat):
    """noise floor of the displacement method (ordinary gpumd, same model and step)."""
    mass = np.array([MASS_AMU[s] for s in sym])
    sigma = np.sqrt(K_B * T_INITIAL / mass) / TIME_UNIT_CONVERSION
    vel = np.random.default_rng(20261121).normal(size=(len(sym), 3)) * sigma[:, None]
    res = []
    for tag, tau in (('a', TAU_STEPS), ('b', TAU_STEPS), ('c', TAU_STEPS_OTHER)):
        x1, v1, f = _ordinary_step(tmp_path / ('cal_' + tag), text, sym, pos, lat, vel, mass, tau)
        res.append((_displacement_force(pos, x1, v1, mass, lat, DT_FS), f))
    resid = [_rms(dd - f) for dd, f in res]
    pair = _rms(res[0][0] - res[1][0])
    return {'floor': max(resid + [pair]), 'resid_rms': resid, 'pair_rms': pair,
            'rms_force': _rms(res[0][1])}


@pytest.mark.parametrize('label, fixture, ct, qnep, small', CASES, ids=IDS)
def test_replica_flexible_zbl_displacement_gate(tmp_path, label, fixture, ct, qnep, small):
    sym, pos, lat = _structure(small)
    text = _model(fixture, 'flex', ct, qnep, zt.pair_params(zt.FD_PAIR_RC))
    none = _model(fixture, 'none', ct, qnep)
    # self-calibration (ordinary gpumd)
    info = _floor(tmp_path, text, sym, pos, lat)
    assert info['rms_force'] > 0.05, (label, 'vacuous force', info)
    if info['floor'] > TOL_F / FLOOR_MARGIN:
        pytest.fail(f'{LACKS_RESOLUTION}: {label}: {info}')
    # replica run
    d = tmp_path / 'replica'
    _write_replica_input(d, text, sym, pos, lat)
    r = _run_replica(d)
    assert 'has the flexible ZBL potential' in r.stdout, 'flexible ZBL banner missing'
    assert ('charge_transfer' in r.stdout) == (ct is not None)
    n = len(sym)
    for replica in range(2):
        x1, mass, v1 = _read_restart_full(d, replica, n)
        assert np.all(np.isfinite(v1)) and _rms(v1) > 0.0
        ref = zt.run_static(tmp_path, text, sym, x1, lat)
        ref_none = zt.run_static(tmp_path, none, sym, x1, lat)
        zbl_f = _rms(ref['forces'] - ref_none['forces'])
        assert zbl_f > 1.0, (label, 'ZBL does not contribute clearly: RMS dF_zbl', zbl_f)
        d_rep = _displacement_force(pos, x1, v1, mass, lat, DT_FS)
        rms_df = _rms(d_rep - ref['forces'])
        de = abs(_read_restart_energy(d, replica) - ref['energy']) / n
        print(label, replica, 'RMS dF', rms_df, 'dE/N', de, 'floor', info['floor'],
              'RMS F', _rms(ref['forces']), 'RMS F_zbl', zbl_f,
              'max |x1-x0|', np.max(np.abs(x1 - pos)))
        assert rms_df <= TOL_F, (label, replica, 'RMS dF', rms_df, 'floor', info)
        assert de <= TOL_E_PER_ATOM, (label, replica, 'dE/N', de)


EQ_CASES = [
    ('qnep1', 'qnep_mode1_BaTiO3.txt', None, None),
    ('qnep2 ct', 'qnep_mode2_BaTiO3.txt', 4.0, None),
    ('qnep3 ct', 'qnep_mode2_BaTiO3.txt', 4.0, 3),
]


EQ_DT_FS = 1e-5  # fs: the atoms hardly move, so equal models give equal energies at the final positions
EQ_TOL_E_PER_ATOM = 1e-6  # eV/atom (observed <= 2e-8 on RTX A6000), 100x below the AC-5 gate


@pytest.mark.parametrize('small', [True, False], ids=['small-box', 'large-box'])
@pytest.mark.parametrize('label, fixture, ct, qnep', EQ_CASES, ids=[c[0] for c in EQ_CASES])
def test_replica_flexible_equals_universal(tmp_path, label, fixture, ct, qnep, small):
    # compressed rattled cluster (a = 2.6 A): the universal ZBL (0.75, 1.5) contributes ~100 eV.
    # Replica runs draw their own random initial velocities (differing from run to run), so equal
    # models are compared at (almost) the starting positions: dt = 1e-5 fs, displacement ~1e-9 A.
    sym, pos, lat = zt.batio3_cluster(None if small else zt.LARGE_L)
    energy = {}
    for kind in ('universal', 'flex', 'none'):
        d = tmp_path / kind
        _write_replica_input(d, _model(fixture, kind, ct, qnep), sym, pos, lat, dt=EQ_DT_FS)
        _run_replica(d)
        energy[kind] = [_read_restart_energy(d, k) for k in range(2)]
        x1, _, _ = _read_restart_full(d, 0, len(sym))
        assert np.max(np.abs(_min_image(x1 - pos, lat))) < 1e-6, 'atoms moved'
    for k in range(2):
        assert abs(energy['universal'][k] - energy['none'][k]) > 1.0, (label, 'ZBL does not contribute')
        de = abs(energy['flex'][k] - energy['universal'][k]) / len(sym)
        print(label, small, k, 'flex-universal dE/N', de)
        assert de <= EQ_TOL_E_PER_ATOM, (label, small, k, energy)
