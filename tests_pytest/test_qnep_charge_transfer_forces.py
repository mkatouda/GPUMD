"""charge_transfer qNEP: forces and virial vs finite differences (DR-005 stage 2c, VER-0033/0034).

Synthetic random-weight model (VER-0032 make_ct_model.py).  gpumd's own total potential energy of
static runs (time_step 0, kspace ewald, dump_xyz precision double) is differentiated by central
differences (delta 0.005 A for positions, strain 0.005 for the virial) and compared with the dumped
forces and total virial.  Cases: thin n12 / L = 7.5 with rc 4 / 4 (small-box kernels, thickness
<= 2.5 (rc + 1)) and n45 / L = 14 with rc 3.5 / 3.5 (L > 2.5 (3.5 + 1) = 11.25: large-box kernels).
Gates (VER-0034): force RMS error <= 1e-3 eV/A; virial RMS error <= 3e-3 |W|_RMS; both tests also
require |F|_RMS / |W|_RMS above a floor so that a vanishing signal cannot pass.  Mutations of the
kernel (drop Dt, drop the pair term, negate it) were checked to violate them (VER-0034).
"""
import sys

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPTS = REPO_ROOT / 'specs' / 'verification' / 'VER-0034-qnep-charge-transfer-forces-gpu' / 'scripts'
if not (SCRIPTS / 'ct_forces_fd.py').exists():
    pytest.skip('VER-0034 scripts not present', allow_module_level=True)
sys.path.insert(0, str(SCRIPTS))
import ct_forces_fd as fd  # noqa: E402

FORCE_GATE = 1e-3  # eV/A
VIRIAL_REL_GATE = 3e-3
DELTA = 0.005
# (structure, rc_radial, rc_phi, atoms for the force FD, label)
CASES = [
    ('thin_n12_L7.5', 4.0, 4.0, list(range(12)), 'small-box'),
    ('thick_n45_L14', 3.5, 3.5, [0, 17, 33], 'large-box'),
]


def _setup(structure, rc_radial, rc_phi, **kw):
    return fd.setup(structure, 1, False, rc_radial, rc_phi, **kw)


@pytest.mark.parametrize('structure, rc_radial, rc_phi, atoms, label', CASES)
def test_force_matches_finite_difference(tmp_path, structure, rc_radial, rc_phi, atoms, label):
    text, pos, types, lat, _ = _setup(structure, rc_radial, rc_phi)
    _, res = fd.force_fd(str(GPUMD_EXECUTABLE), str(tmp_path), text, pos, types, lat, DELTA, atoms, 'ewald')
    assert res['rms_F'] > 0.1, (label, res)
    assert res['rms_err'] <= FORCE_GATE, (label, res)
    assert res['max_err'] <= 3 * FORCE_GATE, (label, res)


@pytest.mark.parametrize('structure, rc_radial, rc_phi, atoms, label', CASES)
def test_virial_matches_finite_difference(tmp_path, structure, rc_radial, rc_phi, atoms, label):
    text, pos, types, lat, _ = _setup(structure, rc_radial, rc_phi)
    base = fd.run_static(str(GPUMD_EXECUTABLE), str(tmp_path), 'base', text, pos, types, lat)
    res = fd.virial_fd(str(GPUMD_EXECUTABLE), str(tmp_path), text, pos, types, lat, DELTA, 'ewald', base)
    assert res['rms_W'] > 1.0, (label, res)
    assert res['rms_err'] <= VIRIAL_REL_GATE * res['rms_W'], (label, res)


@pytest.mark.parametrize('structure, rc_radial, rc_phi, atoms, label', CASES)
def test_nve_energy_drift_smoke(tmp_path, structure, rc_radial, rc_phi, atoms, label):
    text, pos, types, lat, _ = _setup(structure, rc_radial, rc_phi)
    res = fd.nve_drift(str(GPUMD_EXECUTABLE), str(tmp_path), text, pos, types, lat, steps=600, dt_fs=0.5)
    assert abs(res['drift_fit']) <= 2e-3, (label, res)  # eV over 300 fs
    assert res['max_dev'] <= 5e-3, (label, res)
    assert res['kin_mean'] > 0.1  # the system moves


@pytest.mark.parametrize('structure, rc_radial, rc_phi, atoms, label', CASES)
def test_permutation_translation_determinism(tmp_path, structure, rc_radial, rc_phi, atoms, label):
    text, pos, types, lat, _ = _setup(structure, rc_radial, rc_phi)
    exe = str(GPUMD_EXECUTABLE)
    r1 = fd.run_static(exe, str(tmp_path), 'r1', text, pos, types, lat)
    r2 = fd.run_static(exe, str(tmp_path), 'r2', text, pos, types, lat)
    assert np.array_equal(r1['forces'], r2['forces'])  # run-to-run determinism of a static run
    assert r1['energy'] == r2['energy']
    perm = np.random.default_rng(3).permutation(len(pos))
    rp = fd.run_static(exe, str(tmp_path), 'perm', text, pos[perm], types[perm], lat)
    assert np.max(np.abs(rp['forces'] - r1['forces'][perm])) <= 1e-5
    assert abs(rp['energy'] - r1['energy']) <= 1e-5
    shift = np.array([0.37, -1.21, 2.05])
    rt = fd.run_static(exe, str(tmp_path), 'shift', text, pos + shift, types, lat)
    assert np.max(np.abs(rt['forces'] - r1['forces'])) <= 1e-4
    assert abs(rt['energy'] - r1['energy']) <= 1e-4
    assert np.max(np.abs(r1['forces'].sum(axis=0))) <= 1e-5  # momentum conservation


# extended cases (DR-005 stage 2c review): charge_mode 3, triclinic cells, three types, pppm.
# (label, setup kwargs, kspace, atoms)
EXTENDED = [
    ('mode3 small-box', 'thin_n12_L7.5', 4.0, dict(charge_mode=3), 'ewald', [0, 4, 8, 11]),
    ('mode3 large-box', 'thick_n45_L14', 3.5, dict(charge_mode=3), 'ewald', [0, 17, 33]),
    ('triclinic small-box', 'thin_n12_L7.5', 4.0, dict(triclinic=True), 'ewald', [0, 4, 8, 11]),
    ('triclinic large-box', 'thick_n45_L14', 3.5, dict(triclinic=True), 'ewald', [0, 17, 33]),
    ('3 types small-box', 'thin_n12_L7.5', 4.0, dict(nt=3), 'ewald', [0, 4, 8, 11]),
    ('pppm large-box', 'thick_n45_L14', 3.5, {}, 'pppm', [0, 17, 33]),
]


@pytest.mark.parametrize('label, structure, rc, kw, kspace, atoms', EXTENDED)
def test_extended_force_and_virial_fd(tmp_path, label, structure, rc, kw, kspace, atoms):
    text, pos, types, lat, _ = _setup(structure, rc, rc, **kw)
    exe = str(GPUMD_EXECUTABLE)
    base, res = fd.force_fd(exe, str(tmp_path), text, pos, types, lat, DELTA, atoms, kspace)
    assert res['rms_F'] > 0.1, (label, res)
    assert res['rms_err'] <= FORCE_GATE, (label, res)
    vres = fd.virial_fd(exe, str(tmp_path), text, pos, types, lat, DELTA, kspace, base)
    assert vres['rms_W'] > 1.0, (label, vres)
    assert vres['rms_err'] <= VIRIAL_REL_GATE * vres['rms_W'], (label, vres)
