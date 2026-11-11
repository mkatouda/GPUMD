"""charge_transfer qNEP in gpumd_replica: energy, force and virial against gpumd.

Verification document: specs/verification/VER-0036-qnep-charge-transfer-replica/README.md.

gpumd_replica writes no forces by itself; the testing aid GPUMD_REPLICA_STATE_DUMP=<prefix> writes
<prefix>_replica_<id>.bin (raw doubles: position 3N, velocity 3N, force 3N, potential N, virial 9N,
component-major) at the end of the run. The test runs a very short REMD run (time_step 0.001 fs,
1 step, replicas never share state), takes the final positions of every replica from the dump and
evaluates the same structure and nep.txt with gpumd (static run, time_step 0). Each replica is
therefore compared with a gpumd evaluation of its own configuration; a single-replica run is not
possible in REMD (replicas > 1), so the 2-replica and 4-replica runs are the multi-stream cases.

Gates (human-approved 2026-11-11): |dE| / N <= 1e-4 eV/atom, RMS dF <= 1e-4 eV/A,
RMS dW / RMS W <= 1e-4 (W = the six independent components of the total virial).
Skipped when the VER-0032 generator, gpumd or gpumd_replica is missing. Needs one GPU.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

VER32 = REPO_ROOT / 'specs' / 'verification' / 'VER-0032-qnep-charge-transfer-gpu' / 'scripts'
VER34 = REPO_ROOT / 'specs' / 'verification' / 'VER-0034-qnep-charge-transfer-forces-gpu' / 'scripts'
if not (VER32 / 'make_ct_model.py').exists() or not (VER34 / 'ct_forces_fd.py').exists():
    pytest.skip('VER-0032/VER-0034 generator scripts not present', allow_module_level=True)
sys.path.insert(0, str(VER34))
sys.path.insert(0, str(VER32))
import ct_forces_fd as fd  # noqa: E402
import make_ct_model as mk  # noqa: E402

REPLICA = os.environ.get('GPUMD_REPLICA', str(REPO_ROOT / 'src' / 'gpumd_replica'))
GPUMD = str(GPUMD_EXECUTABLE)
if not os.path.exists(REPLICA):
    pytest.skip('gpumd_replica not built', allow_module_level=True)
if not os.path.exists(GPUMD):
    pytest.skip('gpumd not built', allow_module_level=True)

TOL_E_PER_ATOM = 1e-4  # eV/atom
TOL_F_RMS = 1e-4  # eV/A
TOL_W_REL = 1e-4  # of RMS W

# (structure, rc_radial = rc_phi, label); the thick cell with rc 3.5 uses the large-box kernels,
# the thin cell (box < 2.5 rc) the small-box kernels
CASES = [
    ('thick_n45_L14', 3.5, 'large-box'),
    ('thin_n12_L7.5', 4.0, 'small-box'),
]


def _write_replica_input(directory, text, pos, types, lat, replicas, steps=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'nep.txt').write_text(text)
    fd.write_xyz(str(directory / 'model.xyz'), pos, types, lat)
    (directory / 'run.in').write_text(
        'potential nep.txt\ntime_step 0.001\nensemble nvt_bdp 300 400 100\n'
        f'multi_replica remd replicas {replicas} exchange 1 temp 300 400 equilibrate 0\n'
        f'kspace ewald\nrun {steps}\n'
    )


def _run_replica(directory, replicas):
    env = dict(os.environ)
    env['GPUMD_REPLICA_STATE_DUMP'] = str(directory / 'state')
    r = subprocess.run([REPLICA], cwd=directory, capture_output=True, text=True, timeout=300, env=env)
    (directory / 'run.log').write_text(r.stdout + r.stderr)
    return r


def _read_state(directory, replica, natoms):
    data = np.fromfile(directory / f'state_replica_{replica}.bin', dtype=np.float64)
    assert data.size == 19 * natoms, (data.size, natoms)
    n = natoms
    pos = data[:3 * n].reshape(3, n).T.copy()
    force = data[6 * n:9 * n].reshape(3, n).T.copy()
    energy = float(np.sum(data[9 * n:10 * n]))
    v = data[10 * n:19 * n].reshape(9, n).sum(axis=1)  # components 0..5: xx yy zz xy xz yz
    virial = np.array([v[0], v[1], v[2], v[3], v[4], v[5]])
    return pos, force, energy, virial


def _gpumd_reference(tmp_path, tag, text, pos, types, lat):
    res = fd.run_static(GPUMD, str(tmp_path), tag, text, pos, types, lat, 'ewald', props='force')
    w = res['virial']
    virial = np.array([w[0, 0], w[1, 1], w[2, 2], w[0, 1], w[0, 2], w[1, 2]])
    return res['forces'], res['energy'], virial


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _check_replica_vs_gpumd(tmp_path, directory, text, types, lat, replica, label):
    n = len(types)
    pos, force, energy, virial = _read_state(directory, replica, n)
    f_ref, e_ref, w_ref = _gpumd_reference(tmp_path, f'ref_{replica}', text, pos, types, lat)
    assert _rms(f_ref) > 0.05, (label, 'vacuous: gpumd force RMS', _rms(f_ref))
    assert _rms(w_ref) > 0.5, (label, 'vacuous: gpumd virial RMS', _rms(w_ref))
    assert abs(energy - e_ref) / n <= TOL_E_PER_ATOM, (label, replica, energy, e_ref)
    assert _rms(force - f_ref) <= TOL_F_RMS, (label, replica, _rms(force - f_ref))
    assert _rms(virial - w_ref) / _rms(w_ref) <= TOL_W_REL, (label, replica, _rms(virial - w_ref) / _rms(w_ref))


@pytest.mark.parametrize('structure, rc, label', CASES)
@pytest.mark.parametrize('replicas', [2, 4])
def test_replica_matches_gpumd(tmp_path, structure, rc, label, replicas):
    if replicas == 4 and label != 'small-box':
        pytest.skip('4 replicas only on the thin cell (GPU time)')
    text, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc)
    d = tmp_path / 'replica'
    _write_replica_input(d, text, pos, types, lat, replicas)
    r = _run_replica(d, replicas)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'charge_transfer' in r.stdout, 'charge_transfer banner missing: model not recognised'
    for replica in range(replicas):
        _check_replica_vs_gpumd(tmp_path, d, text, types, lat, replica, f'{label} R={replicas}')


def test_replica_triclinic_large_and_small_box(tmp_path):
    for structure, rc, label in CASES:
        text, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc, triclinic=True)
        d = tmp_path / f'tri_{label}'
        _write_replica_input(d, text, pos, types, lat, 2)
        r = _run_replica(d, 2)
        assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
        for replica in range(2):
            _check_replica_vs_gpumd(tmp_path / label, d, text, types, lat, replica, f'triclinic {label}')


def test_replica_charge_mode3_is_rejected_cleanly(tmp_path):
    # gpumd_replica supports charge_mode 1 and 2 only (no nep4_charge3), also with charge_transfer
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, False, 4.0, 4.0, charge_mode=3)
    d = tmp_path / 'mode3'
    _write_replica_input(d, text, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode != 0
    assert 'charge3' in (r.stdout + r.stderr) or 'unsupported' in (r.stdout + r.stderr).lower()


def test_replica_without_charge_transfer_unchanged_path(tmp_path):
    # The same generator without the charge_transfer line (ordinary qNEP): old code path, must
    # still run and agree with gpumd (the old kernels are untouched; this guards the branches).
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, True, 4.0, 4.0)
    d = tmp_path / 'old'
    _write_replica_input(d, text, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'charge_transfer' not in r.stdout
    pos0, force, energy, virial = _read_state(d, 0, len(types))
    f_ref, e_ref, w_ref = _gpumd_reference(tmp_path, 'ref_old', text, pos0, types, lat)
    # the old kernels of the two codes are not guaranteed to agree to the ct gates (float noise of
    # the unchanged kernels is larger); this only guards against gross breakage
    assert abs(energy - e_ref) / len(types) <= 1e-3
    assert _rms(force - f_ref) <= 5e-3


def test_replica_rejects_malformed_phi_block(tmp_path):
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, False, 4.0, 4.0)
    lines = text.rstrip('\n').split('\n')
    for label, bad in (
        ('missing', '\n'.join(lines[:-1]) + '\n'),
        ('extra', text + '0.1\n'),
    ):
        d = tmp_path / label
        _write_replica_input(d, bad, pos, types, lat, 2)
        r = _run_replica(d, 2)
        assert r.returncode != 0, label
        assert 'charge_transfer' in (r.stdout + r.stderr), (label, (r.stdout + r.stderr)[-400:])
    # rc_phi above the radial cutoff
    d = tmp_path / 'rcphi'
    bad = text.replace('charge_transfer 4', 'charge_transfer 4.5')
    assert bad != text
    _write_replica_input(d, bad, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode != 0
    assert 'rc_phi must not exceed' in (r.stdout + r.stderr)
