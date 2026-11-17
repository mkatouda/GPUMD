"""charge_transfer qNEP in replica dynamics (multi_replica in gpumd): energy against ordinary gpumd.

Verification document: specs/verification/VER-0036-qnep-charge-transfer-replica/README.md.

Replica dynamics no longer has a separate executable or the GPUMD_REPLICA_STATE_DUMP testing aid.
The test runs a very short REMD run (time_step 0.001 fs, 1 step, replicas never share state),
reads the final positions and the potential energy of every replica from its restart file
(`remd_replica_<id>_restart.xyz`: positions and potential_energy with 17 significant digits) and
evaluates the same structure and nep.txt with ordinary gpumd (static run, time_step 0). Each
replica is therefore compared with a gpumd evaluation of its own configuration; a single-replica
run is not possible in REMD (replicas > 1), so the 2-replica and 4-replica runs are the
multi-stream cases.

Gate (human-approved 2026-11-11, unchanged): |dE| / N <= 1e-4 eV/atom.
NOT CHECKED HERE: forces and virial of the replica code. The restart files and remd_thermo.out
carry only positions, velocities and the potential energy, so the former force and virial gates
(RMS dF <= 1e-4 eV/A, RMS dW / RMS W <= 1e-4) cannot be evaluated without a source hook
(open item reported to the owner). The gpumd reference force and virial RMS are still required to
be non-vacuous, so that the energy comparison is made on a configuration with real forces.
Skipped when the VER-0032 generator or gpumd is missing. Needs one GPU.
"""
import os
import shlex
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

GPUMD = str(GPUMD_EXECUTABLE)
if not os.path.exists(GPUMD):
    pytest.skip('gpumd not built', allow_module_level=True)

TOL_E_PER_ATOM = 1e-4  # eV/atom

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
    r = subprocess.run([GPUMD], cwd=directory, capture_output=True, text=True, timeout=300)
    (directory / 'run.log').write_text(r.stdout + r.stderr)
    return r


def _read_state(directory, replica, natoms):
    """positions (N, 3) and potential energy (eV) of a replica at the end of the run."""
    lines = (directory / f'remd_replica_{replica}_restart.xyz').read_text().splitlines()
    assert len(lines) == natoms + 2, (len(lines), natoms)
    fields = dict(field.split('=', 1) for field in shlex.split(lines[1]))
    assert int(fields['restart_step']) == 1 and int(fields['replica']) == replica
    pos = np.array([[float(x) for x in row.split()[1:4]] for row in lines[2:]])
    return pos, float(fields['potential_energy'])


def _gpumd_reference(tmp_path, tag, text, pos, types, lat):
    res = fd.run_static(GPUMD, str(tmp_path), tag, text, pos, types, lat, 'ewald', props='force')
    w = res['virial']
    virial = np.array([w[0, 0], w[1, 1], w[2, 2], w[0, 1], w[0, 2], w[1, 2]])
    return res['forces'], res['energy'], virial


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _check_replica_vs_gpumd(tmp_path, directory, text, types, lat, replica, label):
    n = len(types)
    pos, energy = _read_state(directory, replica, n)
    f_ref, e_ref, w_ref = _gpumd_reference(tmp_path, f'ref_{replica}', text, pos, types, lat)
    assert _rms(f_ref) > 0.05, (label, 'vacuous: gpumd force RMS', _rms(f_ref))
    assert _rms(w_ref) > 0.5, (label, 'vacuous: gpumd virial RMS', _rms(w_ref))
    assert abs(energy - e_ref) / n <= TOL_E_PER_ATOM, (label, replica, energy, e_ref)


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


def test_replica_charge_mode3_matches_gpumd(tmp_path):
    # replica dynamics reads qnep1/qnep2/qnep3 through the common model reader (the former
    # separate replica executable refused charge mode 3, so this replaces the old rejection test)
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, False, 4.0, 4.0, charge_mode=3)
    d = tmp_path / 'mode3'
    _write_replica_input(d, text, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'charge_transfer' in r.stdout, 'charge_transfer banner missing: model not recognised'
    for replica in range(2):
        _check_replica_vs_gpumd(tmp_path, d, text, types, lat, replica, 'mode3')


def test_replica_without_charge_transfer_unchanged_path(tmp_path):
    # The same generator without the charge_transfer field (ordinary qNEP): old code path, must
    # still run and agree with gpumd (the old kernels are untouched; this guards the branches).
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, True, 4.0, 4.0)
    d = tmp_path / 'old'
    _write_replica_input(d, text, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'charge_transfer' not in r.stdout
    pos0, energy = _read_state(d, 0, len(types))
    f_ref, e_ref, w_ref = _gpumd_reference(tmp_path, 'ref_old', text, pos0, types, lat)
    # the old kernels of the two codes are not guaranteed to agree to the ct gates (float noise of
    # the unchanged kernels is larger); this only guards against gross breakage
    assert abs(energy - e_ref) / len(types) <= 1e-3


def test_replica_rejects_malformed_phi_block(tmp_path):
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, False, 4.0, 4.0)
    lines = text.rstrip('\n').split('\n')
    for label, bad, message in (
        ('missing', '\n'.join(lines[:-1]) + '\n', "block 'phi_parameters' requires 15 values, found 14"),
        ('extra', text + '0.1\n', "block 'phi_parameters' requires 15 values, found 16"),
    ):
        d = tmp_path / label
        _write_replica_input(d, bad, pos, types, lat, 2)
        r = _run_replica(d, 2)
        assert r.returncode != 0, label
        assert message in (r.stdout + r.stderr), (label, (r.stdout + r.stderr)[-400:])
    # rc_phi above the radial cutoff
    d = tmp_path / 'rcphi'
    bad = text.replace('charge_transfer 4', 'charge_transfer 4.5')
    assert bad != text
    _write_replica_input(d, bad, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode != 0
    assert 'charge_transfer requires 0 < rc_phi <= the radial cutoff' in (r.stdout + r.stderr)
