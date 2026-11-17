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

FORCE gate (human decision 2026-11-18 (a): displacement method, no src change; (b): a test-only
src dump hook only if this method cannot reach the resolution): RMS dF <= 1e-4 eV/A, see
`test_replica_force_by_displacement`. VIRIAL gate (RMS dW / RMS W <= 1e-4): not verified (human
decision c; it stays open, it is not retired), because neither restart files nor remd_thermo.out
carry the replica virial and a displacement does not give it. The gpumd reference force and virial
RMS are still required to be non-vacuous in the energy test.

Displacement method (derivation). Replica dynamics (src/integrate/replica) draws its own initial
velocities from the seed and ignores `vel` in model.xyz (replica_remd.cu initialize_slot calls
Velocity::initialize_cpu(false, ...)), so identical initial velocities for replica and gpumd are
impossible and v0 is unknown. The identity below does not need v0. One velocity-Verlet step is
  u := (x1 - x0)/dt = v0 + a0 dt/2           (kick, drift; thermostat has not acted yet)
  v1 = f * (u + a1 dt/2),   a1 = F(x1)/m      (second half kick, then the BDP factor f)
so  F(x1) = 2 m (v1/f - u)/dt  exactly, and the test uses f = 1:  dF := 2 m C^2 (v1 - u)/dt - F_ref(x1),
with x1, v1, m read from the restart file (17 significant digits), x0 from model.xyz and F_ref(x1)
from ordinary gpumd (static, same nep.txt, forces at 17 digits). It tests the replica force at x1
(the second half kick); the force at x0 enters only through x1 and cannot be separated without v0.
Unit factor: positions in A, v in A/fs (restart/dump), m in amu, F in eV/A. Natural units have the
time unit T0 = TIME_UNIT_CONVERSION = 10.18051 fs (src/utilities/common.cuh:26; a_nat = F/m in
A/T0^2), so a [A/fs^2] = (F/m)/C^2 and F = m a C^2 with C = 10.18051 (C^2 = 103.6426 eV fs^2/(A^2
amu), i.e. 1/C^2 = 9.6485e-3 as the usual eV/amu -> (A/fs)^2). Hence the factor 2 m C^2/dt above.

BDP and the noise bound. In src/integrate/replica/ensemble_bdp_replica.cu and ensemble_bdp.cu the
thermostat acts only after the second half kick: f^2 = K'/K with
K' = K + (1-c)(sigma (S+R^2)/n - K) + 2 R sqrt(K sigma (1-c) c / n), c = exp(-1/tau), tau in steps,
sigma = n k T0 /2, R ~ N(0,1), S ~ chi^2(n-1), n = 3N (ordinary) or 3N - removed dof (replica).
The positions after one step are independent of the thermostat; only v1 carries f, so
|dF| <= |f - 1| (|F| + 2 m C^2 |u|/dt) with the sampled f unknown. With eps = 1-c ~ 1/tau and
r = sigma/K <= 2: |f - 1| <= 1/2 [eps |r (1 + k sqrt(2/n)) - 1| + 2 k sqrt(r eps/n)] at k sigma of the
noise (the test uses k = 6, n = 3N - 3). Choice: tau = 1e12 steps -> eps = 1e-12, sqrt(r eps/n) ~ 1e-7
for N = 45, so |f - 1| <~ 1e-6 and the velocity term is <~ 1e-6 * (|F| + 2 m C^2 |u|/dt)
~ 1e-6 eV/A (|F| ~ 0.5, initial velocities at 1-2 K, dt = 10 fs; the second term is
2 m C^2 |v0|/dt ~ 0.1 for m = 25 amu). Initial velocities must be nonzero (BDP divides by the
kinetic energy) and are small only to keep that second term small.
Float and rounding terms: positions/velocities of the restart/dump are doubles printed with 17
digits (relative 1e-16); x1 - x0 has an absolute error <~ 4e-15 A, giving 2 m C^2 * 4e-15/dt^2
~ 2e-11 eV/A at dt = 10 fs. The forces themselves are evaluated in float arithmetic (relative
~1e-7, so ~1e-7 eV/A at |F| ~ 1): this is the quantity under test, and it is also the
irreducible floor of the replica-versus-gpumd comparison. EXPECTED RESOLUTION (analytic):
~1e-6 eV/A from the velocity term, ~1e-7 from float forces, both far below the 1e-4 gate.
THESE NUMBERS ARE NOT MEASURED YET. They must be measured on the GPU; the self-calibration below
(`_displacement_floor`) does so on every run.

Self-calibration (resolution of the method). Ordinary gpumd runs the same one-step nvt_bdp
(vel in model.xyz, dump_xyz precision double velocity force) three times: tau = 1e12 twice and
tau = 1e14. The same displacement residual with gpumd's own dumped force, RMS(D - F_dump), and the
difference of D between the two equal-tau runs, RMS(D_a - D_b), give the noise floor of the method.
The replica gate is evaluated only if floor <= gate/10 = 1e-5 eV/A; otherwise the test FAILS with
'displacement method lacks resolution; hook needed (human decision b)'. A passing replica test
therefore means RMS dF <= 1e-4 with a demonstrated resolution 10 times better than the gate.

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
        ('missing', ('\n'.join(lines[:-1]) + '\n').replace('@ phi_parameters 15', '@ phi_parameters 14'), "block 'phi_parameters' requires 15 values, found 14"),
        ('extra', (text + '0.1\n').replace('@ phi_parameters 15', '@ phi_parameters 16'), "block 'phi_parameters' requires 15 values, found 16"),
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


# ---------------------------------------------------------------------------------------------
# Force gate by the displacement method (derivation and expected resolution: module docstring)
# ---------------------------------------------------------------------------------------------
TIME_UNIT_CONVERSION = 1.018051e+1  # src/utilities/common.cuh:26 (natural time unit in fs)
K_B = 8.617343e-5  # src/utilities/common.cuh:22 (eV/K)
TOL_F = 1e-4  # eV/A, RMS dF (human-decided 2026-11-11, unchanged)
FLOOR_MARGIN = 10.0  # the replica gate is evaluated only if floor <= TOL_F / FLOOR_MARGIN
DT_FS = 10.0  # fs; large enough that the position change is dominated by the force kick
TAU_STEPS = 1e12  # BDP tau in steps (noise bound in the module docstring)
TAU_STEPS_OTHER = 1e14  # second tau of the self-calibration
T_INITIAL = 1.0  # K, initial velocity temperature of the ordinary calibration runs
MASS_AMU = {'Na': 22.98977, 'Cl': 35.453, 'Mg': 24.305}  # explicit masses for the ordinary runs
LACKS_RESOLUTION = 'displacement method lacks resolution; hook needed (human decision b)'

_FLOOR_CACHE = {}


def _f_bound(n_dof, tau, r=2.0, k=6.0):
    """Bound of |f - 1| of the BDP velocity factor at k sigma of the noise (module docstring)."""
    eps = -np.expm1(-1.0 / tau)
    return 0.5 * (eps * abs(r * (1.0 + k * np.sqrt(2.0 / n_dof)) - 1.0) + 2.0 * k * np.sqrt(r * eps / n_dof))


def _write_model_with_velocity(path, pos, types, lat, vel):
    with open(path, 'w') as f:
        f.write('%d\n' % len(pos))
        f.write(
            'Lattice="%s" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 pbc="T T T"\n'
            % ' '.join('%.15g' % x for x in np.asarray(lat).reshape(-1))
        )
        for p, t, v in zip(pos, types, vel):
            sym = mk.SYMBOLS[t]
            f.write('%s %.15f %.15f %.15f %.15g %.15g %.15g %.15g\n' % (sym, *p, MASS_AMU[sym], *v))


def _read_model_positions(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(x) for x in row.split()[1:4]] for row in lines[2:2 + n]])


def _parse_dump_columns(path, natoms):
    lines = path.read_text().splitlines()
    assert len(lines) == natoms + 2, (len(lines), natoms, 'expected exactly one dumped frame')
    props = lines[1].split('Properties=')[1].split()[0].split(':')
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    rows = [row.split() for row in lines[2:]]
    return {
        name: np.array([[float(x) for x in row[a:a + w]] for row in rows])
        for name, (a, w) in col.items() if name != 'species'
    }


def _min_image(dx, lat):
    s = dx @ np.linalg.inv(np.asarray(lat))
    return (s - np.round(s)) @ np.asarray(lat)


def _displacement_force(x0, x1, v1, mass, lat, dt_fs=DT_FS):
    """D = 2 m C^2 (v1 - u) / dt (eV/A): the force at x1 implied by one velocity-Verlet step (f = 1)."""
    u = _min_image(x1 - x0, lat) / dt_fs  # A/fs
    return 2.0 * mass[:, None] * TIME_UNIT_CONVERSION ** 2 * (v1 - u) / dt_fs


def _f_hat(x0, x1, v1, mass, lat, f_ref, dt_fs=DT_FS):
    """least-squares estimate of the BDP factor f in v1 = f (u + a1 dt/2), a diagnostic only."""
    u = _min_image(x1 - x0, lat) / dt_fs
    w = u + 0.5 * f_ref / (mass[:, None] * TIME_UNIT_CONVERSION ** 2) * dt_fs
    return float(np.sum(v1 * w) / np.sum(w * w))


def _ordinary_step(directory, text, pos, types, lat, vel, tau):
    """one nvt_bdp step with ordinary gpumd; returns x0, x1, v1 (A/fs), F(x1), masses."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'nep.txt').write_text(text)
    _write_model_with_velocity(directory / 'model.xyz', pos, types, lat, vel)
    (directory / 'run.in').write_text(
        'potential nep.txt\nkspace ewald\n'
        f'ensemble nvt_bdp {T_INITIAL:g} {T_INITIAL:g} {tau:g}\n'
        f'time_step {DT_FS:g}\n'
        'dump_xyz 1 d.xyz precision double velocity force\nrun 1\n'
    )
    r = subprocess.run([GPUMD], cwd=directory, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    cols = _parse_dump_columns(directory / 'd.xyz', len(types))
    mass = np.array([MASS_AMU[mk.SYMBOLS[t]] for t in types])
    return _read_model_positions(directory / 'model.xyz'), cols['pos'], cols['vel'], cols['forces'], mass


def _displacement_floor(tmp_path_factory, structure, rc):
    """noise floor (eV/A) of the displacement method from ordinary gpumd runs, cached per structure.

    floor = max( RMS(D_i - F_dump_i) for the three runs, RMS(D_a - D_b) of the two equal-tau runs ).
    """
    key = (structure, rc)
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    text, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc)
    mass = np.array([MASS_AMU[mk.SYMBOLS[t]] for t in types])
    # nonzero Gaussian velocities at T_INITIAL, identical in all three runs (A/fs)
    sigma = np.sqrt(K_B * T_INITIAL / mass) / TIME_UNIT_CONVERSION
    vel = np.random.default_rng(20261118).normal(size=(len(types), 3)) * sigma[:, None]
    base = tmp_path_factory.mktemp('ordinary_floor')
    results = []
    for tag, tau in (('a', TAU_STEPS), ('b', TAU_STEPS), ('c', TAU_STEPS_OTHER)):
        x0, x1, v1, f_dump, m = _ordinary_step(base / tag, text, pos, types, lat, vel, tau)
        results.append((_displacement_force(x0, x1, v1, m, lat), f_dump, _f_hat(x0, x1, v1, m, lat, f_dump)))
    resid = [_rms(d - f) for d, f, _ in results]
    pair = _rms(results[0][0] - results[1][0])
    floor = max(resid + [pair])
    info = {
        'floor': floor,
        'resid_rms': resid,
        'pair_rms': pair,
        'f_hat_minus_1': [fh - 1.0 for _, _, fh in results],
        'f_bound': float(_f_bound(3 * len(types) - 3, TAU_STEPS)),
        'rms_force': _rms(results[0][1]),
    }
    _FLOOR_CACHE[key] = info
    return info


def _require_resolution(info, label):
    assert info['rms_force'] > 0.05, (label, 'vacuous: ordinary gpumd force RMS', info['rms_force'])
    if info['floor'] > TOL_F / FLOOR_MARGIN:
        pytest.fail(f'{LACKS_RESOLUTION}: {label}: {info}')


def _write_replica_displacement_input(directory, text, pos, types, lat, replicas):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'nep.txt').write_text(text)
    fd.write_xyz(str(directory / 'model.xyz'), pos, types, lat)
    # exchange 1000 > 1 step: no exchange, so no exchange rescaling of the velocities at step 1
    (directory / 'run.in').write_text(
        f'potential nep.txt\ntime_step {DT_FS:g}\nensemble nvt_bdp 1 2 {TAU_STEPS:g}\n'
        f'multi_replica remd replicas {replicas} exchange 1000 temp 1 2 equilibrate 0\n'
        'kspace ewald\nrun 1\n'
    )


def _read_restart_full(directory, replica, natoms):
    """positions (A), masses (amu) and velocities (A/fs) of the restart file, 17 digits."""
    lines = (directory / f'remd_replica_{replica}_restart.xyz').read_text().splitlines()
    assert len(lines) == natoms + 2, (len(lines), natoms)
    fields = dict(field.split('=', 1) for field in shlex.split(lines[1]))
    assert int(fields['restart_step']) == 1 and int(fields['replica']) == replica
    assert 'vel:R:3' in lines[1]
    rows = [row.split() for row in lines[2:]]
    pos = np.array([[float(x) for x in row[1:4]] for row in rows])
    mass = np.array([float(row[4]) for row in rows])
    vel = np.array([[float(x) for x in row[5:8]] for row in rows])
    return pos, mass, vel


@pytest.mark.parametrize('structure, rc, label', CASES)
def test_replica_force_by_displacement(tmp_path, tmp_path_factory, structure, rc, label):
    # The numbers of the resolution estimate are analytic (module docstring) and still have to be
    # measured on the GPU; the floor below is measured on every run before the gate is evaluated.
    info = _displacement_floor(tmp_path_factory, structure, rc)
    _require_resolution(info, label)
    text, pos, types, lat, _ = fd.setup(structure, 1, False, rc, rc)
    d = tmp_path / 'replica'
    _write_replica_displacement_input(d, text, pos, types, lat, 2)
    r = _run_replica(d, 2)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'charge_transfer' in r.stdout, 'charge_transfer banner missing: model not recognised'
    exchanges = [row for row in (d / 'remd_exchange.out').read_text().splitlines() if not row.startswith('#')]
    assert not exchanges, ('an exchange happened at step 1; velocities are rescaled', exchanges)
    x0 = _read_model_positions(d / 'model.xyz')
    n = len(types)
    for replica in range(2):
        x1, mass, v1 = _read_restart_full(d, replica, n)
        assert np.all(np.isfinite(v1)) and _rms(v1) > 0.0, 'zero velocity: BDP degenerate'
        f_ref, _, _ = _gpumd_reference(tmp_path, f'ref_{replica}', text, x1, types, lat)
        assert _rms(f_ref) > 0.05, (label, 'vacuous: gpumd force RMS', _rms(f_ref))
        d_rep = _displacement_force(x0, x1, v1, mass, lat)
        rms_df = _rms(d_rep - f_ref)
        assert rms_df <= TOL_F, (
            label, replica, 'RMS dF', rms_df, 'gate', TOL_F, 'floor', info,
            'f_hat-1', _f_hat(x0, x1, v1, mass, lat, f_ref),
        )
    # Virial gate (RMS dW / RMS W <= 1e-4): not verified (human decision c).


def test_displacement_method_resolution_ordinary(tmp_path_factory):
    # the self-calibration alone (ordinary gpumd only): fails with LACKS_RESOLUTION if the floor is
    # not at least FLOOR_MARGIN below the force gate
    structure, rc, label = CASES[1]
    _require_resolution(_displacement_floor(tmp_path_factory, structure, rc), label)


@pytest.mark.parametrize(
    'keyword_line',
    ['add_efield 0 0 0 0.1 0 bec', 'compute_dpdt 1', 'dump_exyz 1 precision double'],
)
def test_replica_unsupported_keyword_is_rejected(tmp_path, keyword_line):
    # parse_run_input() in src/integrate/replica/replica.cu exits with
    # PRINT_INPUT_ERROR("A multi_replica run supports potential, time_step, multi_replica, ensemble, "
    # "kspace, dump_xyz, and run commands only.") for any other keyword. main() calls
    # print_gpu_information() first, so a visible GPU is needed to reach it, but the error comes
    # before reading model.xyz and before any kernel (no replica output is created).
    text, pos, types, lat, _ = fd.setup('thin_n12_L7.5', 1, False, 4.0, 4.0)
    d = tmp_path / 'badkw'
    _write_replica_input(d, text, pos, types, lat, 2)
    lines = (d / 'run.in').read_text().splitlines()
    assert lines[-1].startswith('run')
    (d / 'run.in').write_text('\n'.join(lines[:-1] + [keyword_line, lines[-1]]) + '\n')
    r = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert (
        'A multi_replica run supports potential, time_step, multi_replica, ensemble, kspace, '
        'dump_xyz, and run commands only.'
    ) in r.stderr, r.stderr[-400:]
    assert 'Started initializing positions' not in r.stdout
    assert not list(d.glob('remd_*')), 'replica output exists: the error came after the run started'
