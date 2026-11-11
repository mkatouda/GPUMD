"""charge_transfer qNEP in the NEP trainer (DR-005 stage 3a, VER-0035).

Synthetic data only. Tests: (1) A4: `nep` prediction vs gpumd static runs on the same new-format nep.txt
(BEC <= 5e-6 e RMS, forces <= 1e-5 eV/A RMS, energy <= 1e-5 eV/atom), modes qnep2 and qnep3;
(2) the trainer's own forces/virial against finite differences of the trainer's energy (oracle independent
of gpumd); (3) input errors of the new keyword; (4) gpumd species compaction of a charge_transfer nep.txt;
(5) a short SNES training smoke test (loss decreases, the written nep.txt is read by gpumd and by `nep`,
exact restart check).
"""
import subprocess
import sys

import pytest

from conftest import GPUMD_EXECUTABLE, NEP_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPTS = (
    REPO_ROOT / 'specs' / 'verification' / 'VER-0035-nep-charge-transfer-trainer' / 'scripts'
)
if not (SCRIPTS / 'ct_a4_compare.py').exists() or not (
    REPO_ROOT / 'specs' / 'verification' / 'VER-0032-qnep-charge-transfer-gpu' / 'scripts'
    / 'make_ct_model.py'
).exists():
    pytest.skip('VER-0032/VER-0035 generator scripts not present', allow_module_level=True)
sys.path.insert(0, str(SCRIPTS))
import ct_a4_compare as a4  # noqa: E402
import ct_compact_check as compact  # noqa: E402
import ct_fd_trainer as fdt  # noqa: E402
import ct_train_smoke as smoke  # noqa: E402
import make_ct_trainer_inputs as mt  # noqa: E402

BEC_GATE = 5e-6  # e (DR-005 A4 is 1e-5; tightened after review)
FORCE_GATE = 1e-5  # eV/A
ENERGY_GATE = 1e-5  # eV/atom
# a trained model has |F| ~ 26 eV/A (random teacher data): float noise scales with |F|, so the force
# gate there is RELATIVE to the force RMS (human decision 2026-11-11; measured ~5e-7), BEC and energy keep the
# absolute gates above
TRAINED_FORCE_REL_GATE = 1e-6


def _a4(tmp_path, charge_mode, structures, nt=2, rc_phi=4.0):
    mk = mt.load_mk()
    idx = [int(s) for s in structures.split(',')]
    structs = mt.make_structures(mk, idx, nt)
    nd = tmp_path / 'nep'
    nd.mkdir()
    mt.write_model(str(nd / 'nep.txt'), mk, 1, 4.0, 3.5, rc_phi, True, nt, charge_mode)
    mt.write_nep_in(str(nd / 'nep.in'), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 1)
    mt.write_train_xyz(str(nd / 'train.xyz'), mk, structs)
    return a4.compare(str(NEP_EXECUTABLE), str(GPUMD_EXECUTABLE), str(nd), str(tmp_path), structs, idx, mk,
                      verbose=False)


@pytest.mark.parametrize(
    'charge_mode, structures, nt, rc_phi',
    [(2, '0,1,3,4', 2, 4.0), (3, '1,3', 2, 4.0), (2, '0,1,3,4', 3, 3.0), (2, '1,4', 3, 4.0)],
)
def test_nep_prediction_matches_gpumd(tmp_path, charge_mode, structures, nt, rc_phi):
    r = _a4(tmp_path, charge_mode, structures, nt, rc_phi)
    assert r['bec_rms'] <= BEC_GATE, r
    assert r['force_rms'] <= FORCE_GATE, r
    assert r['max_dE_per_atom'] <= ENERGY_GATE, r
    assert r['charge_rms'] <= 1e-5, r
    # the test must not be vacuous
    assert r['rms_bec'] > 0.1 and r['rms_force'] > 0.1, r


@pytest.mark.parametrize('structure, label', [(3, 'thin n12 L7.5'), (1, 'thick n45 L14')])
def test_trainer_forces_virial_vs_finite_difference(tmp_path, structure, label):
    """The trainer's own energy is differentiated (central FD, 12 atoms, six strain components) and
    compared with the trainer's forces and virial. Gates as in VER-0034: forces RMS <= 1e-3 eV/A and max
    <= 3e-3; virial RMS error <= 3x the error of the same weights without charge_transfer (old-model
    floor) and <= 3e-3 of the RMS virial."""
    ct = fdt.fd_check(str(NEP_EXECUTABLE), str(tmp_path / 'ct'), structure, 12, True)
    old = fdt.fd_check(str(NEP_EXECUTABLE), str(tmp_path / 'old'), structure, 12, False)
    assert ct['rms_F'] > 0.1 and ct['rms_W'] > 1.0, (label, ct)  # not vacuous
    assert ct['F_err_rms'] <= 1e-3 and ct['F_err_max'] <= 3e-3, (label, ct)
    assert ct['W_err_rms'] <= 3.0 * old['W_err_rms'], (label, ct, old)
    assert ct['W_err_rms'] <= 3e-3 * ct['rms_W'], (label, ct)


@pytest.mark.parametrize('fmt, gate', [('new', 1e-6), ('legacy', 1e-5)])
def test_gpumd_species_compaction_of_charge_transfer_file(tmp_path, fmt, gate):
    """3-type charge_transfer model on a 2-species structure (gpumd compacts the file) against the
    2-type model selected by numpy from the same arrays."""
    r = compact.check(str(GPUMD_EXECUTABLE), str(tmp_path), fmt)
    assert r['rms_Z'] > 0.05 and r['rms_F'] > 0.1, r
    assert max(r['dq'], r['dF'], r['dZ'], r['dE']) <= gate, r


def _run_nep(directory):
    return subprocess.run([str(NEP_EXECUTABLE)], cwd=directory, capture_output=True, text=True)


@pytest.mark.parametrize(
    'edit, message',
    [
        ('charge_transfer\n', 'should have 1 parameter'),
        ('charge_transfer 0\n', 'should be > 0'),
        ('charge_transfer -1\n', 'should be > 0'),
        ('charge_transfer 1e-50\n', 'should be > 0'),
        ('charge_transfer 4.5\n', 'should be <= the radial cutoff'),
        ('charge_transfer 3 3\n', 'should have 1 parameter'),
        ('charge_transfer 3\ncharge_transfer 3\n', 'only once'),
        ('charge_mode 1\ncharge_transfer 3\n', 'requires charge_mode 2 or 3'),
    ],
)
def test_charge_transfer_input_errors(tmp_path, edit, message):
    mk = mt.load_mk()
    mt.write_nep_in(str(tmp_path / 'nep.in'), mk, 4.0, 3.5, 4.0, False, 2, 2, 0, 'generation 1\n' + edit)
    if edit.startswith('charge_mode'):
        text = (tmp_path / 'nep.in').read_text().replace('model qnep2\n', '')
        (tmp_path / 'nep.in').write_text(text)
    mt.write_train_xyz(str(tmp_path / 'train.xyz'), mk, mt.make_structures(mk, [3], 2))
    r = _run_nep(tmp_path)
    assert r.returncode != 0
    assert message in r.stdout + r.stderr, (r.stdout + r.stderr)[-500:]


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', 'abc'])
def test_charge_transfer_rejects_non_finite_values(tmp_path, value):
    mk = mt.load_mk()
    mt.write_nep_in(str(tmp_path / 'nep.in'), mk, 4.0, 3.5, 4.0, False, 2, 2, 0,
                    'generation 1\ncharge_transfer %s\n' % value)
    mt.write_train_xyz(str(tmp_path / 'train.xyz'), mk, mt.make_structures(mk, [3], 2))
    r = _run_nep(tmp_path)
    assert r.returncode != 0
    assert 'charge_transfer' in r.stdout + r.stderr, (r.stdout + r.stderr)[-500:]


@pytest.mark.parametrize('other', ['model qnep3_vdw\n', 'model nep\n', 'model nep_vdw\n', 'model nep_dipole\n',
                                   'model_type 1\n'])
def test_charge_transfer_rejected_with_other_models(tmp_path, other):
    mk = mt.load_mk()
    lines = ['type 2 Na Cl', other.strip(), 'cutoff 4 3.5', 'n_max 3 2', 'basis_size 4 4', 'l_max 4 0 0',
             'neuron 10', 'charge_transfer 4', 'generation 1']
    (tmp_path / 'nep.in').write_text('\n'.join(lines) + '\n')
    mt.write_train_xyz(str(tmp_path / 'train.xyz'), mk, mt.make_structures(mk, [3], 2))
    r = _run_nep(tmp_path)
    assert r.returncode != 0
    text = r.stdout + r.stderr
    assert 'charge_transfer requires charge_mode 2 or 3' in text or 'cannot be combined with charge_vdw' in text, \
        text[-500:]


def test_charge_transfer_rejects_nonzero_total_charge_label(tmp_path):
    mk = mt.load_mk()
    mt.write_nep_in(str(tmp_path / 'nep.in'), mk, 4.0, 3.5, 4.0, True, 2, 2, 0, 'generation 1\n')
    mt.write_train_xyz(str(tmp_path / 'train.xyz'), mk, mt.make_structures(mk, [3], 2))
    lines = (tmp_path / 'train.xyz').read_text().split('\n')
    lines[1] += ' charge=0.5'
    (tmp_path / 'train.xyz').write_text('\n'.join(lines))
    r = _run_nep(tmp_path)
    assert r.returncode != 0
    assert 'charge_transfer' in r.stdout + r.stderr and 'charge=' in r.stdout + r.stderr, \
        (r.stdout + r.stderr)[-500:]
    # a zero label is accepted
    lines[1] = lines[1].replace(' charge=0.5', ' charge=0')
    (tmp_path / 'train.xyz').write_text('\n'.join(lines))
    assert _run_nep(tmp_path).returncode == 0


def test_training_smoke(tmp_path):
    r = smoke.run_smoke(str(NEP_EXECUTABLE), str(GPUMD_EXECUTABLE), str(tmp_path), generations=30)
    # loss: best never worse than the first evaluated generation, strictly lower at the end
    assert r['loss_min'] <= r['loss_first'], r
    assert r['loss_last'] < r['loss_first'], r
    # nep.txt written by the trainer is read by gpumd and agrees with nep prediction (A4 on a trained model)
    a = r['a4_trained']
    assert a['rms_force'] > 1.0, a            # the model really is a large-force one (relative gate is meaningful)
    assert a['bec_rms'] <= BEC_GATE and a['force_rms'] <= TRAINED_FORCE_REL_GATE * a['rms_force'], a
    assert a['max_dE_per_atom'] <= ENERGY_GATE, a
    # nep.txt round trip: prediction RMSE of the forces reproduces the training loss of the best member
    assert abs(r['force_rmse_pred'] - r['force_rmse_loss']) <= 1e-3 * max(1.0, r['force_rmse_loss']), r
    # exact restart check (all sigma set to 0 in nep.restart: every member equals mu, so nep.txt of the
    # restarted run must hold exactly the restart mu, including the 15 phi values at the tail)
    assert r['num_phi'] == 15, r
    assert r['sigma0_phi_maxdiff'] <= 1e-8, r
    assert r['sigma0_parameters_maxdiff'] <= 1e-8, r
    # secondary: normal runs are O(sigma) apart (mu is not the best member), and the restarted run starts
    # far below the fresh-start loss and close to the previous best
    assert r['restart_vs_txt_phi_maxdiff_normal'] <= 5.0 * r['sigma_phi_max'], r
    assert r['restart_first'] < 0.75 * r['loss_first'], r
    assert r['restart_first'] <= 1.25 * r['loss_last'], r
