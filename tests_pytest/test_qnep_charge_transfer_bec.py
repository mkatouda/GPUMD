"""charge_transfer qNEP: neutrality and BEC vs finite differences (DR-005 stage 2b, VER-0032).

Synthetic random-weight model (VER-0032 make_ct_model.py). gpumd `dump_xyz bec` / `charge` of static
runs (time_step 0, test-only opt-in GPUMD_CHARGE_TRANSFER_BEC_ONLY=1) against central finite
differences (delta 0.02 A) of the single-valued pair-form dipole P = sum_a h_a V_a built on the host
from the h of the same weights without the charge_transfer line (scripts/ct_bec_fd.py).
Gates: RMS(BEC/sqrt(eps_inf) - Z_FD) <= 1e-3 e (A1), relative RMS <= 1e-2, max abs <= 1e-3 e.
Box <= 2.5 (rc_radial + 1) = 12.5 A uses the small-box kernels, larger boxes the large-box kernels.
"""
import os
import sys

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPTS = (
    REPO_ROOT / 'specs' / 'verification' / 'VER-0032-qnep-charge-transfer-gpu' / 'scripts'
)
if not (SCRIPTS / 'ct_bec_fd.py').exists():
    pytest.skip('VER-0032 scripts not present', allow_module_level=True)
sys.path.insert(0, str(SCRIPTS))
import ct_bec_fd as fd  # noqa: E402
import make_ct_model as mk  # noqa: E402

ABS_RMS_GATE = 1e-3  # e, DR-005 A1
ABS_MAX_GATE = 1e-3  # e
REL_RMS_GATE = 1e-2
SUM_Q_GATE = 1e-6  # e (float charges)
THICK = 1  # mk.STRUCTURES index: n45, L14 (large-box kernels, RMS Z ~ 0.14)
THIN = 3  # n12, L7.5 (small-box kernels)


def _structure(index, nt=2):
    name, natoms, box = mk.STRUCTURES[index]
    pos, types = mk.make_structure(natoms, box, 100 + index, nt=nt)
    return pos, types, box


@pytest.mark.parametrize(
    'index, atoms, rc_phi, nt, label',
    [
        (THICK, [0, 9, 20, 33, 44], 4.0, 2, 'large-box'),
        (THIN, [0, 4, 9], 4.0, 2, 'small-box'),
        (THICK, [0, 9, 20, 33], 3.0, 2, 'large-box rc_phi < rc_radial'),
        (THICK, [1, 12, 25, 40], 4.0, 3, 'large-box 3 types'),
    ],
)
def test_bec_matches_finite_difference(tmp_path, index, atoms, rc_phi, nt, label):
    pos, types, box = _structure(index, nt)
    res = fd.compare(
        str(GPUMD_EXECUTABLE), str(tmp_path), 1, pos, types, box, 0.02, atoms, rc_phi=rc_phi, nt=nt
    )
    assert abs(res['sum_q_float']) <= SUM_Q_GATE, (label, res)
    assert abs(res['sum_q_double']) <= SUM_Q_GATE, (label, res)
    assert res['max_abs_q_gpu_minus_host'] <= 1e-5, (label, res)
    assert res['rms_bec_minus_fd_pair_P'] <= ABS_RMS_GATE, (label, res)
    assert res['max_abs_bec_minus_fd_pair_P'] <= ABS_MAX_GATE, (label, res)
    assert res['rms_Z'] > 0.02, (label, res)  # the test must not be vacuous
    assert res['rel_rms_bec_minus_fd'] <= REL_RMS_GATE, (label, res)


def _run_static(gpumd_dir, pos, types, box):
    return fd.run_gpumd(
        str(GPUMD_EXECUTABLE),
        str(gpumd_dir.parent),
        mk.make_model_text(1, 4.0, 3.5, 4.0, True),
        pos,
        types,
        box,
        gpumd_dir.name,
    )


@pytest.mark.parametrize('index', [THICK, THIN])
def test_atom_permutation_and_determinism(tmp_path, index):
    pos, types, box = _structure(index)
    q1, bec1 = _run_static(tmp_path / 'run1', pos, types, box)
    q2, bec2 = _run_static(tmp_path / 'run2', pos, types, box)
    assert np.array_equal(q1, q2)  # static run: charges bit-identical run to run
    perm = np.random.default_rng(5).permutation(len(pos))
    qp, becp = _run_static(tmp_path / 'perm', pos[perm], types[perm], box)
    assert np.max(np.abs(qp - q1[perm])) <= 1e-5
    assert np.max(np.abs(becp - bec1[perm])) <= 1e-5
