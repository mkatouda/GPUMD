"""Guard: flexible ZBL (`zbl 0 0` + per-pair parameters) stays accepted for qnep1/2/3 (DR-007).

Upstream GPUMD accepts flexible ZBL only for `model nep` (upstream commits dbecce90 and 4508bba2
added the restrictions).  This fork lifts them for the charge models qnep1/qnep2/qnep3, with and
without `charge_transfer`.  Each test below is a cheap smoke test of one place where the
restriction would reappear.  After an upstream merge, a failure here means that the upstream
restriction came back in that place; re-apply the fork change instead of changing the test.

  a. gpumd reader / validation of the model file     (test_gpumd_accepts_flexible_zbl_qnep)
  b. `nep` trainer (finish_reading)                  (test_nep_trainer_accepts_flexible_zbl_qnep)
  c. nep_model_converter.py (old-format models)      (test_converter_accepts_flexible_zbl_qnep)
  d. replica dynamics (`multi_replica`)              (test_replica_accepts_flexible_zbl_qnep)

The helpers are those of the detailed DR-007 tests (test_qnep_flexible_zbl.py,
test_nep_flexible_zbl_train.py, test_nep_model_converter_flexible_zbl.py); this file only checks
acceptance, not accuracy.  It does not need specs/.  Needs one GPU.
"""
import subprocess

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE

import test_nep_flexible_zbl_train as tt
import test_nep_model_converter_flexible_zbl as cv
import test_qnep_flexible_zbl as zt

pytestmark = pytest.mark.fast

# label, fixture, charge_transfer rc_phi, model number (None: as in the fixture)
MODELS = [
    ('qnep1', 'qnep_mode1_BaTiO3.txt', None, None),
    ('qnep2', 'qnep_mode2_BaTiO3.txt', None, None),
    ('qnep2-ct', 'qnep_mode2_BaTiO3.txt', 4.0, None),
    ('qnep3-ct', 'qnep_mode2_BaTiO3.txt', 4.0, 3),
]
IDS = [m[0] for m in MODELS]


@pytest.mark.parametrize('label, fixture, ct, qnep', MODELS, ids=IDS)
def test_gpumd_accepts_flexible_zbl_qnep(tmp_path, label, fixture, ct, qnep):
    """Guards the gpumd model reader: a new-format qnep1/2/3 model with `zbl 0 0` and a
    `zbl_parameters` block (also with `charge_transfer`) must load, and the ZBL term must act."""
    base = zt.read_fixture(fixture)
    sym, pos, lat = zt.batio3_cluster()
    flex = zt.run_static(tmp_path, zt.model_text(base, 'flex', ct, qnep), sym, pos, lat)
    none = zt.run_static(tmp_path, zt.model_text(base, 'none', ct, qnep), sym, pos, lat)
    assert np.isfinite(flex['energy']) and np.all(np.isfinite(flex['forces']))
    assert abs(flex['energy'] - none['energy']) > 1.0, (label, flex['energy'], none['energy'])


@pytest.mark.parametrize('charge_mode, ct', [(2, False), (3, True)], ids=['qnep2', 'qnep3-ct'])
def test_nep_trainer_accepts_flexible_zbl_qnep(tmp_path, charge_mode, ct):
    """Guards the `nep` trainer: `model qnep2/3` (with or without `charge_transfer`) with
    `zbl <cutoff>` and a zbl.in file must train and write `zbl 0 0` plus `zbl_parameters`."""
    structs = tt.make_structures()[:1]
    tt.write_train_xyz(tmp_path / 'train.xyz', structs)
    tt.write_nep_in(tmp_path / 'nep.in', charge_mode, ct, 'population 10\ngeneration 1\nseed 1\n'
                    'batch 100000\noutput_interval 1\n')
    tt.write_zbl_in(tmp_path / 'zbl.in', tt.zbl_values())
    r = tt.run_nep(tmp_path)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    text = (tmp_path / 'nep.txt').read_text()
    assert '\nzbl 0 0\n' in text and '@ zbl_parameters 30' in text


@pytest.mark.parametrize('label, fixture, ct, qnep', MODELS[1:], ids=IDS[1:])
def test_converter_accepts_flexible_zbl_qnep(tmp_path, label, fixture, ct, qnep):
    """Guards nep_model_converter.py: an old-format `nep4_zbl_charge1/2/3` file with `zbl 0 0`
    (also with `charge_transfer`) must convert to a new-format qnep model with zbl_parameters."""
    mode = qnep or int(fixture.split('mode')[1][0])
    legacy, _ = cv.to_legacy(cv.new_model(fixture, ct, mode), mode)
    r, dst = cv.convert(tmp_path, legacy)
    assert r.returncode == 0, r.stderr
    text = dst.read_text()
    assert '\nmodel qnep%d\n' % mode in text
    assert '\nzbl 0 0\n' in text and '@ zbl_parameters 60' in text


@pytest.mark.parametrize('label, fixture, ct, qnep', MODELS[1:3], ids=IDS[1:3])
def test_replica_accepts_flexible_zbl_qnep(tmp_path, label, fixture, ct, qnep):
    """Guards replica dynamics: `multi_replica` with a flexible-ZBL qnep model must run (no
    refusal) and give a finite energy."""
    sym, pos, lat = zt.batio3_cluster(None, a=zt.FD_A, cells=zt.FD_CELLS)
    text = zt.model_text(zt.read_fixture(fixture), 'flex', ct, qnep, zt.pair_params(zt.FD_PAIR_RC),
                         zt.FD_CAPACITY)
    text = zt.scale_nep(text, zt.FD_NEP_SCALE)
    (tmp_path / 'nep.txt').write_text(text)
    zt.write_xyz(tmp_path / 'model.xyz', sym, pos, lat)
    (tmp_path / 'run.in').write_text(
        'potential nep.txt\ntime_step 2\nensemble nvt_bdp 1 2 1e12\n'
        'multi_replica remd replicas 2 exchange 1000 temp 1 2 equilibrate 0\nkspace ewald\nrun 1\n')
    r = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=tmp_path, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    line = (tmp_path / 'remd_replica_0_restart.xyz').read_text().splitlines()[1]
    assert np.isfinite(float(line.split('potential_energy=')[1].split()[0]))
