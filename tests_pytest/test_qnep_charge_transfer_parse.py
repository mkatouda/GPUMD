"""Parser tests for the qNEP `charge_transfer` nep.txt line (DR-005 stage 2b, ADR-0006).

Synthetic models only (random weights from VER-0032 make_ct_model.py). A valid charge_transfer
model parses, prints a summary and the stage-2b warning (forces are not valid yet), and runs;
invalid combinations stop with the documented input errors.
"""
import importlib.util
import os
import subprocess

import pytest

from conftest import GPUMD_EXECUTABLE, REPO_ROOT

pytestmark = pytest.mark.fast

SCRIPT = (
    REPO_ROOT / 'specs' / 'verification' / 'VER-0032-qnep-charge-transfer-gpu' / 'scripts'
    / 'make_ct_model.py'
)
if not SCRIPT.exists():
    pytest.skip('VER-0032 make_ct_model.py not present', allow_module_level=True)
_spec = importlib.util.spec_from_file_location('make_ct_model', SCRIPT)
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


def _run(
    tmp_path, model_text, structure=('thick_n8_L13', 8, 13.0), run_lines=None
):
    name, natoms, box = structure
    pos, types = mk.make_structure(natoms, box, 7)
    mk.write_extxyz(tmp_path / 'model.xyz', pos, types, box)
    (tmp_path / 'nep.txt').write_text(model_text)
    (tmp_path / 'run.in').write_text(
        run_lines or 'potential nep.txt\nensemble nve\ntime_step 1\nrun 1\n'
    )
    return subprocess.run(
        [str(GPUMD_EXECUTABLE)], cwd=tmp_path, capture_output=True, text=True, check=False
    )


def _model(**kw):
    return mk.make_model_text(1, kw.get('rc_radial', 4.0), 3.5, kw.get('rc_phi', 4.0),
                              kw.get('with_ct', True))


def _out(r):
    return r.stdout + r.stderr


def test_nve_run_works_without_any_opt_in(tmp_path):
    # stage 2c: forces/virial implemented, the stage-2b refusal and its environment variable are gone
    run = 'potential nep.txt\nensemble nve\ntime_step 0.5\ndump_thermo 1\nrun 5\n'
    r = _run(tmp_path, _model(), run_lines=run)
    assert r.returncode == 0, _out(r)[-500:]
    assert 'NOT valid' not in _out(r)
    assert (tmp_path / 'thermo.out').stat().st_size > 0


def test_static_run_works(tmp_path):
    run = 'potential nep.txt\nensemble nve\ntime_step 0\nrun 1\n'
    r = _run(tmp_path, _model(), run_lines=run)
    assert r.returncode == 0, _out(r)[-500:]


def test_model_without_line_runs(tmp_path):
    r = _run(tmp_path, _model(with_ct=False))
    assert r.returncode == 0, _out(r)[-500:]


def test_valid_model_parses_and_runs(tmp_path):
    r = _run(tmp_path, _model())
    assert r.returncode == 0, _out(r)[-500:]
    assert 'charge_transfer: rc_phi = 4 A, 3 symmetric type pairs x 5 phi coefficients' in _out(r)


def test_rc_phi_smaller_than_radial_is_accepted(tmp_path):
    r = _run(tmp_path, _model(rc_phi=3.0))
    assert r.returncode == 0 and 'rc_phi = 3 A' in _out(r)


def test_rc_phi_larger_than_radial_is_error(tmp_path):
    r = _run(tmp_path, _model(rc_phi=4.5))
    assert r.returncode != 0
    assert 'rc_phi must not exceed the radial cutoff' in _out(r)


@pytest.mark.parametrize('value', ['0', '-1'])
def test_rc_phi_nonpositive_is_error(tmp_path, value):
    r = _run(tmp_path, _model().replace('charge_transfer 4', f'charge_transfer {value}'))
    assert r.returncode != 0 and 'rc_phi must be positive' in _out(r)


def test_missing_rc_phi_value_is_error(tmp_path):
    r = _run(tmp_path, _model().replace('charge_transfer 4\n', 'charge_transfer\n'))
    assert r.returncode != 0 and 'This line should be charge_transfer rc_phi' in _out(r)


def test_charge_mode_1_is_error(tmp_path):
    # the check fires at the charge_transfer line, before any parameter is read
    text = _model().replace('nep4_charge2', 'nep4_charge1', 1)
    r = _run(tmp_path, text)
    assert r.returncode != 0
    assert 'only for charge_mode 2 and 3' in _out(r)


def test_too_few_phi_coefficients_is_error(tmp_path):
    lines = _model().rstrip('\n').split('\n')
    r = _run(tmp_path, '\n'.join(lines[:-2]) + '\n')
    assert r.returncode != 0
    assert 'expected 15 phi coefficients' in _out(r)


def test_too_many_phi_coefficients_is_error(tmp_path):
    r = _run(tmp_path, _model() + '0.5\n')
    assert r.returncode != 0
    assert 'parameter-count mismatch' in _out(r)


@pytest.mark.parametrize('structure', mk.STRUCTURES)
def test_model_without_line_still_runs(tmp_path, structure):
    """The same synthetic model without the line and phi block is an ordinary qNEP model."""
    r = _run(tmp_path, _model(with_ct=False), structure)
    assert r.returncode == 0, _out(r)[-500:]
    assert 'charge_transfer' not in _out(r)
