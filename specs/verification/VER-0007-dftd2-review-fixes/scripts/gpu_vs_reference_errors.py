"""Relative errors of the GPU D2 part against the NumPy reference for all GPU cases of
tests_pytest/test_dftd2_reference.py (s6 = 75), including the degenerate cases.

usage: gpu_vs_reference_errors.py <scratch dir>   (gpumd is taken from src/gpumd via conftest)
Prints energy, virial (relative to |E_ref|, max|W_ref|) and force (relative to max|F_ref|) errors,
to be compared with the caps 2e-6, 2e-6, 1e-5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tests_pytest'))
import numpy as np
import dftd2_reference as ref
import test_dftd2_reference as t

scratch = Path(sys.argv[1])
cases = {n: (lambda n=n: t.structure(n), v[1]) for n, v in t.GPU_CASES.items()}
cases.update({n: (b, rc) for n, (b, rc, _) in t.GPU_DERIVED_CASES.items()})
cases.update({n: (b, rc) for n, (b, rc) in t.DEGENERATE_CASES.items()})
for name, (build, rc) in cases.items():
    atoms = build()
    E, F, W = ref.energy_forces_virial(atoms, t.S6_GPU, rc)
    e0, f0, v0 = t._run(scratch / name / 'off', atoms, None)
    e1, f1, v1 = t._run(scratch / name / 'on', atoms, f'dftd2 {t.S6_GPU} {rc}')
    rel = lambda err, scale: err / scale if scale > 0 else (0.0 if err == 0 else float('inf'))
    print(f'{name:32s} E {rel(abs(e1 - e0 - E), abs(E)):.2e}  W {rel(np.abs(v1 - v0 - W).max(), np.abs(W).max()):.2e}'
          f'  F {rel(np.abs(f1 - f0 - F).max(), np.abs(F).max()):.2e}', flush=True)
