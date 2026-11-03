"""Run-to-run variation of the finite-difference errors of tests_pytest/test_dftd2_consistency.py.

usage: fd_variation.py <gpumd executable> <scratch dir> <n runs>
Reuses the helpers of the test module (same systems, steps and 5-point stencils); prints, per run,
max|F_fd - F_gpu| / max|F_D2| for cubic and open, and the virial analogue for cubic and triclinic,
to be compared with the tolerance 2e-4.
"""
import os
import sys
from pathlib import Path

exe, scratch, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
os.environ['DFTD2_TEST_GPUMD'] = exe
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'tests_pytest'))
import numpy as np
import test_dftd2_consistency as t

for run in range(n):
    out = []
    for name in ('cubic', 'open'):
        atoms = t._systems()[name]
        r = t._Runner(scratch / f'run{run}_{name}_F')
        _, F, _ = r.d2(atoms)
        comps = [(k, a) for k in (0, 1, 40) for a in range(3)]
        jobs = [((lambda s, k=k, a=a: t._displaced(atoms, k, a, s)), t.H_FORCE) for k, a in comps]
        fd = [-d for d in t._derivative_jobs(r, jobs)]
        err = max(abs(f - F[k, a]) for f, (k, a) in zip(fd, comps))
        out.append(f'F[{name}] {err / np.abs(F).max():.3e}')
    for name, comps in (('cubic', [(0, 0), (1, 2)]), ('triclinic', [(1, 1), (0, 1)])):
        atoms = t._systems()[name]
        r = t._Runner(scratch / f'run{run}_{name}_W')
        _, _, W = r.d2(atoms)
        jobs = [((lambda s, a=a, b=b: t._strained(atoms, a, b, s)), t.H_STRAIN) for a, b in comps]
        fd = [-d for d in t._derivative_jobs(r, jobs)]
        err = max(abs(w - W[a, b]) for w, (a, b) in zip(fd, comps))
        out.append(f'W[{name}] {err / np.abs(W).max():.3e}')
    print(f'run {run}: ' + '  '.join(out), flush=True)
