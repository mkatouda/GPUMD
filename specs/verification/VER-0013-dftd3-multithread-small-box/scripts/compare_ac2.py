"""AC-2 comparison of the `dftd3_*` regression cases (DR-001 AC-2 wording): candidate vs baseline on
thermo.out and restart.xyz with rtol 1e-5, atol 1e-12 (|a - b| <= atol + rtol |b|); non-numeric tokens
must be identical.  usage: compare_ac2.py WORKDIR   (WORKDIR = tests_regression/.work/<run>, from a
`run_regression.py --suite full --case 'dftd3_*' --keep-all` run).  Prints the largest abs/rel difference."""
import re, sys
from pathlib import Path

RTOL, ATOL = 1e-5, 1e-12
NUM = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')


def compare(a, b):
    ta, tb = a.read_text().split(), b.read_text().split()
    if len(ta) != len(tb):
        return False, 0.0, 0.0, 'token count differs'
    worst_abs = worst_rel = 0.0
    ok = True
    for x, y in zip(ta, tb):
        if NUM.match(x) and NUM.match(y):
            fx, fy = float(x), float(y)
            d = abs(fx - fy)
            worst_abs = max(worst_abs, d)
            if fy != 0:
                worst_rel = max(worst_rel, d / abs(fy))
            if d > ATOL + RTOL * abs(fy):
                ok = False
        elif x != y:
            return False, 0.0, 0.0, f'token {x!r} != {y!r}'
    return ok, worst_abs, worst_rel, ''


work = Path(sys.argv[1])
allok = True
for case in sorted((work / 'cases').iterdir()):
    for name in ('thermo.out', 'restart.xyz'):
        a, b = case / 'candidate' / name, case / 'baseline' / name
        if not b.exists():
            continue
        ok, da, dr, msg = compare(a, b)
        allok &= ok
        print(f'{case.name:24s} {name:12s} {"PASS" if ok else "FAIL"}  max abs diff {da:.3e}  max rel diff {dr:.3e} {msg}')
print('AC-2 (rtol 1e-5, atol 1e-12):', 'PASS' if allok else 'FAIL')
sys.exit(0 if allok else 1)
