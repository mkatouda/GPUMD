"""Write the VER-0012 test systems (extxyz) into OUTDIR.  usage: make_systems.py REPO_ROOT OUTDIR
Water systems are repeats of tests_pytest/fixtures/dftd3/water27.xyz (81 atoms, 9.3 A cube).
  w81, w648 (2x2x2, small box), clus972 (2x2x3, non-periodic), w41472 (8x8x8, large box),
  scan_<N> : repeats (a,b,c) with b = 3 so one thickness stays <= 2.5 rc = 30 A (small-box path for rc = 12);
  no thickness may exceed 10 rc = 120 A (x: 12 x 9.3 = 111.6 A),
  dense systems for the neighbor-list length check: cu864 (fcc Cu 6x6x6), c1728 (diamond C 6x6x6), w648 at rc 20.
"""
import sys
from pathlib import Path
from ase.io import read, write
from ase.build import bulk

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
w = read(repo / 'tests_pytest/fixtures/dftd3/water27.xyz')
for name, rep, pbc in [('w81', (1, 1, 1), True), ('w648', (2, 2, 2), True), ('clus972', (2, 2, 3), False),
                       ('w41472', (8, 8, 8), True)]:
    a = w.repeat(rep)
    a.pbc = pbc
    write(out / f'{name}.xyz', a, format='extxyz')
for rep in [(1, 1, 2), (1, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3), (4, 3, 3), (6, 3, 3), (8, 3, 3), (12, 3, 3), (12, 3, 6), (12, 3, 12)]:
    a = w.repeat(rep)
    write(out / f'scan_{len(a)}.xyz', a, format='extxyz')
write(out / 'cu864.xyz', bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((6, 6, 6)), format='extxyz')
write(out / 'c1728.xyz', bulk('C', 'diamond', a=3.567, cubic=True).repeat((6, 6, 6)), format='extxyz')
