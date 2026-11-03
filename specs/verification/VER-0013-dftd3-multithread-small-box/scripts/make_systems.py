"""Systems of VER-0013 = those of VER-0012 (make_systems.py, run first into the same OUTDIR) plus
w5184 (water27 x (4,4,4), large box) and cu4000 (fcc Cu 10x10x10, a = 3.61 A, large box).
usage: make_systems.py REPO_ROOT OUTDIR"""
import sys
from pathlib import Path
from ase.io import read, write
from ase.build import bulk

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
w = read(repo / 'tests_pytest/fixtures/dftd3/water27.xyz')
write(out / 'w5184.xyz', w.repeat((4, 4, 4)), format='extxyz')
write(out / 'cu4000.xyz', bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((10, 10, 10)), format='extxyz')
