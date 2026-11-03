"""Build the three test structures (ASE). usage: make_structures.py OUTDIR"""
import sys
import numpy as np
from ase import Atoms
from ase.build import make_supercell, bulk
from ase.lattice.hexagonal import Graphite
from ase.io import write

out = sys.argv[1]
rng = np.random.default_rng(12345)

# (a) water: 6x6x6 grid of molecules, random orientation, density 0.997 g/cc (216 molecules)
n = 6
L = (216 * 18.015 / 0.6022 / 0.997) ** (1 / 3)
h = L / n
oh, ang = 0.9572, np.deg2rad(104.52)
base = np.array([[0, 0, 0], [oh, 0, 0], [oh * np.cos(ang), oh * np.sin(ang), 0]])
base -= base[0]
pos, sym = [], []
for ix in range(n):
    for iy in range(n):
        for iz in range(n):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            a, b, c, d = q
            R = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                          [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                          [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
            cen = (np.array([ix, iy, iz]) + 0.5) * h
            m = (base - base.mean(0)) @ R.T + cen
            pos += list(m); sym += ['O', 'H', 'H']
write(f'{out}/water0.xyz', Atoms(sym, positions=pos, cell=[L]*3, pbc=True))

# (b) graphite AB, a=2.462 c=6.708 (literature), 9x9x3 cells of 4 atoms = 972 atoms
g = Graphite('C', latticeconstant={'a': 2.462, 'c': 6.708}, size=(9, 9, 3))
g.pbc = True
write(f'{out}/graphite_ideal.xyz', g)
g = g.copy(); g.rattle(stdev=0.05, seed=7)   # ideal lattice gives F=0 by symmetry
write(f'{out}/graphite.xyz', g)

# (c) MgO rock salt a=4.212, 4x4x4 conventional cells = 512 atoms
m = bulk('MgO', 'rocksalt', a=4.212, cubic=True).repeat((4, 4, 4))
write(f'{out}/mgo_ideal.xyz', m)
m = m.copy(); m.rattle(stdev=0.05, seed=8)
write(f'{out}/mgo.xyz', m)
print(len(g), len(m), L)
