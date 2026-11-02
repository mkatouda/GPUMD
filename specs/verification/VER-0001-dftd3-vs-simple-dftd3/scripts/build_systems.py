import numpy as np
from ase.build import molecule
from ase import Atoms
from ase.lattice.hexagonal import Graphite
from ase.io import write
rng = np.random.default_rng(12345)

# 1) non-periodic benzene dimer (stacked, 3.7 A)
b1 = molecule('C6H6'); b2 = molecule('C6H6'); b2.translate([0, 0, 3.7]); b2.rotate(30, 'z')
bz = b1 + b2; bz.center(vacuum=20.0); bz.pbc = False
write('benzene_dimer.xyz', bz)

# 2) periodic rattled graphite 3x3x1 (36 atoms)
g = Graphite('C', latticeconstant={'a': 2.46, 'c': 6.70}, size=(3, 3, 1))
g.rattle(0.05, seed=1); g.pbc = True
write('graphite.xyz', g)

# 3) periodic water box, 27 randomly rotated H2O on a 3.1 A grid (81 atoms)
atoms = Atoms(cell=[9.3] * 3, pbc=True)
for i in range(3):
    for j in range(3):
        for k in range(3):
            w = molecule('H2O')
            w.rotate(rng.uniform(0, 360), rng.normal(size=3))
            w.translate(np.array([i, j, k]) * 3.1 + 1.0)
            atoms += w
atoms.rattle(0.03, seed=2)
write('water27.xyz', atoms)
for n in ['benzene_dimer', 'graphite', 'water27']:
    print(n)

# 4) triclinic (sheared) copy of the water box
tri = atoms.copy()
newcell = np.array([[9.3, 0, 0], [2.5, 9.0, 0], [-1.5, 2.0, 8.8]])
tri.set_cell(newcell, scale_atoms=True)
write('water27_tri.xyz', tri)
