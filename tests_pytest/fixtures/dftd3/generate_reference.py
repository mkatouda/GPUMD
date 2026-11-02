"""Generate the structures and independent D3 reference values for test_dftd3_reference.py.

Reference: simple-dftd3 (conda-forge `simple-dftd3` + `dftd3-python`), PBE with rational (BJ)
damping, two-body only (the ATM three-body term is off), double precision, real-space cutoffs
set to the same values GPUMD is given (pair cutoff = `dftd3` potential_cutoff, CN cutoff =
coordination_number_cutoff). It is NOT part of the pytest run; run it only to regenerate the
fixtures, in an environment with `ase`, `numpy`, and `dftd3`:

    python generate_reference.py            # writes *.xyz and reference.json here

Stored values use GPUMD conventions: energy in eV, forces in eV/A, virial = -sigma (3x3, eV) --
the sign was fixed by agreement with GPUMD in specs/verification/VER-0001 and is checked there.
Do not edit the numbers by hand and do not regenerate them to make a failing test pass.
"""
import json
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write
from ase.lattice.hexagonal import Graphite
from dftd3 import __version__ as DFTD3_VERSION
from dftd3.interface import DispersionModel, RationalDampingParam

HA, BOHR = 27.211386245988, 0.529177210903
rng = np.random.default_rng(12345)


def water_box():
    atoms = Atoms(cell=[9.3] * 3, pbc=True)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                w = molecule('H2O')
                w.rotate(rng.uniform(0, 360), rng.normal(size=3))
                w.translate(np.array([i, j, k]) * 3.1 + 1.0)
                atoms += w
    atoms.rattle(0.03, seed=2)
    return atoms


def build():
    cases = {}
    b1, b2 = molecule('C6H6'), molecule('C6H6')
    b2.translate([0, 0, 3.7]); b2.rotate(30, 'z')
    bz = b1 + b2; bz.center(vacuum=20.0); bz.pbc = False
    cases['benzene_dimer'] = (bz, (12, 6), 'non-periodic')
    g = Graphite('C', latticeconstant={'a': 2.46, 'c': 6.70}, size=(3, 3, 1))
    g.rattle(0.05, seed=1); g.pbc = True
    cases['graphite'] = (g, (12, 6), 'small-box path, hexagonal cell')
    w = water_box()
    cases['water27'] = (w, (12, 6), 'small-box path, cubic cell')
    cases['water27_rc25'] = (w.copy(), (25, 15), 'small-box path, large cutoffs')
    tri = w.copy()
    tri.set_cell(np.array([[9.3, 0, 0], [2.5, 9.0, 0], [-1.5, 2.0, 8.8]]), scale_atoms=True)
    cases['water27_tri'] = (tri, (12, 6), 'small-box path, triclinic cell')
    wl = w.repeat((3, 3, 3)); wl.rattle(0.05, seed=3)
    cases['water_large'] = (wl, (8, 4), 'large-box path (27.9 A > 2.5 x 8 A)')
    cu = bulk('Cu', 'fcc', a=3.61, cubic=True).repeat((7, 7, 7)); cu.rattle(0.05, seed=4)
    cases['cu_large'] = (cu, (8, 4), 'large-box path, Cu (many reference CNs)')
    return cases


def reference(atoms, cutoffs):
    pbc = atoms.pbc.any()
    m = DispersionModel(atoms.numbers, atoms.positions / BOHR,
                        atoms.cell.array / BOHR if pbc else None, atoms.pbc if pbc else None)
    m.set_realspace_cutoff(cutoffs[0] / BOHR, cutoffs[0] / BOHR, cutoffs[1] / BOHR)
    r = m.get_dispersion(RationalDampingParam(method='pbe'), grad=True)
    return (float(r['energy'] * HA), (-r['gradient'] * HA / BOHR),
            (-r['virial'] * HA))


if __name__ == '__main__':
    out = {'_provenance': {
        'reference': 'simple-dftd3 via dftd3-python', 'dftd3_python_version': DFTD3_VERSION,
        'functional': 'pbe', 'damping': 'rational (BJ), two-body only (no ATM)',
        'units': 'eV, eV/A; virial = -sigma (GPUMD convention)'}}
    for name, (atoms, cut, note) in build().items():
        write(f'{name}.xyz', atoms)
        e, f, v = reference(atoms, cut)
        out[name] = dict(natoms=len(atoms), potential_cutoff=cut[0], cn_cutoff=cut[1], note=note,
                         energy=e, forces=np.round(f, 12).tolist(), virial=np.round(v, 12).tolist())
        print(name, len(atoms), cut, e)
    json.dump(out, open('reference.json', 'w'))
