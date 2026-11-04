"""S1 diagnostic: error of the D3 contribution versus the largest (approximate) coordination number, for non-periodic
H (and Cu/H) clusters of 70 atoms in spheres of decreasing radius (min pair distance 0.6 A).  The approximate CN uses the D3
counting function with the D3 covalent radii (H 0.32 A, Cu 1.20 A? only for orientation), the reference is simple-dftd3.
usage: cn_scan.py --make-reference --work DIR   (env with dftd3)   /   cn_scan.py BINARY... --work DIR   (env with ase)"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write, read
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extreme_cn as X

RCOV = {'H': 0.32, 'Cu': 1.32}   # Pyykko-like radii used only for the orientation CN (not used by any code under test)


def make(R, el, n=70, seed=5):
    rng = np.random.default_rng(seed)
    pos = []
    tries = 0
    while len(pos) < n and tries < 2000000:
        tries += 1
        x = rng.uniform(-R, R, 3)
        if np.linalg.norm(x) < R and all(np.linalg.norm(x - y) > 0.6 for y in pos):
            pos.append(x)
    a = Atoms([el] * len(pos), positions=np.array(pos)); a.center(vacuum=20.0); a.pbc = False
    return a


def approx_cn(a):
    d = a.get_all_distances(); s = a.get_chemical_symbols()
    rc = np.array([[ (4/3)*(RCOV[x]+RCOV[y]) for y in s] for x in s])
    with np.errstate(over='ignore'):
        c = 1/(1+np.exp(-16*(rc/np.where(d > 0, d, 1) - 1)))
    np.fill_diagonal(c, 0); c[d > 6.0] = 0
    return c.sum(1)


SPECS = [(R, el) for el in ('H', 'Cu') for R in (6.0, 4.0, 3.0, 2.4, 2.0, 1.7)]
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('binaries', nargs='*'); ap.add_argument('--work', required=True)
    ap.add_argument('--make-reference', action='store_true'); a = ap.parse_args()
    work = Path(a.work); (work / 'xyz').mkdir(parents=True, exist_ok=True)
    if a.make_reference:
        sys.path.insert(0, str(X.REPO / 'tests_pytest/fixtures/dftd3'))
        import importlib.util
        spec = importlib.util.spec_from_file_location('gen', X.REPO / 'tests_pytest/fixtures/dftd3/generate_reference.py')
        gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
        res = {}
        for R, el in SPECS:
            at = make(R, el); name = f'{el}70_R{R}'
            write(work / 'xyz' / f'{name}.xyz', at, format='extxyz')
            e, f, v = gen.reference(at, (12, 6))
            res[name] = dict(cutoffs=(12, 6), energy=e, forces=f.tolist(), virial=v.tolist(), cn_max=float(approx_cn(at).max()),
                             cn_min=float(approx_cn(at).min()))
        json.dump(res, open(work / 'reference.json', 'w')); sys.exit()
    REF = json.load(open(work / 'reference.json'))
    for name, ref in REF.items():
        line = f"{name:12s} approx CN {ref['cn_min']:6.2f}..{ref['cn_max']:6.2f}"
        for b in a.binaries:
            tag = Path(b).parent.parent.name
            n0 = X.run(b, work, name, (12, 6), False, tag); d1 = X.run(b, work, name, (12, 6), True, tag)
            e, f, v = d1[0]-n0[0], d1[1]-n0[1], d1[2]-n0[2]
            line += f" | {tag}: dE {abs(e-ref['energy']):.1e} dF {np.abs(f-np.array(ref['forces'])).max():.1e} dV {np.abs(v-np.array(ref['virial'])).max():.1e}"
        print(line, flush=True)
