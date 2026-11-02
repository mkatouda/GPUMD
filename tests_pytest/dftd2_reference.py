"""Independent NumPy reference for the DFT-D2 pairwise dispersion correction (DR-002 / DES-0002).

Model (Grimme, J. Comput. Chem. 27, 1787 (2006)), per pair at distance r:

    phi(r) = -s6 * C6_ij * f(r) / r^6,   f(r) = 1 / (1 + exp(-d (r / R0_ij - 1))),   d = 20
    C6_ij = sqrt(C6_i C6_j),   R0_ij = R0_i + R0_j  (published R0 = 1.1 * r0 of the source table)

Truncation at rc is shifted-force (DR-002 D1):  phi_sf(r) = phi(r) - phi(rc) - phi'(rc) (r - rc)
for r < rc, 0 otherwise.  With `shifted=False` the plain sharp truncation (what dftd3-lib does) is
returned instead; that is the limit in which the dftd3-lib D2 mode is the numerical oracle.

Units: positions in Angstrom, energy in eV, force in eV/A.  Conventions (GPUMD): the virial is
W_ab = sum_i r_ia F_ib = -dE/d(eps_ab) (a 3x3 matrix in eV, symmetric here), with the deformation
x -> (1 + eps) x of positions and lattice vectors.

The parameter table is `fixtures/dftd2/dftd2_params.csv` (generated from dftd3-lib by
`generate_params.py`).  Pairs and periodic images come from ase.neighbor_list (cell-list based).
Nothing here is imported by, or shares code with, the GPU implementation.
"""
from pathlib import Path

import numpy as np
from ase.neighborlist import neighbor_list

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures' / 'dftd2'
PARAMS_CSV = FIXTURE_DIR / 'dftd2_params.csv'
D_DAMP = 20.0
Z_MAX = 86


def load_params(path=PARAMS_CSV):
    """Return dict with arrays indexed by Z (index 0 unused): symbol, c6 (eV A^6), R0 (A)."""
    rows = [ln for ln in Path(path).read_text().splitlines() if ln and not ln.startswith('#')]
    header = rows[0].split(',')
    col = {name: header.index(name) for name in header}
    c6 = np.full(Z_MAX + 1, np.nan)
    R0 = np.full(Z_MAX + 1, np.nan)
    symbols = [''] * (Z_MAX + 1)
    for row in rows[1:]:
        f = row.split(',')
        z = int(f[col['Z']])
        symbols[z] = f[col['symbol']]
        c6[z] = float(f[col['c6_eV_A6']])
        R0[z] = 1.1 * float(f[col['r0_A']])
    return dict(symbol=symbols, c6=c6, R0=R0)


PARAMS = load_params()


def check_elements(atoms):
    """Raise ValueError naming every element with Z > 86 (DR-002 D5)."""
    bad = sorted({int(z) for z in atoms.numbers if z > Z_MAX or z < 1})
    if bad:
        from ase.data import chemical_symbols
        raise ValueError('no DFT-D2 parameters for: ' + ', '.join(
            chemical_symbols[z] if z < len(chemical_symbols) else str(z) for z in bad))


def pair_parameters(zi, zj):
    c6 = np.sqrt(PARAMS['c6'][zi] * PARAMS['c6'][zj])
    R0 = PARAMS['R0'][zi] + PARAMS['R0'][zj]
    return c6, R0


def phi_dphi(r, c6, R0, s6):
    """Plain (untruncated) pair energy and dphi/dr; arrays broadcast."""
    e = np.exp(-D_DAMP * (r / R0 - 1.0))
    f = 1.0 / (1.0 + e)
    dfdr = (D_DAMP / R0) * e * f * f
    phi = -s6 * c6 * f / r ** 6
    dphi = -s6 * c6 * (dfdr / r ** 6 - 6.0 * f / r ** 7)
    return phi, dphi


def _full_pairs(atoms, rc):
    """Full (both directions) pair list within rc: i, j, vector D = x_j - x_i + image shift."""
    i, j, D = neighbor_list('ijD', atoms, rc)
    return i, j, D


def energy_forces_virial(atoms, s6, rc, shifted=True):
    """Return (E [eV], F [N,3] eV/A, W [3,3] eV) of the D2 correction."""
    check_elements(atoms)
    z = atoms.numbers
    n = len(atoms)
    i, j, D = _full_pairs(atoms, rc)
    r = np.linalg.norm(D, axis=1)
    keep = r < rc
    i, j, D, r = i[keep], j[keep], D[keep], r[keep]
    c6, R0 = pair_parameters(z[i], z[j])
    phi, dphi = phi_dphi(r, c6, R0, s6)
    if shifted:
        phic, dphic = phi_dphi(rc, c6, R0, s6)
        phi = phi - phic - dphic * (r - rc)
        dphi = dphi - dphic
    energy = 0.5 * phi.sum()           # full list counts each unordered pair twice
    # F_i = + sum_j phi'(r) D_ij / r over the full list (first index i)
    fvec = (dphi / r)[:, None] * D
    forces = np.zeros((n, 3))
    np.add.at(forces, i, fvec)
    virial = -0.5 * np.einsum('pa,pb->ab', fvec, D)
    return float(energy), forces, virial


def numerical_virial(atoms, s6, rc, shifted=True, h=1e-4):
    """-dE/d(eps_ab) by a 5-point central difference of the deformation x -> (1 + h e_ab) x."""
    W = np.zeros((3, 3))
    base_cell = atoms.cell.array.copy()
    base_pos = atoms.positions.copy()
    for a in range(3):
        for b in range(3):
            def e_of(step):
                F = np.eye(3)
                F[a, b] += step
                at = atoms.copy()
                at.set_cell(base_cell @ F.T, scale_atoms=False)
                at.set_positions(base_pos @ F.T)
                return energy_forces_virial(at, s6, rc, shifted)[0]
            d = (e_of(-2 * h) - 8 * e_of(-h) + 8 * e_of(h) - e_of(2 * h)) / (12 * h)
            W[a, b] = -d
    return W


def numerical_forces(atoms, s6, rc, shifted=True, h=1e-4, indices=None):
    """-dE/dx by a 5-point central difference (optionally only for the given atom indices)."""
    idx = range(len(atoms)) if indices is None else indices
    F = np.zeros((len(atoms), 3))
    for k in idx:
        for a in range(3):
            def e_of(step):
                at = atoms.copy()
                p = at.positions
                p[k, a] += step
                at.set_positions(p)
                return energy_forces_virial(at, s6, rc, shifted)[0]
            F[k, a] = -(e_of(-2 * h) - 8 * e_of(-h) + 8 * e_of(h) - e_of(2 * h)) / (12 * h)
    return F


def shift_terms_bruteforce(atoms, s6, rc):
    """Shift-force terms (E_sf - E_plain, F_sf - F_plain, W_sf - W_plain) by explicit enumeration
    of lattice translations (no neighbor list), for converting an unshifted oracle result at the
    same rc into the shifted-force result.  Independent of `energy_forces_virial`'s pair search.
    """
    z = atoms.numbers
    pos = atoms.positions
    n = len(atoms)
    cell = atoms.cell.array
    pbc = atoms.pbc
    ranges = []
    for k in range(3):
        if pbc[k]:
            # perpendicular height of the cell along axis k
            cross = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
            height = abs(np.dot(cell[k], cross)) / np.linalg.norm(cross)
            m = int(np.ceil(rc / height)) + 1
        else:
            m = 0
        ranges.append(range(-m, m + 1))
    E = 0.0
    F = np.zeros((n, 3))
    W = np.zeros((3, 3))
    for t0 in ranges[0]:
        for t1 in ranges[1]:
            for t2 in ranges[2]:
                T = t0 * cell[0] + t1 * cell[1] + t2 * cell[2]
                zero_shift = (t0 == 0 and t1 == 0 and t2 == 0)
                for a in range(n):
                    D = pos - pos[a] + T                     # x_b - x_a + T for all b
                    r = np.linalg.norm(D, axis=1)
                    ok = (r < rc) & (r > 1e-12)
                    if zero_shift:
                        ok[a] = False
                    for b in np.nonzero(ok)[0]:
                        c6, R0 = pair_parameters(z[a], z[b])
                        phic, dphic = phi_dphi(rc, c6, R0, s6)
                        dE = -(phic + dphic * (r[b] - rc))
                        dF = -dphic * D[b] / r[b]            # shift part of phi'(r): -phi'(rc)
                        E += 0.5 * dE
                        F[a] += dF
                        W -= 0.5 * np.outer(dF, D[b])
    return E, F, W
