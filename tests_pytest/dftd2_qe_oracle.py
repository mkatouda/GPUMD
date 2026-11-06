"""Double-precision numpy oracle of `dftd2 ... table qe` for any element Z <= 86 (DR-004, ADR-0004).

The pair parameters come from the QE 7.2 tables stored as CSV (fixtures/dftd2_qe/, VER-0025):
C6_ij = c6ab(Zi, Zj, 1, 1, 1) [Hartree bohr^6] * HARTREE * BOHR^6, R0_ij = 1.1 * r0ab_ij [A],
phi(r) = - s6 C6 f / r^6, f = 1 / (1 + exp(-20 (r / R0 - 1))). CODATA 2018 constants (the QE 7.2
`constants.f90` values, as in the requester's d2_reference.py, which this module extends to the full
table without modifying that copy). Truncation `plain` (hard cutoff, r < rc) or `sf` (shifted force).
Virial in the GPUMD convention (-dE/d(strain), positive under compression).
"""
import itertools
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / 'fixtures' / 'dftd2_qe'
BOHR = 0.529177210903
HARTREE = 27.211386245988
AU_C6 = HARTREE * BOHR ** 6
RS6, D_DAMP = 1.1, 20.0


def load_tables():
    c6 = np.loadtxt(FIXTURE_DIR / 'qe72_c6_first_point_Eh_bohr6_Z86.csv', delimiter=',', comments='#')
    r0 = np.loadtxt(FIXTURE_DIR / 'qe72_r0ab_angstrom_Z86.csv', delimiter=',', comments='#')
    assert c6.shape == r0.shape == (86, 86)
    return c6, r0


_C6, _R0AB = load_tables()


def pair_parameters(Z):
    """C6 [eV A^6] and R0 [A] matrices for the atomic numbers Z (1-based)."""
    idx = np.asarray(Z, int) - 1
    return _C6[np.ix_(idx, idx)] * AU_C6, RS6 * _R0AB[np.ix_(idx, idx)]


def phi_dphi(r, s6, c6, r0):
    z = np.exp(-D_DAMP * (r / r0 - 1.0))
    f = 1.0 / (1.0 + z)
    dfdr = D_DAMP / r0 * z * f * f
    return -s6 * c6 * f / r ** 6, -s6 * c6 * (dfdr / r ** 6 - 6.0 * f / r ** 7)


def d2(Z, X, cell, pbc, trunc, rc, s6=0.75):
    """(E [eV], F [eV/A], W [eV, 3x3]) of the D2 term; pbc: bool (all directions) or False."""
    c6, r0 = pair_parameters(Z)
    X = np.asarray(X, float)
    L = np.asarray(cell, float)
    n = len(X)
    if trunc == 'sf':
        prc, dprc = phi_dphi(rc, s6, c6, r0)
    else:
        assert trunc == 'plain'
    if pbc:
        vol = abs(np.linalg.det(L))
        nim = [int(np.ceil(rc / (vol / np.linalg.norm(np.cross(L[(k + 1) % 3], L[(k + 2) % 3]))))) + 1
               for k in range(3)]
    else:
        nim = [0, 0, 0]
    E, F, W = 0.0, np.zeros((n, 3)), np.zeros((3, 3))
    for img in itertools.product(*[range(-m, m + 1) for m in nim]):
        d = X[None, :, :] - X[:, None, :] + np.array(img) @ L          # d[i, j] = x_j - x_i + s
        r2 = (d ** 2).sum(-1)
        if img == (0, 0, 0):
            np.fill_diagonal(r2, np.inf)
        m = r2 < rc * rc                                               # pairs at exactly rc are excluded
        if not m.any():
            continue
        r = np.sqrt(np.where(m, r2, 1.0))
        p, dp = phi_dphi(r, s6, c6, r0)
        if trunc == 'sf':
            p = p - prc - dprc * (r - rc)
            dp = dp - dprc
        p, dp = np.where(m, p, 0.0), np.where(m, dp, 0.0)
        E += 0.5 * p.sum()
        g = (dp / r)[:, :, None] * d
        F += g.sum(1)
        W -= 0.5 * np.einsum('ija,ijb->ab', g, d)
    return E, F, W
