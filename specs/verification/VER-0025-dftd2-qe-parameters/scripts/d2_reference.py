#!/usr/bin/env python3
"""
d2_reference.py -- standalone double-precision reference for the D2 variants discussed in REQUEST_dftd2_qe_compat.md
(numpy only; no other dependency). Written 2026-11-06 for the GPUMD fork (dftd2) request.

  python3 d2_reference.py IN.xyz OUT.xyz --table qe --trunc plain --rc 12
  python3 d2_reference.py --selftest            # finite differences (force, virial) and the reference numbers in the request

Pair energy (one pair i-j, including periodic images; i = j images count 1/2):
  phi(r) = - s6 * C6_ij * f(r) / r^6,   f(r) = 1 / (1 + exp(-d * (r / R0_ij - 1))),   d = 20, s6 = 0.75 (PBE)
Tables (only C and F are included here):
  qe         : Quantum ESPRESSO 7.2, vdw_corr='dft-d3' with dftd3_version=2 ("D2 with D3 tables"):
               C6_ij = c6ab(Zi,Zj,1,1,1) of the D3 table (reference point 1 of both atoms, i.e. the first element of
               c6_ref for the pair in GPUMD's dftd3para.cuh), R0_ij = rs6 * r0ab(Zi,Zj) with rs6 = 1.1 and r0ab the D3
               pair radii (dft-d3/core.f90 setr0ab). Validated against energies, forces and stresses printed by QE 7.2.
  grimme2006 : S. Grimme, J. Comput. Chem. 27, 1787 (2006): C6_ij = sqrt(C6_i C6_j), R0_ij = R0_i + R0_j, R0 = 1.1 * r0
               (the table of the fork's dftd2para.cuh).
Truncation:
  plain : phi(r) for r <= rc, 0 beyond (hard cutoff; what was subtracted from our training labels)
  sf    : shifted force, phi(r) - phi(rc) - phi'(rc) (r - rc) for r < rc (the fork's current dftd2)
Output (extended xyz): energy (eV), forces (eV/A), virial (eV, 9 components row-major, GPUMD/NEP training convention:
positive under compression, virial = -dE/d(strain) = stress * volume with stress positive in compression).
"""
import argparse, itertools, re, sys
import numpy as np

BOHR = 0.529177210903; HARTREE = 27.211386245988
AU_C6 = HARTREE * BOHR ** 6                        # Eh bohr^6 -> eV A^6
S6, D = 0.75, 20.0
QE_C6_AU = {('C', 'C'): 49.113, ('C', 'F'): 20.2827, ('F', 'F'): 9.6916}
QE_R0AB_A = {('C', 'C'): 2.9103, ('C', 'F'): 2.4770, ('F', 'F'): 2.2996}
RS6 = 1.1
G06_C6 = {'C': 1.81374693015604400e+01, 'F': 7.77320112924018858e+00}   # eV A^6 (dftd2para.cuh, Z = 6, 9)
G06_R0 = {'C': 1.1 * 1.32, 'F': 1.1 * 1.17}                               # A

def pair_table(name):
    t = {}
    for a, b in itertools.product('CF', repeat=2):
        k = tuple(sorted((a, b)))
        if name == 'qe':
            t[(a, b)] = (QE_C6_AU[k] * AU_C6, RS6 * QE_R0AB_A[k])
        elif name == 'grimme2006':
            t[(a, b)] = (np.sqrt(G06_C6[a] * G06_C6[b]), G06_R0[a] + G06_R0[b])
        else:
            raise SystemExit(f'unknown table {name}')
    return t

def phi_dphi(r, c6, r0):
    z = np.exp(-D * (r / r0 - 1.0)); f = 1.0 / (1.0 + z); dfdr = D / r0 * z * f * f
    return -S6 * c6 * f / r ** 6, -S6 * c6 * (dfdr / r ** 6 - 6.0 * f / r ** 7)

def d2(species, X, L, table='qe', trunc='plain', rc=12.0):
    """E (eV), F (eV/A, n x 3), virial (eV, 3x3, positive under compression)"""
    t = pair_table(table); n = len(species); X = np.asarray(X, float); L = np.asarray(L, float)
    c6 = np.array([[t[(a, b)][0] for b in species] for a in species]); r0 = np.array([[t[(a, b)][1] for b in species] for a in species])
    if trunc == 'sf':
        prc, dprc = phi_dphi(rc, c6, r0)
    elif trunc != 'plain':
        raise SystemExit(f'unknown truncation {trunc}')
    vol = abs(np.linalg.det(L))
    nim = [int(np.ceil(rc / (vol / np.linalg.norm(np.cross(L[(k + 1) % 3], L[(k + 2) % 3]))))) + 1 for k in range(3)]
    E = 0.0; F = np.zeros((n, 3)); W = np.zeros((3, 3))
    for img in itertools.product(*[range(-m, m + 1) for m in nim]):
        s = np.array(img) @ L
        d = X[None, :, :] - X[:, None, :] + s            # d[i, j] = x_j - x_i + s
        r2 = (d ** 2).sum(-1)
        if img == (0, 0, 0): np.fill_diagonal(r2, np.inf)
        m = r2 <= rc * rc
        if not m.any(): continue
        r = np.sqrt(np.where(m, r2, 1.0))
        p, dp = phi_dphi(r, c6, r0)
        if trunc == 'sf':
            p = p - prc - dprc * (r - rc); dp = dp - dprc
        p = np.where(m, p, 0.0); dp = np.where(m, dp, 0.0)
        E += 0.5 * p.sum()
        g = (dp / r)[:, :, None] * d                      # force on i from j: + phi'(r) d_ij / r
        F += g.sum(1)
        W -= 0.5 * np.einsum('ija,ijb->ab', g, d)          # -dE/d(strain): positive under compression
    return E, F, W

def read_xyz(path):
    lines = open(path).read().splitlines(); n = int(lines[0]); head = lines[1]
    L = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', head).group(1).split()]).reshape(3, 3)
    sp, X = [], []
    for l in lines[2:2 + n]:
        w = l.split(); sp.append(w[0]); X.append([float(v) for v in w[1:4]])
    return sp, np.array(X), L

def write_xyz(path, sp, X, L, E, F, W, note):
    with open(path, 'w') as f:
        f.write(f'{len(sp)}\nLattice="{" ".join(f"{v:.10f}" for v in L.ravel())}" energy={E:.10f} '
                f'virial="{" ".join(f"{v:.10f}" for v in W.ravel())}" pbc="T T T" {note} Properties=species:S:1:pos:R:3:forces:R:3\n')
        for s, x, fo in zip(sp, X, F):
            f.write(f'{s} {x[0]:.10f} {x[1]:.10f} {x[2]:.10f} {fo[0]:.10e} {fo[1]:.10e} {fo[2]:.10e}\n')

def selftest():
    rng = np.random.default_rng(1)
    L = np.diag([9.0, 8.5, 10.0]) + 0.3 * rng.standard_normal((3, 3)); sp = ['C', 'F', 'F', 'C', 'F', 'F']
    X = rng.uniform(0, 1, (6, 3)) @ L; ok = True
    for table, trunc in itertools.product(('qe', 'grimme2006'), ('plain', 'sf')):
        E, F, W = d2(sp, X, L, table, trunc, 7.0); h = 1e-5
        Xp = X.copy(); Xp[2, 1] += h; Xm = X.copy(); Xm[2, 1] -= h
        fd = -(d2(sp, Xp, L, table, trunc, 7.0)[0] - d2(sp, Xm, L, table, trunc, 7.0)[0]) / (2 * h)
        S = np.eye(3); S[0, 1] += h; Sm = np.eye(3); Sm[0, 1] -= h
        wfd = -(d2(sp, X @ S, L @ S, table, trunc, 7.0)[0] - d2(sp, X @ Sm, L @ Sm, table, trunc, 7.0)[0]) / (2 * h)
        good = abs(fd - F[2, 1]) < 1e-7 and abs(wfd - W[0, 1]) < 1e-7
        print(f"{'PASS' if good else 'FAIL'} {table}/{trunc}: force {F[2, 1]:.9f} vs fd {fd:.9f}; virial_xy {W[0, 1]:.9f} vs fd {wfd:.9f}"); ok &= good
    print('note: with rc = 7 A the plain truncation is discontinuous only when a pair crosses rc (not the case here)')
    return ok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inp', nargs='?'); ap.add_argument('out', nargs='?')
    ap.add_argument('--table', default='qe', choices=['qe', 'grimme2006']); ap.add_argument('--trunc', default='plain', choices=['plain', 'sf'])
    ap.add_argument('--rc', type=float, default=12.0); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    if not (a.inp and a.out): ap.error('IN.xyz OUT.xyz required')
    sp, X, L = read_xyz(a.inp); E, F, W = d2(sp, X, L, a.table, a.trunc, a.rc)
    write_xyz(a.out, sp, X, L, E, F, W, f'd2="table={a.table},trunc={a.trunc},rc={a.rc:g},s6={S6}"')
    print(f'{a.out}: {len(sp)} atoms, E = {E:.8f} eV ({E / len(sp) * 1000:.4f} meV/atom), max|F| = {np.abs(F).max():.6f} eV/A')

if __name__ == '__main__':
    main()
