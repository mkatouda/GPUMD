"""Finite-difference check of the D2 force/virial formulas of docs_ja/dftd/A2_derivation_d2.md.

CPU, NumPy, float64.  Uses the real C6/R0 of H, C, O from tests_pytest/fixtures/dftd2/dftd2_params.csv
(path relative to this script) and an arbitrary s6.  The energy is the shifted-force sum
E = sum_{i<j} phi_sf(r_ij), phi_sf = phi - phi(rc) - phi'(rc)(r - rc) for r < rc.
Run:  python check_d2_derivation.py
"""
from pathlib import Path
import numpy as np

D = 20.0
S6 = 0.75
RC = 9.0
CSV = Path(__file__).resolve().parents[3] / "tests_pytest" / "fixtures" / "dftd2" / "dftd2_params.csv"


def load():
    rows = [l for l in CSV.read_text().splitlines() if l and not l.startswith("#")]
    hdr = rows[0].split(",")
    c6, r0 = {}, {}
    for l in rows[1:]:
        f = l.split(",")
        z = int(f[hdr.index("Z")])
        c6[z] = float(f[hdr.index("c6_eV_A6")])
        r0[z] = 1.1 * float(f[hdr.index("r0_A")])
    return c6, r0


C6, R0 = load()


def pair_par(zi, zj):
    return S6 * np.sqrt(C6[zi] * C6[zj]), R0[zi] + R0[zj]


def phi(r, s6c6, R):
    f = 1.0 / (1.0 + np.exp(-D * (r / R - 1.0)))
    return -s6c6 * f / r ** 6


def dphi(r, s6c6, R):
    e = np.exp(-D * (r / R - 1.0))
    f = 1.0 / (1.0 + e)
    dfdr = D / R * e * f * f
    return -s6c6 * (dfdr / r ** 6 - 6.0 * f / r ** 7)


def phi_sf(r, s6c6, R):
    return phi(r, s6c6, R) - phi(RC, s6c6, R) - dphi(RC, s6c6, R) * (r - RC)


def dphi_sf(r, s6c6, R):
    return dphi(r, s6c6, R) - dphi(RC, s6c6, R)


def pairs(x, H, nmax=2):
    out = []
    n = len(x)
    for i in range(n):
        for j in range(n):
            for sh in np.ndindex(2 * nmax + 1, 2 * nmax + 1, 2 * nmax + 1):
                nn = np.array(sh) - nmax
                r = x[j] + H @ nn - x[i]
                if 1e-8 < np.linalg.norm(r):
                    out.append((i, j, nn))
    return out


def energy(x, H, z, pl, rc_list=None):
    e = 0.0
    for i, j, nn in pl:
        r = np.linalg.norm(x[j] + H @ nn - x[i])
        if r < RC:
            e += 0.5 * phi_sf(r, *pair_par(z[i], z[j]))
    return e


def analytic(x, H, z, pl):
    N = len(x)
    F = np.zeros((N, 3))
    W = np.zeros((3, 3))
    for i, j, nn in pl:
        d = x[j] + H @ nn - x[i]
        r = np.linalg.norm(d)
        if r < RC:
            g = dphi_sf(r, *pair_par(z[i], z[j])) / r
            fp = g * d
            F[i] += fp
            W -= 0.5 * np.outer(d, fp)
    return F, W


def run(x, H, z, label, h=1e-5):
    pl = pairs(x, H)
    rr = np.array([np.linalg.norm(x[j] + H @ nn - x[i]) for i, j, nn in pl])
    assert np.min(np.abs(rr - RC)) > 100 * h
    F, W = analytic(x, H, z, pl)
    Ffd = np.zeros_like(F)
    for i in range(len(x)):
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, a] += h
            xm[i, a] -= h
            Ffd[i, a] = -(energy(xp, H, z, pl) - energy(xm, H, z, pl)) / (2 * h)
    Wfd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            eps = np.zeros((3, 3))
            eps[a, b] = h
            Gp, Gm = np.eye(3) + eps, np.eye(3) - eps
            Wfd[a, b] = -(energy(x @ Gp.T, Gp @ H, z, pl) - energy(x @ Gm.T, Gm @ H, z, pl)) / (2 * h)
    eF = np.max(np.abs(F - Ffd)) / np.max(np.abs(Ffd))
    eW = np.max(np.abs(W - Wfd)) / np.max(np.abs(Wfd))
    print(f"[{label}] max|F|={np.max(np.abs(Ffd)):.3e} eV/A: force rel err {eF:.2e}; virial rel err {eW:.2e}")
    return max(eF, eW)


def check_dphi_and_continuity():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(100):
        zi, zj = rng.choice([1, 6, 8], 2)
        p = pair_par(zi, zj)
        r = rng.uniform(1.5, 8.5)
        h = 1e-6
        fd = (phi(r + h, *p) - phi(r - h, *p)) / (2 * h)
        worst = max(worst, abs(dphi(r, *p) - fd) / abs(fd))
    p = pair_par(6, 8)
    print(f"[phi'] analytic vs central FD, 100 random pairs/r: max rel err {worst:.2e}")
    print(f"[shift] phi_sf(rc) = {phi_sf(RC, *p):.1e}, dphi_sf(rc) = {dphi_sf(RC, *p):.1e} (exactly 0 by construction)")
    return worst


if __name__ == "__main__":
    w = check_dphi_and_continuity()
    rng = np.random.default_rng(3)
    z = [6, 1, 8, 1, 6]
    x = rng.uniform(0, 5.0, size=(5, 3))
    w = max(w, run(x, np.eye(3) * 1000.0, z, "cluster, 5 atoms (C,H,O,H,C)"))
    H = np.array([[7.0, 0.6, 0.3], [0.0, 6.5, 0.4], [0.0, 0.0, 7.5]])
    x = rng.uniform(0, 1, size=(4, 3)) @ H.T
    w = max(w, run(x, H, [6, 1, 8, 1], "triclinic periodic, 4 atoms"))
    print(f"MAX relative error: {w:.2e}")
