"""Finite-difference check of the D3(0) (zero damping) formulas of docs_ja/dftd/A1_derivation_d3.md, A1.9.

CPU, NumPy, float64 (plus float32 for the overflow test).  Parameter tables are SYNTHETIC (random C6
references, made-up reference CNs and R0): the check concerns the algebra of the derivatives and the
overflow-safe evaluation, not the values of the D3 tables.  Formulas follow the code
(src/force/dftd3.cu: d3_zero_term, d3_damped_pair); the CN machinery is that of check_d3_derivation.py.
Run:  python check_d3zero_derivation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_d3_derivation as b  # noqa: E402  (BJ check; main is guarded)

rng = np.random.default_rng(2026)
S6, S8, SR6, SR8 = 1.0, 0.722, 1.217, 1.0       # PBE zero-damping values (VER-0023), synthetic C6
R0 = {(0, 0): 2.2, (0, 1): 3.0, (1, 0): 3.0, (1, 1): 3.4}   # synthetic R0 [A]


# ---------- 1. g(r) = f/r^n, dg/dr = -(g/r)(n - alpha(1-f)) ----------
def g_naive(r, a, n):
    t = r / a
    return 1.0 / (1.0 + 6.0 * t ** (-(n + 8))) / r ** n


def g_h(r, a, n, dtype=np.float64):
    """overflow-safe form of d3_zero_term: returns g and h = n - alpha (1 - f)"""
    r, a = dtype(r), dtype(a)
    alpha = dtype(n + 8)
    t = r / a
    if t < 1:
        p = t ** (n + 8)
        inv = dtype(1) / (p + dtype(6))
        g = t ** 8 * (dtype(1) / a) ** n * inv
        h = n - alpha * dtype(6) * inv
    else:
        q = (dtype(1) / t) ** (n + 8)
        finv = dtype(1) / (dtype(1) + dtype(6) * q)
        g = (dtype(1) / r) ** n * finv
        h = n - alpha * dtype(6) * q * finv
    return g, h


def check_dg():
    worst = 0.0
    for n in (6, 8):
        for _ in range(400):
            a = rng.uniform(1.5, 4.5)
            r = a * 10 ** rng.uniform(-0.7, 0.7)      # t = 0.2 ... 5
            hh = 1e-6 * r
            fd = (g_naive(r + hh, a, n) - g_naive(r - hh, a, n)) / (2 * hh)
            g, h = g_h(r, a, n)
            an = -g * h / r
            worst = max(worst, abs(an - fd) / abs(fd))
            # naive form of 1-f: 6 t^-alpha f
            t = r / a
            f = 1.0 / (1.0 + 6.0 * t ** (-(n + 8)))
            assert abs((1 - f) - 6.0 * t ** (-(n + 8)) * f) < 1e-14
    print(f"[dg/dr] -(g/r)(n - alpha(1-f)) vs central FD, 800 cases (t = 0.2..5): max rel err = {worst:.2e}")
    return worst


def check_overflow():
    # float32 safe form vs float64 reference over many decades; naive float32 form overflows
    worst, nan_naive, nan_safe = 0.0, 0, 0
    with np.errstate(all="ignore"):
        for n in (6, 8):
            a = np.float32(2.7)
            for r in np.float32(10.0) ** np.linspace(-9, 1.7, 2000).astype(np.float32):
                g32, h32 = g_h(r, a, n, np.float32)
                g64, h64 = g_h(float(r), float(a), n, np.float64)
                if not np.isfinite(g32) or not np.isfinite(h32):
                    nan_safe += 1
                    continue
                if g64 > 1e-30:      # float32 normal range
                    worst = max(worst, abs(g32 - g64) / g64)
                gn = np.float32(1) / (np.float32(1) + np.float32(6) * (r / a) ** np.float32(-(n + 8))) / r ** np.float32(n)
                if not np.isfinite(gn):
                    nan_naive += 1
    print(f"[overflow] r = 1e-9..50 A, float32: safe form non-finite in {nan_safe} of 4000 points, "
          f"naive form non-finite in {nan_naive}; safe float32 vs float64 max rel err = {worst:.2e}")
    return worst, nan_safe, nan_naive


# ---------- 2. whole-system force and virial including the CN derivative ----------
def r0pair(zi, zj):
    return R0[(zi, zj)]


def c8c6(zi, zj):
    return 3.0 * b.R2R4[zi] * b.R2R4[zj]


def energy(x, H, types, pl_pot, pl_cn):
    cn = b.cn_all(x, H, types, pl_cn)
    e = 0.0
    for p in pl_pot:
        i, j = p[0], p[1]
        r = np.linalg.norm(b.rvec(x, H, p))
        zi, zj = types[i], types[j]
        c6 = b.c6_double_sum(zi, zj, cn[i], cn[j])
        g6 = g_naive(r, SR6 * r0pair(zi, zj), 6)
        g8 = g_naive(r, SR8 * r0pair(zi, zj), 8)
        e -= 0.5 * (S6 * c6 * g6 + S8 * c6 * c8c6(zi, zj) * g8)
    return e


def analytic(x, H, types, pl_pot, pl_cn):
    N = len(x)
    cn = b.cn_all(x, H, types, pl_cn)
    F = np.zeros((N, 3))
    Wd = np.zeros((N, 3, 3))
    D = np.zeros(N)
    for p in pl_pot:
        i, j = p[0], p[1]
        zi, zj = types[i], types[j]
        r12 = b.rvec(x, H, p)
        r = np.linalg.norm(r12)
        c6, dc6 = b.c6_factorized(zi, zj, cn[i], cn[j])
        c8 = c6 * c8c6(zi, zj)
        g6, h6 = g_h(r, SR6 * r0pair(zi, zj), 6)
        g8, h8 = g_h(r, SR8 * r0pair(zi, zj), 8)
        f2 = 0.5 * (S6 * c6 * g6 * h6 + S8 * c8 * g8 * h8) / r ** 2     # code: f2 (force = 2 f2 r12)
        f12 = r12 * f2
        F[i] += 2.0 * f12
        Wd[i] -= np.outer(r12, f12)
        D[i] += dc6 * (S6 * g6 + c8c6(zi, zj) * S8 * g8)               # D_i = -dE/dCN_i
    Fx = np.zeros((N, 3))
    Wx = np.zeros((N, 3, 3))
    for p in pl_cn:
        i, j = p[0], p[1]
        r12 = b.rvec(x, H, p)
        r = np.linalg.norm(r12)
        e = np.exp(-b.K1 * ((b.RCOV[types[i]] + b.RCOV[types[j]]) / r - 1.0))
        k12 = e * b.K1 * (b.RCOV[types[i]] + b.RCOV[types[j]]) / ((1 + e) ** 2 * r ** 3)
        Fx[i] += k12 * (D[i] + D[j]) * r12
        Wx[i] -= np.outer(r12, r12) * k12 * D[j]
    return F, Fx, Wd, Wx


def fd_checks(x, H, types, rc_pot, rc_cn, h=1e-5, label=""):
    pl_pot = b.pair_list(x, H, types, rc_pot)
    pl_cn = b.pair_list(x, H, types, rc_cn)
    all_d = np.array([np.linalg.norm(b.rvec(x, H, p)) for p in b.pair_list(x, H, types, 1.5 * rc_pot)])
    gap = min(np.min(np.abs(all_d - rc_pot)), np.min(np.abs(all_d - rc_cn)))
    assert gap > 100 * h, "a pair is too close to a cutoff for finite differences"
    N = len(x)
    F, Fx, Wd, Wx = analytic(x, H, types, pl_pot, pl_cn)
    Ffd = np.zeros((N, 3))
    for i in range(N):
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, a] += h
            xm[i, a] -= h
            Ffd[i, a] = -(energy(xp, H, types, pl_pot, pl_cn) - energy(xm, H, types, pl_pot, pl_cn)) / (2 * h)
    scale = np.max(np.abs(Ffd))
    err_direct = np.max(np.abs(F - Ffd)) / scale
    err_F = np.max(np.abs(F + Fx - Ffd)) / scale
    Wfd = np.zeros((3, 3))
    for a in range(3):
        for c in range(3):
            eps = np.zeros((3, 3))
            eps[a, c] = h
            G = np.eye(3) + eps
            ep = energy(x @ G.T, G @ H, types, pl_pot, pl_cn)
            G = np.eye(3) - eps
            em = energy(x @ G.T, G @ H, types, pl_pot, pl_cn)
            Wfd[a, c] = -(ep - em) / (2 * h)
    wscale = np.max(np.abs(Wfd))
    err_W = np.max(np.abs((Wd + Wx).sum(axis=0) - Wfd)) / wscale
    print(f"[{label}] pairs pot={len(pl_pot)} cn={len(pl_cn)}  max|F|={scale:.3e} eV/A  max|W|={wscale:.3e} eV")
    print(f"   force : direct term only vs FD = {err_direct:.2e} (must be NOT small: CN term needed)")
    print(f"   force : direct+extra  vs FD    = {err_F:.2e}")
    print(f"   virial: direct+extra  vs FD    = {err_W:.2e}")
    return max(err_F, err_W), err_direct


if __name__ == "__main__":
    w1 = check_dg()
    w2, nan_safe, nan_naive = check_overflow()
    x = rng.uniform(0.0, 4.0, size=(5, 3))
    types = [0, 1, 0, 0, 1]
    w3, d3 = fd_checks(x, np.eye(3) * 1000.0, types, 8.0, 5.0, label="cluster, 5 atoms")
    H = np.array([[6.0, 0.5, 0.2], [0.0, 5.5, 0.3], [0.0, 0.0, 6.5]])
    xp = rng.uniform(0, 1, size=(4, 3)) @ H.T
    w4, d4 = fd_checks(xp, H, [0, 1, 0, 1], 9.0, 5.5, label="triclinic periodic, 4 atoms")
    worst = max(w1, w3, w4)
    print(f"MAX relative error over the finite-difference checks that must pass: {worst:.2e}")
    print(f"float32 safe-form max rel err {w2:.2e}; non-finite: safe {nan_safe}, naive {nan_naive}")
    ok = worst < 1e-7 and nan_safe == 0 and d3 > 0.01 and d4 > 0.01
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
