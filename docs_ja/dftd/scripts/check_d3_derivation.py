"""Finite-difference check of the D3(BJ) force/virial formulas of docs_ja/dftd/A1_derivation_d3.md.

CPU, NumPy, float64, a few atoms, tiny runtime.  The parameter tables are SYNTHETIC (random C6
references, made-up reference CNs): the check concerns the algebra of the derivatives, not the
numerical values of the D3 tables.  The formulas are written in the same form as the code
(src/force/dftd3/dftd3.cu): per ordered pair (i,j), r = r_j - r_i, per-atom virial split of the CN term.
Run:  python check_d3_derivation.py
"""
import numpy as np

rng = np.random.default_rng(12345)
K1, K3 = 16.0, 4.0
S6, S8, A1, A2 = 1.0, 0.7875, 0.4289, 4.4407 * 0.5291772575  # a2: bohr -> A (PBE values)

# --- synthetic "element" data: 2 elements with 3 and 2 reference CNs
NREF = {0: 3, 1: 2}
CNREF = {0: np.array([0.0, 1.0, 2.0]), 1: np.array([0.0, 1.2])}
RCOV = {0: 0.75, 1: 1.2}            # Angstrom (covalent radius incl. scaling)
R2R4 = {0: 2.0, 1: 3.0}
M = {}
for a in (0, 1):
    for b in (a, 1):
        m = rng.uniform(5.0, 60.0, size=(NREF[a], NREF[b]))   # eV A^6 scale
        if a == b:
            m = 0.5 * (m + m.T)
        M[(a, b)] = m
        M[(b, a)] = m.T


def pair_list(x, H, types, rc, nmax=3):
    """ordered pairs (i, j, n) with 0 < |x_j + H n - x_i| < rc, at the reference geometry"""
    out = []
    N = len(x)
    for i in range(N):
        for j in range(N):
            for n in np.ndindex(2 * nmax + 1, 2 * nmax + 1, 2 * nmax + 1):
                nn = np.array(n) - nmax
                r = x[j] + H @ nn - x[i]
                d = np.linalg.norm(r)
                if 1e-8 < d < rc:
                    out.append((i, j, nn))
    return out


def rvec(x, H, p):
    i, j, n = p
    return x[j] + H @ n - x[i]


def cn_all(x, H, types, plist_cn):
    cn = np.zeros(len(x))
    for p in plist_cn:
        r = np.linalg.norm(rvec(x, H, p))
        i, j = p[0], p[1]
        cn[i] += 1.0 / (1.0 + np.exp(-K1 * ((RCOV[types[i]] + RCOV[types[j]]) / r - 1.0)))
    return cn


def c6_double_sum(zi, zj, cni, cnj):
    wi = np.exp(-K3 * (cni - CNREF[zi]) ** 2)
    wj = np.exp(-K3 * (cnj - CNREF[zj]) ** 2)
    L = np.outer(wi, wj)
    return np.sum(M[(zi, zj)] * L) / np.sum(L)


def c6_factorized(zi, zj, cni, cnj):
    """Z/W form of the code (get_c6_factorized); returns C6 and dC6/dCN_i"""
    wi = np.exp(-K3 * (cni - CNREF[zi]) ** 2)
    dwi = wi * (-2.0 * K3 * (cni - CNREF[zi]))
    wj = np.exp(-K3 * (cnj - CNREF[zj]) ** 2)
    t = M[(zi, zj)] @ wj
    Z, dZ = wi @ t, dwi @ t
    W, dW = wi.sum() * wj.sum(), dwi.sum() * wj.sum()
    c6 = Z / W
    return c6, dZ / W - c6 * dW / W


def pair_terms(zi, zj, c6, r):
    c8_over_c6 = 3.0 * R2R4[zi] * R2R4[zj]
    c8 = c6 * c8_over_c6
    damp = A1 * np.sqrt(c8_over_c6) + A2
    d6 = 1.0 / (r ** 6 + damp ** 6)
    d8 = 1.0 / (r ** 8 + damp ** 8)
    return c8_over_c6, c8, d6, d8


def energy(x, H, types, pl_pot, pl_cn):
    cn = cn_all(x, H, types, pl_cn)
    e = 0.0
    for p in pl_pot:
        i, j = p[0], p[1]
        r = np.linalg.norm(rvec(x, H, p))
        c6 = c6_double_sum(types[i], types[j], cn[i], cn[j])
        _, c8, d6, d8 = pair_terms(types[i], types[j], c6, r)
        e -= 0.5 * (S6 * c6 * d6 + S8 * c8 * d8)
    return e


def analytic(x, H, types, pl_pot, pl_cn):
    N = len(x)
    cn = cn_all(x, H, types, pl_cn)
    F = np.zeros((N, 3))
    Wd = np.zeros((N, 3, 3))     # per-atom virial, direct term
    D = np.zeros(N)
    for p in pl_pot:
        i, j = p[0], p[1]
        r12 = rvec(x, H, p)
        r = np.linalg.norm(r12)
        c6, dc6 = c6_factorized(types[i], types[j], cn[i], cn[j])
        c8_c6, c8, d6, d8 = pair_terms(types[i], types[j], c6, r)
        f2 = S6 * c6 * 3 * r ** 4 * d6 ** 2 + S8 * c8 * 4 * r ** 6 * d8 ** 2
        f12 = r12 * f2
        F[i] += 2.0 * f12
        Wd[i] -= np.outer(r12, f12)
        D[i] += dc6 * S6 * d6 + dc6 * c8_c6 * S8 * d8
    Fx = np.zeros((N, 3))
    Wx = np.zeros((N, 3, 3))
    Wx_sym = np.zeros((N, 3, 3))
    for p in pl_cn:
        i, j = p[0], p[1]
        r12 = rvec(x, H, p)
        r = np.linalg.norm(r12)
        e = np.exp(-K1 * ((RCOV[types[i]] + RCOV[types[j]]) / r - 1.0))
        k12 = e * K1 * (RCOV[types[i]] + RCOV[types[j]]) / ((1 + e) ** 2 * r ** 3)
        Fx[i] += k12 * (D[i] + D[j]) * r12
        Wx[i] -= np.outer(r12, r12) * k12 * D[j]            # code: asymmetric attribution
        Wx_sym[i] -= 0.5 * np.outer(r12, r12) * k12 * (D[i] + D[j])  # symmetric alternative
    return F, Fx, Wd, Wx, Wx_sym, D


def fd_checks(x, H, types, rc_pot, rc_cn, h=1e-5, label=""):
    pl_pot = pair_list(x, H, types, rc_pot)
    pl_cn = pair_list(x, H, types, rc_cn)
    # keep the FD away from the sharp cutoffs
    for pl, rc in ((pl_pot, rc_pot), (pl_cn, rc_cn)):
        gaps = [abs(np.linalg.norm(rvec(x, H, p)) - rc) for p in pl]
    all_d = np.array([np.linalg.norm(rvec(x, H, p)) for p in pair_list(x, H, types, 1.5 * rc_pot)])
    gap = min(np.min(np.abs(all_d - rc_pot)), np.min(np.abs(all_d - rc_cn)))
    assert gap > 100 * h, "a pair is too close to a cutoff for finite differences"
    N = len(x)
    F, Fx, Wd, Wx, Wx_sym, D = analytic(x, H, types, pl_pot, pl_cn)
    Ftot = F + Fx
    # force: -dE/dx
    Ffd = np.zeros((N, 3))
    for i in range(N):
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, a] += h
            xm[i, a] -= h
            Ffd[i, a] = -(energy(xp, H, types, pl_pot, pl_cn) - energy(xm, H, types, pl_pot, pl_cn)) / (2 * h)
    scale = np.max(np.abs(Ffd))
    err_direct_only = np.max(np.abs(F - Ffd)) / scale
    err_F = np.max(np.abs(Ftot - Ffd)) / scale
    # virial: W_ab = -dE/d eps_ab, x -> (1+eps) x, H -> (1+eps) H
    Wfd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            eps = np.zeros((3, 3))
            eps[a, b] = h
            G = np.eye(3) + eps
            ep = energy(x @ G.T, G @ H, types, pl_pot, pl_cn)
            G = np.eye(3) - eps
            em = energy(x @ G.T, G @ H, types, pl_pot, pl_cn)
            Wfd[a, b] = -(ep - em) / (2 * h)
    Wcode = (Wd + Wx).sum(axis=0)
    Wsym = (Wd + Wx_sym).sum(axis=0)
    # -r (x) F_pair per ordered pair equivalent: sum_i x_i (x) F_i (cluster/periodic with images, use pair form)
    wscale = np.max(np.abs(Wfd))
    err_W = np.max(np.abs(Wcode - Wfd)) / wscale
    err_Wsym = np.max(np.abs(Wsym - Wfd)) / wscale
    err_Wsum = np.max(np.abs(Wx.sum(0) - Wx_sym.sum(0))) / wscale
    asym = np.max(np.abs(Wx - Wx_sym)) / wscale
    err_Wsymmetric = np.max(np.abs(Wfd - Wfd.T)) / wscale
    print(f"[{label}] pairs pot={len(pl_pot)} cn={len(pl_cn)}  max|F|={scale:.3e} eV/A  max|W|={wscale:.3e} eV")
    print(f"   force   : direct term only vs FD = {err_direct_only:.2e} (must be NOT small: extra term needed)")
    print(f"   force   : direct+extra  vs FD    = {err_F:.2e}")
    print(f"   virial  : code form (D_j only)   = {err_W:.2e}")
    print(f"   virial  : symmetric (D_i+D_j)/2  = {err_Wsym:.2e}")
    print(f"   per-atom CN virial: code vs symmetric, max per-atom difference (rel) = {asym:.2e}; totals differ by {err_Wsum:.2e}")
    return max(err_F, err_W, err_Wsym), err_direct_only


def check_dc6_and_factorization():
    worst_fac, worst_d, fdmax = 0.0, 0.0, 0.0
    for _ in range(200):
        zi, zj = rng.integers(0, 2, size=2)
        cni, cnj = rng.uniform(0.0, 2.5, size=2)
        c6f, dc6 = c6_factorized(zi, zj, cni, cnj)
        c6d = c6_double_sum(zi, zj, cni, cnj)
        worst_fac = max(worst_fac, abs(c6f - c6d) / abs(c6d))
        h = 1e-6
        fd = (c6_double_sum(zi, zj, cni + h, cnj) - c6_double_sum(zi, zj, cni - h, cnj)) / (2 * h)
        worst_d = max(worst_d, abs(dc6 - fd))
        fdmax = max(fdmax, abs(fd))
    print(f"[C6] factorized Z/W vs double sum, 200 random cases: max rel diff = {worst_fac:.2e}")
    worst_d /= fdmax
    print(f"[C6] dC6/dCN_i (code formula) vs central FD of the double sum: max abs err / max|dC6/dCN| = {worst_d:.2e}")
    return max(worst_fac, worst_d)


def check_k12():
    worst, fmax = 0.0, 0.0
    for _ in range(200):
        rsum = rng.uniform(1.0, 3.0)
        r = rng.uniform(1.0, 5.0)
        c = lambda rr: 1.0 / (1.0 + np.exp(-K1 * (rsum / rr - 1.0)))
        h = 1e-6
        fd = (c(r + h) - c(r - h)) / (2 * h)
        e = np.exp(-K1 * (rsum / r - 1.0))
        k12 = e * K1 * rsum / ((1 + e) ** 2 * r ** 3)
        worst = max(worst, abs(-k12 * r - fd))
        fmax = max(fmax, abs(fd))
    print(f"[CN] dc/dr = -k12 r with k12 = 16 R e/((1+e)^2 r^3): max abs err / max|dc/dr| = {worst / fmax:.2e}")
    return worst / fmax


if __name__ == "__main__":
    worst = check_dc6_and_factorization()
    worst = max(worst, check_k12())
    # cluster: 5 atoms, non-periodic (large H so that no image is within the cutoffs)
    x = rng.uniform(0.0, 4.0, size=(5, 3))
    types = [0, 1, 0, 0, 1]
    Hbig = np.eye(3) * 1000.0
    w1, d1 = fd_checks(x, Hbig, types, 8.0, 5.0, label="cluster, 5 atoms")
    # periodic triclinic cell, 4 atoms, images included
    H = np.array([[6.0, 0.5, 0.2], [0.0, 5.5, 0.3], [0.0, 0.0, 6.5]])
    xf = rng.uniform(0, 1, size=(4, 3))
    xp = xf @ H.T
    types4 = [0, 1, 0, 1]
    w2, d2 = fd_checks(xp, H, types4, 9.0, 5.5, label="triclinic periodic, 4 atoms")
    worst = max(worst, w1, w2)
    print(f"MAX relative error over all checks that must pass: {worst:.2e}")
