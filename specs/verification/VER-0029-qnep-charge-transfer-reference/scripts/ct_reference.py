#!/usr/bin/env python3
"""VER-0029 / DR-005 stage 2a: float64 numpy reference of the charge-transfer qNEP and its exact BEC.

q_a = sum_b phi_ab (h_a - h_b),  P = sum_a q_a r_a = 0.5 sum_(a,b,s) phi (h_a-h_b)(r_a-r_b-s),
Z[k,beta,alpha] = dP_beta/dr_k,alpha.  Neighbours are (b, lattice shift) pairs so that rc > L/2 also works
(informational case); the gated cases use rc < L/2 (then this is the minimum image convention).
Usage: python ct_reference.py [--quick]
"""
import sys
import numpy as np

# ---------------------------------------------------------------- basis, cutoff
def fc(r, rc):
    return 0.5 * (np.cos(np.pi * r / rc) + 1.0), -0.5 * np.pi / rc * np.sin(np.pi * r / rc)


def basis(r, rc, K):
    """NEP-like radial basis f_k(r) fc(r), k < K; returns value, derivative (K, len(r))."""
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    dx = 4.0 * (r / rc - 1.0) / rc
    T = [np.ones_like(x), x]
    dT = [np.zeros_like(x), np.ones_like(x)]
    for k in range(1, K):
        T.append(2 * x * T[k] - T[k - 1])
        dT.append(2 * T[k] + 2 * x * dT[k] - dT[k - 1])
    c, dc = fc(r, rc)
    f = np.array([0.5 * (T[k] + 1.0) for k in range(K)])
    df = np.array([0.5 * dT[k] * dx for k in range(K)])
    return f * c, df * c + f * dc


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, seed, rc_d, rc_p, ntype=2, nrad=3, nang=2, K=4, nh=10):
        rng = np.random.default_rng(seed)
        self.rc_d, self.rc_p, self.nt, self.nrad, self.nang, self.K, self.nh = rc_d, rc_p, ntype, nrad, nang, K, nh
        sym = lambda A: 0.5 * (A + A.swapaxes(0, 1))  # symmetric in the type pair (first two axes)
        self.cr = sym(rng.normal(0, 0.5, (ntype, ntype, nrad, K)))
        self.ca = sym(rng.normal(0, 0.5, (ntype, ntype, nang, K)))
        self.cphi = sym(rng.uniform(0.0, 1.0, (ntype, ntype, 1, K)))[:, :, 0, :]
        nd = nrad + nang
        self.W0 = rng.normal(0, 0.7, (ntype, nh, nd))
        self.b0 = rng.normal(0, 0.3, (ntype, nh))
        self.w1 = rng.normal(0, 0.7, (ntype, nh))
        self.b1 = rng.normal(0, 0.3, ntype)
        self.nd = nd


def neighbours(pos, L, rc):
    """per atom: (index b, vec = r_b + s - r_a, r) for 0 < r < rc, all lattice images (b,s) incl. b=a, s!=0."""
    N = len(pos)
    n = int(np.ceil(rc / L))
    rng_ = np.arange(-n, n + 1)
    sh = np.array(np.meshgrid(rng_, rng_, rng_, indexing="ij")).reshape(3, -1).T * L
    d = pos[None, :, :] - pos[:, None, :]
    d -= L * np.round(d / L)
    out = []
    for a in range(N):
        v = d[a][:, None, :] + sh[None, :, :]
        r = np.linalg.norm(v, axis=2)
        ib, is_ = np.nonzero((r > 1e-12) & (r < rc))
        out.append((ib, v[ib, is_], r[ib, is_]))
    return out


def descriptor(m, pos, types, L, nl):
    N = len(pos)
    d = np.zeros((N, m.nd))
    cache = []
    for a in range(N):
        ib, v, r = nl[a]
        tb = types[ib]
        fb, dfb = basis(r, m.rc_d, m.K)
        g_r = np.einsum("jnk,kj->nj", m.cr[types[a], tb], fb)      # (nrad, nb)
        dg_r = np.einsum("jnk,kj->nj", m.cr[types[a], tb], dfb)
        g_a = np.einsum("jnk,kj->nj", m.ca[types[a], tb], fb)      # (nang, nb)
        dg_a = np.einsum("jnk,kj->nj", m.ca[types[a], tb], dfb)
        S = np.einsum("mj,jx->mx", g_a, v)                         # (nang,3)
        d[a, :m.nrad] = g_r.sum(1)
        d[a, m.nrad:] = (S ** 2).sum(1)
        cache.append((g_r, dg_r, g_a, dg_a, S))
    return d, cache


def network(m, d, types):
    """h and dh/d(descriptor), tanh network per atom type."""
    z = np.einsum("ahd,ad->ah", m.W0[types], d) + m.b0[types]
    t = np.tanh(z)
    h = (m.w1[types] * t).sum(1) - m.b1[types]
    dh = np.einsum("ah,ahd->ad", m.w1[types] * (1 - t ** 2), m.W0[types])
    return h, dh


def phi_of(m, ta, tb, r):
    f, df = basis(r, m.rc_p, m.K)
    c = m.cphi[ta, tb]
    return np.einsum("jk,kj->j", c, f), np.einsum("jk,kj->j", c, df)


def charges_P(m, pos, types, L):
    """q (via the Laplacian form) and P (via the pair sum), nothing else."""
    N = len(pos)
    nl = neighbours(pos, L, m.rc_d)
    d, _ = descriptor(m, pos, types, L, nl)
    h, _ = network(m, d, types)
    nlp = neighbours(pos, L, m.rc_p)
    q = np.zeros(N)
    P = np.zeros(3)
    for a in range(N):
        ib, v, r = nlp[a]
        ph, _ = phi_of(m, types[a], types[ib], r)
        w = ph * (h[a] - h[ib])
        q[a] = w.sum()
        P += 0.5 * (w[:, None] * (-v)).sum(0)   # 0.5 sum phi (h_a-h_b)(r_a-r_b-s); -v = r_a - r_b - s
    return q, P, h


# ---------------------------------------------------------------- exact BEC, analytic a-centred form
def bec_analytic(m, pos, types, L):
    """Z[k,beta,alpha] = dP_beta/dr_k,alpha = Term A + Term B (DES-0005 section 3), loops over a and its neighbours only."""
    N = len(pos)
    nl = neighbours(pos, L, m.rc_d)
    d, cache = descriptor(m, pos, types, L, nl)
    h, dhdd = network(m, d, types)
    nlp = neighbours(pos, L, m.rc_p)
    # V_a = sum_b phi_ab (r_a - r_b)
    V = np.zeros((N, 3))
    for a in range(N):
        ib, v, r = nlp[a]
        ph, _ = phi_of(m, types[a], types[ib], r)
        V[a] = (ph[:, None] * (-v)).sum(0)
    Z = np.zeros((N, 3, 3))
    # Term A: sum_a V_a,beta dh_a/dr_k,alpha ; force-like, per a over its descriptor neighbours
    for a in range(N):
        ib, v, r = nl[a]
        g_r, dg_r, g_a, dg_a, S = cache[a]
        u = v / r[:, None]
        G = np.einsum("n,nj,jx->jx", dhdd[a, :m.nrad], dg_r, u)            # dh_a/dr_b, radial part
        Sv = v @ S.T                                                       # (nb, nang) = S_m . vec
        coef = dhdd[a, m.nrad:]
        G += 2 * np.einsum("m,mj,jm,jx->jx", coef, dg_a, Sv, u)
        G += 2 * np.einsum("m,mj,mx->jx", coef, g_a, S)
        for j in range(len(ib)):
            Z[ib[j]] += np.outer(V[a], G[j])
        Z[a] -= np.outer(V[a], G.sum(0))
    # Term B: Z_k += sum_b M_kb (h_k - h_b),  M = phi' xhat (x) x + phi 1,  x = r_k - r_b  (includes the q_k 1 diagonal)
    for k in range(N):
        ib, v, r = nlp[k]
        ph, dph = phi_of(m, types[k], types[ib], r)
        x = -v
        M = (dph / r)[:, None, None] * x[:, :, None] * x[:, None, :] + ph[:, None, None] * np.eye(3)
        Z[k] += np.einsum("jab,j->ab", M, h[k] - h[ib])
    return Z


def bec_fd(m, pos, types, L, delta=1e-5):
    N = len(pos)
    Z = np.zeros((N, 3, 3))
    for k in range(N):
        for al in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[k, al] += delta
            pm[k, al] -= delta
            Z[k, :, al] = (charges_P(m, pp, types, L)[1] - charges_P(m, pm, types, L)[1]) / (2 * delta)
    return Z


def jac_q_fd(m, pos, types, L, delta=1e-5):
    N = len(pos)
    J = np.zeros((N, N, 3))   # J[a,k,alpha] = dq_a/dr_k,alpha
    for k in range(N):
        for al in range(3):
            pp, pm = pos.copy(), pos.copy()
            pp[k, al] += delta
            pm[k, al] -= delta
            J[:, k, al] = (charges_P(m, pp, types, L)[0] - charges_P(m, pm, types, L)[0]) / (2 * delta)
    return J


def old_forms(m, pos, types, L, q, J):
    """H0 kernel formula qn*1 + 0.5 T1 + 0.5 T2 (qn = q - mean(q), raw J) and T2-only q 1 + T2, rel = MIC(r_a - r_k)."""
    N = len(pos)
    rel = pos[:, None, :] - pos[None, :, :]       # rel[a,k] = r_a - r_k
    rel -= L * np.round(rel / L)
    T2 = np.einsum("akb,akx->kbx", rel, J)         # sum_a rel_ak (x) dq_a/dr_k
    T1 = np.einsum("jkb,kjx->kbx", rel, J)         # sum_j rel_jk (x) dq_k/dr_j
    qn = q - q.mean()
    eye = np.eye(3)
    return qn[:, None, None] * eye + 0.5 * T1 + 0.5 * T2, q[:, None, None] * eye + T2


# ---------------------------------------------------------------- configurations
def random_config(seed, N, L, rmin=1.2):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < N:
        p = rng.uniform(0, L, 3)
        if all(np.linalg.norm((p - x) - L * np.round((p - x) / L)) > rmin for x in pos):
            pos.append(p)
    return np.array(pos), rng.integers(0, 2, N)


rms = lambda A: float(np.sqrt(np.mean(A ** 2)))
mx = lambda A: float(np.max(np.abs(A)))


def main():
    quick = "--quick" in sys.argv
    ok = True
    # (name, N, L, rc_d, rc_p, gated)  thin: 2 rc < L only slightly; 'image': rc > L/2 (informational)
    cases = [("N8 thick", 8, 10.0, 4.0, 4.0, True),
             ("N45 thick", 45, 12.0, 4.5, 4.0, True),
             ("N59 thick", 59, 14.0, 5.0, 5.0, True),
             ("N12 thin", 12, 7.5, 3.6, 3.7, True),
             ("N59 big L=22", 59, 22.0, 3.5, 3.5, True),
             ("N10 images rc>L/2", 10, 6.0, 4.0, 4.5, False)]
    if quick:
        cases = [cases[0], cases[3]]  # N8 thick, N12 thin
    print("G1/G3: case | max|sum q| | max|Z_an-Z_FD| | max|sum_k Z| | RMS(Z) | RMS(H0-Z) | RMS(T2only-Z) | |Z-Z^T| rms")
    for i, (name, N, L, rd, rp, gated) in enumerate(cases):
        pos, types = random_config(100 + i, N, L)
        m = Model(7 + i, rd, rp)
        q, P, h = charges_P(m, pos, types, L)
        Za, Zf = bec_analytic(m, pos, types, L), bec_fd(m, pos, types, L)
        J = jac_q_fd(m, pos, types, L)
        H0, T2o = old_forms(m, pos, types, L, q, J)
        e_q, e_z, e_s = mx(q.sum()), mx(Za - Zf), mx(Za.sum(0))
        g = (e_q <= 1e-12 and e_z <= 1e-8 and e_s <= 1e-10) if gated else None
        ok &= (g is not False)
        print(f"{name:20s} {e_q:.1e} {e_z:.1e} {e_s:.1e} {rms(Za):.3f} {rms(H0 - Za):.3f} {rms(T2o - Za):.2e} "
              f"{rms(Za - Za.swapaxes(1, 2)):.3f}  gate={'PASS' if g else ('FAIL' if g is False else 'info')}  "
              f"rd+rp={rd+rp} vs L/2={L/2}")
    # G2 invariances on a cluster in a big box
    L = 60.0
    pos, types = random_config(5, 20, 7.0)
    pos = pos + 26.5
    m = Model(3, 4.0, 4.0)
    Z0 = bec_analytic(m, pos, types, L)
    rng = np.random.default_rng(1)
    Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Qm *= np.sign(np.linalg.det(Qm))
    c0 = pos.mean(0)
    Zr = bec_analytic(m, (pos - c0) @ Qm.T + c0, types, L)
    e_rot = mx(Zr - np.einsum("ab,kbc,dc->kad", Qm, Z0, Qm))
    ps = pos.copy()
    ps[0] += np.array([L, 0, 0])
    e_img = mx(bec_analytic(m, ps, types, L) - Z0)
    perm = rng.permutation(len(pos))
    e_perm = mx(bec_analytic(m, pos[perm], types[perm], L) - Z0[perm])
    e_sum = mx(Z0.sum(0))
    e_fd = mx(Z0 - bec_fd(m, pos, types, L))
    print(f"G2 (cluster, L=60): sumZ {e_sum:.1e}  rotation {e_rot:.1e}  image shift {e_img:.1e}  permutation {e_perm:.1e}  analytic-vs-FD {e_fd:.1e}")
    ok &= e_sum <= 1e-10 and e_rot <= 1e-9 and e_img <= 1e-9 and e_perm <= 1e-9 and e_fd <= 1e-8
    print("ALL GATES", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
