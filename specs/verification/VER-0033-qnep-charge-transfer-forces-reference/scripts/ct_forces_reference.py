#!/usr/bin/env python3
"""VER-0033 / DR-005 stage 2c step 1: float64 numpy reference of forces and virial of the charge_transfer qNEP charge energy.

E(r, H) = E_el(q(r,H), r, H) with q_a = sum_b phi_ab (h_a - h_b) (model of VER-0029, imported unchanged) and a TOY electrostatic energy
E_el = 0.5 sum_{a,b,s} q_a q_b K(|r_a - r_b - s|) + 0.5 s0 sum_a q_a^2   (K = erfc(r/s)/r * 0.5(cos(pi r/rcK)+1), s = 1.5, s0 = 0.3).
The NEP-side formulas (Dt weight, ct pair term, virial) use only D_a = dE_el/dq_a and the explicit fixed-q pair force; they do not depend on K.
Conventions: v = r_b + s - r_a (neighbour vector, as VER-0029); forces F = -dE/dr; virial W_ab = sum_a r_a,alpha F_a,beta (periodic: pair form)
 = -dE/d eps_(beta alpha) for r -> (1+eps) r, H -> H (1+eps)^T (GPUMD virial_per_atom index xy = r_x F_y).
Usage: python ct_forces_reference.py [--quick]
"""
import os
import sys
import importlib.util
import numpy as np
from scipy.special import erfc

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "VER-0029-qnep-charge-transfer-reference", "scripts", "ct_reference.py")
_s = importlib.util.spec_from_file_location("ct_reference", _p)
ct = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ct)
Model, descriptor, network, phi_of, random_config = ct.Model, ct.descriptor, ct.network, ct.phi_of, ct.random_config

KS, S0 = 1.5, 0.3
DELTA4 = 5e-5   # step of the 5-point stencil used for the gate (fixed from delta_scan.py; see README)
rms = lambda A: float(np.sqrt(np.mean(np.asarray(A) ** 2)))
mx = lambda A: float(np.max(np.abs(A)))


def Kfun(r, rcK):
    w = 0.5 * (np.cos(np.pi * r / rcK) + 1.0)
    dw = -0.5 * np.pi / rcK * np.sin(np.pi * r / rcK)
    e = erfc(r / KS) / r
    de = -2.0 / (KS * np.sqrt(np.pi)) * np.exp(-(r / KS) ** 2) / r - erfc(r / KS) / r ** 2
    inside = r < rcK
    return np.where(inside, e * w, 0.0), np.where(inside, de * w + e * dw, 0.0)


def neighbours(pos, H, rc):
    """general cell (rows of H = lattice vectors): per atom (ib, v = r_b + s - r_a, r), 0 < r < rc, all images."""
    N = len(pos)
    f = pos @ np.linalg.inv(H)
    pos = (f - np.floor(f)) @ H
    Lmin = min(np.linalg.norm(H, axis=1))
    n = 1 if rc < 0.499 * Lmin else int(np.ceil(rc / Lmin)) + 1
    g = np.arange(-n, n + 1)
    sh = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T @ H
    d = pos[None, :, :] - pos[:, None, :]
    out = []
    for a in range(N):
        v = d[a][:, None, :] + sh[None, :, :]
        r = np.linalg.norm(v, axis=2)
        ib, is_ = np.nonzero((r > 1e-12) & (r < rc))
        out.append((ib, v[ib, is_], r[ib, is_]))
    return out


def cutoff_gap(m, pos, H, rcK):
    """min over all pairs of |r - rc| for rc in (rc_d, rc_p, rcK): the energy is only C1 at the cutoffs (fc'' jumps),
    so a central FD stencil is only valid if no pair crosses a cutoff inside it."""
    gap = 1e9
    for rc in (m.rc_d, m.rc_p, rcK):
        for ib, v, r in neighbours(pos, H, rc + 0.5):
            if len(r):
                gap = min(gap, float(np.min(np.abs(r - rc))))
    return gap


def energy(m, pos, H, types, rcK):
    N = len(pos)
    nl = neighbours(pos, H, m.rc_d)
    d, _ = descriptor(m, pos, types, None, nl)
    h, _ = network(m, d, types)
    nlp = neighbours(pos, H, m.rc_p)
    q = np.zeros(N)
    for a in range(N):
        ib, v, r = nlp[a]
        ph, _ = phi_of(m, types[a], types[ib], r)
        q[a] = (ph * (h[a] - h[ib])).sum()
    nlK = neighbours(pos, H, rcK)
    E = 0.5 * S0 * (q ** 2).sum()
    for a in range(N):
        ib, v, r = nlK[a]
        K, _ = Kfun(r, rcK)
        E += 0.5 * q[a] * (q[ib] * K).sum()
    return E


def analytic(m, pos, H, types, rcK, old_weight=False, ct_pair=True):
    """a-centred forces and virial. old_weight: weight D - mean(D) and no ct pair term (the zero_mean_D_real form).
    Returns F, W_total and dict of pieces."""
    N = len(pos)
    nl = neighbours(pos, H, m.rc_d)
    d, cache = descriptor(m, pos, types, None, nl)
    h, dhdd = network(m, d, types)
    nlp = neighbours(pos, H, m.rc_p)
    nlK = neighbours(pos, H, rcK)
    q = np.zeros(N)
    for a in range(N):
        ib, v, r = nlp[a]
        ph, _ = phi_of(m, types[a], types[ib], r)
        q[a] = (ph * (h[a] - h[ib])).sum()
    # D_a = dE_el/dq_a (here from the toy K; in GPUMD from the long-range solver)
    D = S0 * q.copy()
    for a in range(N):
        ib, v, r = nlK[a]
        K, _ = Kfun(r, rcK)
        D[a] += (q[ib] * K).sum()
    # weight of the descriptor force kernels
    if old_weight:
        wt = D - D.mean()
    else:
        wt = np.zeros(N)
        for a in range(N):
            ib, v, r = nlp[a]
            ph, _ = phi_of(m, types[a], types[ib], r)
            wt[a] = (ph * (D[a] - D[ib])).sum()
    F = np.zeros((N, 3))
    Wd = np.zeros((3, 3))
    for a in range(N):                      # descriptor part: force-like with weight wt_a, own rc_desc loop
        ib, v, r = nl[a]
        g_r, dg_r, g_a, dg_a, S = cache[a]
        u = v / r[:, None]
        G = np.einsum("n,nj,jx->jx", dhdd[a, :m.nrad], dg_r, u)
        Sv = v @ S.T
        coef = dhdd[a, m.nrad:]
        G += 2 * np.einsum("m,mj,jm,jx->jx", coef, dg_a, Sv, u)
        G += 2 * np.einsum("m,mj,mx->jx", coef, g_a, S)
        for j in range(len(ib)):
            F[ib[j]] -= wt[a] * G[j]
        F[a] += wt[a] * G.sum(0)
        Wd -= wt[a] * np.einsum("ja,jb->ab", v, G)
    Wc = np.zeros((3, 3))
    Fc = np.zeros((N, 3))
    if ct_pair and not old_weight:           # ct pair term, own rc_phi loop
        for k in range(N):
            ib, v, r = nlp[k]
            _, dph = phi_of(m, types[k], types[ib], r)
            g = dph * (h[k] - h[ib]) * (D[k] - D[ib])
            Fc[k] += (g[:, None] * v / r[:, None]).sum(0)
            Wc -= 0.5 * np.einsum("j,ja,jb->ab", g / r, v, v)
    Wk = np.zeros((3, 3))
    Fk = np.zeros((N, 3))
    for k in range(N):                       # explicit fixed-q pair force of the solver (toy K), own rcK loop
        ib, v, r = nlK[k]
        _, dK = Kfun(r, rcK)
        g = q[k] * q[ib] * dK
        Fk[k] += (g[:, None] * v / r[:, None]).sum(0)
        Wk -= 0.5 * np.einsum("j,ja,jb->ab", g / r, v, v)
    Ftot = F + Fc + Fk
    return Ftot, Wd + Wc + Wk, dict(Fdesc=F, Fct=Fc, FK=Fk, Wdesc=Wd, Wct=Wc, WK=Wk, q=q, D=D, wt=wt)


def deriv(fun, delta, order):
    """d fun/dx at 0: order 2 = central 3-point, order 4 = 5-point (-f2 + 8 f1 - 8 f-1 + f-2)/(12 delta)."""
    if order == 2:
        return (fun(delta) - fun(-delta)) / (2 * delta)
    return (-fun(2 * delta) + 8 * fun(delta) - 8 * fun(-delta) + fun(-2 * delta)) / (12 * delta)


def fd_forces(m, pos, H, types, rcK, delta=DELTA4, order=4):
    F = np.zeros_like(pos)
    for k in range(len(pos)):
        for al in range(3):
            def fun(x):
                p = pos.copy()
                p[k, al] += x
                return energy(m, p, H, types, rcK)
            F[k, al] = -deriv(fun, delta, order)
    return F


def fd_virial(m, pos, H, types, rcK, delta=DELTA4, order=4):
    """W[alpha,beta] = -dE/d eps_(beta,alpha), r -> (1+eps) r, H -> H (1+eps)^T."""
    W = np.zeros((3, 3))
    for be in range(3):
        for al in range(3):
            def fun(x):
                eps = np.zeros((3, 3))
                eps[be, al] = x
                T = (np.eye(3) + eps).T
                return energy(m, pos @ T, H @ T, types, rcK)
            W[al, be] = -deriv(fun, delta, order)
    return W


def main():
    quick = "--quick" in sys.argv
    ok = True
    cases = [("N8 thick", 8, 10.0, 4.0, 4.0, True),
             ("N45 thick", 45, 12.0, 4.5, 4.0, True),
             ("N59 thick", 59, 14.0, 5.0, 5.0, True),
             ("N12 thin", 12, 7.5, 3.6, 3.7, True),
             ("N59 big L=22", 59, 22.0, 3.5, 3.5, True),
             ("N10 images rc>L/2", 10, 6.0, 4.0, 4.5, False)]
    if quick:
        cases = [cases[0], cases[3]]
    print("case | rcK | max|F_an-F_FD4| | max|W_an-W_FD4| | max|sumF| | RMS F | RMS W | asym W | "
          "RMS(Fold-FD) | RMS(Fold-FD, desc w/ Dt, no pair) | RMS(ct pair F) | gate")
    for i, (name, N, L, rd, rp, gated) in enumerate(cases):
        pos, types = random_config(100 + i, N, L)
        m = Model(7 + i, rd, rp)
        H = L * np.eye(3)
        rcK = min(4.0, 0.49 * L)
        F, W, P = analytic(m, pos, H, types, rcK)
        Ff, Wf = fd_forces(m, pos, H, types, rcK), fd_virial(m, pos, H, types, rcK)
        Fo, _, _ = analytic(m, pos, H, types, rcK, old_weight=True)
        Fn, _, _ = analytic(m, pos, H, types, rcK, ct_pair=False)
        eF, eW = mx(F - Ff), mx(W - Wf)
        gap = cutoff_gap(m, pos, H, rcK)
        reach = 2 * DELTA4 * max(1.0, rd, rp, rcK)   # largest change of any pair distance inside the 5-point stencil (strain: delta r)
        eF2 = mx(F - fd_forces(m, pos, H, types, rcK, 1e-5, 2))
        eW2 = mx(W - fd_virial(m, pos, H, types, rcK, 1e-5, 2))
        g = (eF <= 1e-8 and eW <= 1e-8 and mx(F.sum(0)) <= 1e-10) if gated else None
        ok &= g is not False
        print(f"{name:18s} {rcK:.2f} {eF:.1e} {eW:.1e} (2nd order d=1e-5: {eF2:.1e} {eW2:.1e}) {mx(F.sum(0)):.1e} {rms(Ff):.3f} {rms(Wf):.3f} {mx(W - W.T):.1e} "
              f"{rms(Fo - Ff):.3f} {rms(Fn - Ff):.3f} {rms(P['Fct']):.3f} cutoff gap {gap:.1e} (conservative stencil reach {reach:.1e}) gate={'PASS' if g else ('FAIL' if g is False else 'info')}  "
              f"[|sum q| {abs(P['q'].sum()):.0e}, W parts: desc {rms(P['Wdesc']):.2f} ct {rms(P['Wct']):.2f} K {rms(P['WK']):.2f}]", flush=True)
    # invariances on a cluster in a big box
    L = 60.0
    H = L * np.eye(3)
    pos, types = random_config(5, 20, 7.0)
    pos = pos + 26.5
    m = Model(3, 4.0, 4.0)
    rcK = 4.0
    F0, W0, _ = analytic(m, pos, H, types, rcK)
    rng = np.random.default_rng(1)
    Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Qm *= np.sign(np.linalg.det(Qm))
    c0 = pos.mean(0)
    Fr, Wr, _ = analytic(m, (pos - c0) @ Qm.T + c0, H, types, rcK)
    e_rot = max(mx(Fr - F0 @ Qm.T), mx(Wr - Qm @ W0 @ Qm.T))
    Ft, Wt, _ = analytic(m, pos + rng.uniform(-30, 30, 3), H, types, rcK)
    e_tr = max(mx(Ft - F0), mx(Wt - W0))
    ps = pos.copy()
    ps[0] += np.array([L, 0, 0])
    ps[3] -= np.array([0, 2 * L, L])
    Fi, Wi, _ = analytic(m, ps, H, types, rcK)
    e_img = max(mx(Fi - F0), mx(Wi - W0))
    perm = rng.permutation(len(pos))
    Fp, Wp, _ = analytic(m, pos[perm], H, types[perm], rcK)
    e_perm = max(mx(Fp - F0[perm]), mx(Wp - W0))
    e_sum = mx(F0.sum(0))
    eF = mx(F0 - fd_forces(m, pos, H, types, rcK))
    eW = mx(W0 - fd_virial(m, pos, H, types, rcK))
    print(f"G3 cluster L=60: sumF {e_sum:.1e}  rotation {e_rot:.1e}  translation {e_tr:.1e}  image shift {e_img:.1e}  permutation {e_perm:.1e}  "
          f"FD force {eF:.1e}  FD virial {eW:.1e}")
    ok &= max(e_sum, e_rot, e_tr, e_img, e_perm) <= 1e-12 * 1e2 and eF <= 1e-8 and eW <= 1e-8
    print("ALL GATES", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
