#!/usr/bin/env python3
"""DR-005 stage 2b: gpumd `dump_xyz bec` / `charge` of a charge_transfer qNEP vs finite differences.

Checks, for a synthetic model (make_ct_model.py):
  (1) sum_a q_a = 0 (float dump, and a double-accumulated host sum) and q (gpumd) vs q rebuilt on the
      host from h and phi (h from the same weights without the charge_transfer line:
      q_noct = h - mean(h), only differences of h enter q and P);
  (2) BEC_gpumd / sqrt(eps_inf) vs central finite differences of the single-valued pair-form dipole
      P = sum_a h_a V_a = -sum_pairs phi_ab (h_a - h_b) v_ab (v = r_b + image - r_a, all lattice images),
      Z_k[beta, alpha] = dP_beta/dr_k,alpha, index [3 beta + alpha] (VER-0029).
  (3) only for wide boxes (rc_desc + rc_phi < L/2): the formula of the task text
      Z_k = q_k 1 + sum_a (r_a - r_k)_MIC dq_a/dr_k with the FD Jacobian of gpumd's own charge dump
      (exact only in that regime, VER-0029 section "Findings").
Each static run: potential nep.txt; time_step 0 (positions unchanged); run 1; dump_xyz double.
Usage: ct_bec_fd.py GPUMD_BINARY WORKDIR [--structure NAME|wide] [--delta 0.02] [--max-atoms N]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_ct_model as mk  # noqa: E402


def basis(r, rc, K):
    """radial basis f_k(r) fc(r), k < K (NEP form), value and derivative, r array."""
    x = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    dx = 4.0 * (r / rc - 1.0) / rc
    T, dT = [np.ones_like(x), x], [np.zeros_like(x), np.ones_like(x)]
    for k in range(1, K):
        T.append(2 * x * T[k] - T[k - 1])
        dT.append(2 * T[k] + 2 * x * dT[k] - dT[k - 1])
    c = 0.5 * (np.cos(np.pi * r / rc) + 1.0)
    f = np.array([0.5 * (T[k] + 1.0) for k in range(K)])
    return f * c


def read_model(path, nt=mk.NUM_TYPES):
    """phi coefficients (num_pairs, K), sqrt_eps_inf, rc_phi, rc_radial from a legacy nep.txt."""
    lines = open(path).read().split("\n")
    rc_radial = float(lines[1].split()[1])
    rc_phi = float(lines[2].split()[1]) if lines[2].startswith("charge_transfer") else None
    first = 7 if rc_phi is not None else 6
    vals = np.array([float(x) for x in lines[first:] if x.strip()])
    num_L = mk.L_MAX + mk.HAS_Q_222 + mk.HAS_Q_1111
    dim = (mk.N_MAX_RADIAL + 1) + (mk.N_MAX_ANGULAR + 1) * num_L
    sqrt_eps = vals[nt * (dim + 3) * mk.NEURONS]
    K = mk.BASIS_RADIAL + 1
    npair = nt * (nt + 1) // 2
    phi = vals[-npair * K:].reshape(npair, K) if rc_phi is not None else None
    return phi, sqrt_eps, rc_phi, rc_radial


def pair_neighbours(pos, L, rc):
    """for each atom a: list of (b, v) with v = r_b + s - r_a, 0 < |v| < rc, all images."""
    n = int(np.ceil(rc / L))
    rr = np.arange(-n, n + 1)
    sh = np.array(np.meshgrid(rr, rr, rr, indexing="ij")).reshape(3, -1).T * L
    d = pos[None, :, :] - pos[:, None, :]
    out = []
    for a in range(len(pos)):
        v = d[a][:, None, :] + sh[None, :, :]
        r = np.linalg.norm(v, axis=2)
        b, s = np.where((r > 1e-9) & (r < rc))
        out.append((b, v[b, s]))
    return out


def q_and_P(phi_coef, rc_phi, pos, types, L, h, nt=mk.NUM_TYPES):
    """q_a = sum_b phi (h_a - h_b), V_a = sum_b phi (r_a - r_b), P = sum_a h_a V_a (float64)."""
    K = phi_coef.shape[1]
    q = np.zeros(len(pos))
    V = np.zeros((len(pos), 3))
    for a, (b, v) in enumerate(pair_neighbours(pos, L, rc_phi)):
        r = np.linalg.norm(v, axis=1)
        f = basis(r, rc_phi, K)
        ph = np.empty(len(b))
        for i, bb in enumerate(b):
            ph[i] = phi_coef[mk.pair_index(types[a], types[bb], nt)] @ f[:, i]
        q[a] = np.sum(ph * (h[a] - h[b]))
        V[a] = -np.sum(ph[:, None] * v, axis=0)
    return q, V.T @ h


def run_gpumd(gpumd, workdir, model_text, pos, types, box, tag):
    d = os.path.join(workdir, tag)
    shutil.rmtree(d, ignore_errors=True)  # dump_xyz appends: always start from an empty directory
    os.makedirs(d)
    open(os.path.join(d, "nep.txt"), "w").write(model_text)
    mk.write_extxyz(os.path.join(d, "model.xyz"), pos, types, box)
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\nkspace ewald\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 d.xyz precision double charge bec\nrun 1\n"
    )
    r = subprocess.run([gpumd], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gpumd failed in %s:\n%s" % (d, (r.stdout + r.stderr)[-600:]))
    L = open(os.path.join(d, "d.xyz")).read().split("\n")
    n = int(L[0])
    props = L[1].split("Properties=")[1].split()[0].split(":")
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    data = [l.split() for l in L[2:2 + n]]
    get = lambda name: np.array([[float(x) for x in row[col[name][0]:col[name][0] + col[name][1]]] for row in data])
    return get("charge")[:, 0], get("bec")


def compare(gpumd, workdir, seed, pos, types, box, delta, atoms, rc_radial=4.0, rc_phi=4.0,
            wide_check=False, nt=mk.NUM_TYPES, return_arrays=False):
    ct_text = mk.make_model_text(seed, rc_radial, 3.5, rc_phi, True, nt)
    no_text = mk.make_model_text(seed, rc_radial, 3.5, rc_phi, False, nt)
    os.makedirs(workdir, exist_ok=True)
    open(os.path.join(workdir, "ct_nep.txt"), "w").write(ct_text)
    phi, sqrt_eps, rc_phi, _ = read_model(os.path.join(workdir, "ct_nep.txt"), nt)
    q_ct, bec = run_gpumd(gpumd, workdir, ct_text, pos, types, box, "base_ct")
    h, _ = run_gpumd(gpumd, workdir, no_text, pos, types, box, "base_noct")
    q_host, _ = q_and_P(phi, rc_phi, pos, types, box, h, nt)
    res = {
        "sum_q_float": float(np.sum(q_ct.astype(np.float32))),
        "sum_q_double": float(np.sum(q_ct.astype(np.float64))),
        "max_abs_q": float(np.max(np.abs(q_ct))),
        "max_abs_q_gpu_minus_host": float(np.max(np.abs(q_ct - q_host))),
    }
    Z_gpu = bec[:, :9] / sqrt_eps
    err, nrm, errs_wide = [], [], []
    for k in atoms:
        Z_fd = np.zeros(9)
        Jq = np.zeros((len(pos), 3))  # only used for the wide check
        dq = np.zeros((3, len(pos)))
        for alpha in range(3):
            P = []
            qs = []
            for sign in (+1, -1):
                p2 = pos.copy()
                p2[k, alpha] += sign * delta
                hh, _ = run_gpumd(gpumd, workdir, no_text, p2, types, box, "fd_noct")
                _, Pv = q_and_P(phi, rc_phi, p2, types, box, hh, nt)
                P.append(Pv)
                if wide_check:
                    qq, _ = run_gpumd(gpumd, workdir, ct_text, p2, types, box, "fd_ct")
                    qs.append(qq)
            Zcol = (P[0] - P[1]) / (2 * delta)
            for beta in range(3):
                Z_fd[3 * beta + alpha] = Zcol[beta]
            if wide_check:
                dq[alpha] = (qs[0] - qs[1]) / (2 * delta)
        err.append(Z_gpu[k] - Z_fd)
        nrm.append(Z_fd)
        if wide_check:
            Zw = np.zeros(9)
            for a in range(len(pos)):
                dr = pos[a] - pos[k]
                dr -= box * np.round(dr / box)
                for beta in range(3):
                    for alpha in range(3):
                        Zw[3 * beta + alpha] += dr[beta] * dq[alpha][a]
            for beta in range(3):
                Zw[3 * beta + beta] += q_ct[k]
            errs_wide.append(Z_gpu[k] - Zw)
    err, nrm = np.array(err), np.array(nrm)
    res["rms_bec_minus_fd_pair_P"] = float(np.sqrt(np.mean(err**2)))
    res["max_abs_bec_minus_fd_pair_P"] = float(np.max(np.abs(err)))
    res["rms_Z"] = float(np.sqrt(np.mean(nrm**2)))
    res["rel_rms_bec_minus_fd"] = res["rms_bec_minus_fd_pair_P"] / res["rms_Z"]
    if wide_check:
        res["rms_bec_minus_fd_gpumd_charge"] = float(np.sqrt(np.mean(np.array(errs_wide) ** 2)))
    if return_arrays:
        res["arrays"] = (q_ct, Z_gpu)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gpumd")
    ap.add_argument("workdir")
    ap.add_argument("--structure", default="thick_n8_L12", help="a STRUCTURES name or 'wide' (n59, L18)")
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--max-atoms", type=int, default=0, help="0 = all atoms, else evenly spaced subset")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rc-radial", type=float, default=4.0)
    ap.add_argument("--rc-phi", type=float, default=4.0)
    a = ap.parse_args(argv)
    if a.structure == "wide":
        name, n, box, idx = "wide_n59_L18", 59, 18.0, 99
    else:
        idx, (name, n, box) = next((i, s) for i, s in enumerate(mk.STRUCTURES) if s[0] == a.structure)
    pos, types = mk.make_structure(n, box, a.seed * 100 + idx)
    atoms = list(range(n))
    if a.max_atoms and a.max_atoms < n:
        atoms = [int(x) for x in np.linspace(0, n - 1, a.max_atoms)]
    res = compare(os.path.abspath(a.gpumd), os.path.abspath(a.workdir), a.seed, pos, types, box,
                  a.delta, atoms, rc_radial=a.rc_radial, rc_phi=a.rc_phi, wide_check=(a.structure == "wide"))
    path = "small-box" if box <= 2.5 * (a.rc_radial + 1.0) else "large-box"
    print(name, "rc_radial", a.rc_radial, "rc_phi", a.rc_phi, path, "atoms checked:", len(atoms), "delta", a.delta)
    for k, v in res.items():
        print("  %-34s %.3e" % (k, v))


if __name__ == "__main__":
    main()
