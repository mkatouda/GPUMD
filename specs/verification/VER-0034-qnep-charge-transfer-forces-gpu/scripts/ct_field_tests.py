#!/usr/bin/env python3
"""DR-005 stage 2c (VER-0034): electric-field consistency of the charge_transfer qNEP BEC.

P_direct(r) = sum_a h_a V_a = -sum_pairs phi_ab (h_a - h_b) v_ab (all lattice images, float64 on the host) is the
dipole whose derivative Z[3 beta + alpha] = dP_beta/dr_alpha is the BEC (VER-0029/VER-0032).  h is taken from gpumd's
charge dump of the SAME weights without the charge_transfer line (q_noct = h - mean h; P depends on differences of h
only).  gpumd's `add_efield ... bec` adds F_alpha = sum_beta bec[3 beta + alpha] E_beta with bec = sqrt(eps_inf) Z
(src/measure/add_efield.cu), i.e. F = grad_r (sqrt(eps_inf) E.P_direct); thermo energy does not contain -E.P.
Tests: (1) field force of add_efield vs central FD of sqrt(eps) E.P_direct (also Z vs FD);
(2) NVE with a constant field: K + U - sqrt(eps) E.P_direct(r_t) is conserved;
(3) closed-loop work: integral over a closed smooth loop of sum_k Z_k . dr_k = 0 (trapezoid rule on a periodic
integrand), contrasted with the OLD qNEP BEC.
"""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "VER-0032-qnep-charge-transfer-gpu", "scripts"))
import ct_bec_fd as bec  # noqa: E402
import ct_forces_fd as fd  # noqa: E402
import make_ct_model as mk  # noqa: E402


def neighbours_lat(pos, lat, rc):
    """for each atom a: (b, v), v = r_b + s - r_a with 0 < |v| < rc over all lattice images (any cell)."""
    lat = np.asarray(lat, float)
    frac = pos @ np.linalg.inv(lat)
    pos = (frac - np.floor(frac)) @ lat  # wrap into the cell
    vol = abs(np.linalg.det(lat))
    thick = [vol / np.linalg.norm(np.cross(lat[(i + 1) % 3], lat[(i + 2) % 3])) for i in range(3)]
    n = [int(np.ceil(rc / t)) + 1 for t in thick]
    rng = [np.arange(-k, k + 1) for k in n]
    sh = np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T @ lat
    d = pos[None, :, :] - pos[:, None, :]
    out = []
    for a in range(len(pos)):
        v = d[a][:, None, :] + sh[None, :, :]
        r = np.linalg.norm(v, axis=2)
        b, s = np.where((r > 1e-9) & (r < rc))
        out.append((b, v[b, s]))
    return out


def host_P(phi_coef, rc_phi, pos, types, lat, h, nt):
    """q (host, float64) and P_direct = sum_a h_a V_a."""
    K = phi_coef.shape[1]
    q = np.zeros(len(pos))
    V = np.zeros((len(pos), 3))
    for a, (b, v) in enumerate(neighbours_lat(pos, lat, rc_phi)):
        r = np.linalg.norm(v, axis=1)
        f = bec.basis(r, rc_phi, K)
        ph = np.array([phi_coef[mk.pair_index(types[a], types[bb], nt)] @ f[:, i] for i, bb in enumerate(b)])
        q[a] = np.sum(ph * (h[a] - h[b]))
        V[a] = -np.sum(ph[:, None] * v, axis=0)
    return q, V.T @ h


class Model:
    """a ct model and its no-line twin (same weights) with host helpers"""

    def __init__(self, gpumd, work, seed, rc_radial, rc_phi, nt=2, mode=2):
        self.gpumd, self.work, self.nt = gpumd, work, nt
        self.ct = mk.make_model_text(seed, rc_radial, 3.5, rc_phi, True, nt, mode)
        self.noct = mk.make_model_text(seed, rc_radial, 3.5, rc_phi, False, nt, mode)
        os.makedirs(work, exist_ok=True)
        open(os.path.join(work, "ct_nep.txt"), "w").write(self.ct)
        self.phi, self.sqrt_eps, self.rc_phi, _ = bec.read_model(os.path.join(work, "ct_nep.txt"), nt)

    def h(self, pos, types, lat, tag="h"):
        return fd.run_static(self.gpumd, self.work, tag, self.noct, pos, types, lat)["charge"]

    def P(self, pos, types, lat):
        return host_P(self.phi, self.rc_phi, pos, types, lat, self.h(pos, types, lat), self.nt)[1]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def field_force_fd(m, pos, types, lat, E, atoms, delta=0.02):
    """gpumd add_efield force and gpumd bec vs FD of sqrt(eps) E.P_direct / of Z"""
    E = np.asarray(E, float)
    r0 = fd.run_static(m.gpumd, m.work, "f0", m.ct, pos, types, lat, props="charge force bec")
    rE = fd.run_static(m.gpumd, m.work, "fE", m.ct, pos, types, lat, props="charge force bec", efield=E)
    F_field = rE["forces"] - r0["forces"]
    errF, errZ, ref, refZ = [], [], [], []
    for k in atoms:
        Z = np.zeros(9)
        for a in range(3):
            P = []
            for s in (+1, -1):
                p2 = pos.copy()
                p2[k, a] += s * delta
                P.append(m.P(p2, types, lat))
            col = (P[0] - P[1]) / (2 * delta)
            for b in range(3):
                Z[3 * b + a] = col[b]
        F_fd = m.sqrt_eps * (E @ Z.reshape(3, 3))  # F_alpha = sum_beta E_beta Z[3 beta + alpha]
        errF.append(F_field[k] - F_fd)
        errZ.append(r0["bec"][k] / m.sqrt_eps - Z)
        ref.append(F_fd)
        refZ.append(Z)
    return {"rms_Ffield": rms(ref), "rms_err_Ffield": rms(errF), "max_err_Ffield": float(np.max(np.abs(errF))),
            "rms_Z": rms(refZ), "rms_err_Z": rms(errZ)}


def nve_field(m, pos, types, lat, E, steps=1000, dt=0.5, every=20, tag="nvef"):
    """NVE with add_efield; returns K+U and K+U-sqrt(eps) E.P_direct(r_t) at each thermo frame"""
    d = os.path.join(m.work, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "nep.txt"), "w").write(m.ct)
    fd.write_xyz(os.path.join(d, "model.xyz"), pos, types, lat, group=True)
    E = np.asarray(E, float)
    ef = "add_efield 0 0 %.15g %.15g %.15g bec\n" % tuple(E) if np.any(E != 0) else ""
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\nkspace ewald\nvelocity 300\nensemble nve\ntime_step %g\n%s"
        "dump_thermo %d\ndump_xyz %d d.xyz precision double\nrun %d\n" % (dt, ef, every, every, steps))
    r = subprocess.run([m.gpumd], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-600:] + r.stderr[-300:])
    th = np.loadtxt(os.path.join(d, "thermo.out"))
    etot = th[:, 1] + th[:, 2]
    n = len(pos)
    lines = open(os.path.join(d, "d.xyz")).read().split("\n")
    Hc = []
    for f in range(len(etot)):
        rows = [l.split() for l in lines[f * (n + 2) + 2: f * (n + 2) + 2 + n]]
        p = np.array([[float(x) for x in row[1:4]] for row in rows])
        Hc.append(etot[f] - m.sqrt_eps * np.dot(E, m.P(p, types, lat)))
    return etot, np.array(Hc)


def closed_loop(m, pos, types, lat, old_text=None, amp=0.15, npts=32, seed=0):
    """loop integral of sum_k Z_k . dr_k (one value per E direction beta) and the scale of its integrand;
    Z = dumped bec / sqrt(eps).  r_k(theta) = r_k + A (cos - 1) + B sin is closed at theta = 2 pi."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0, amp, pos.shape)
    B = rng.normal(0, amp, pos.shape)
    out = {}
    for name, text in (("ct", m.ct),) + ((("old", old_text),) if old_text else ()):
        I = np.zeros(3)
        S = np.zeros(3)
        for i in range(npts):
            th = 2 * np.pi * i / npts
            r = pos + A * (np.cos(th) - 1) + B * np.sin(th)
            dr = -A * np.sin(th) + B * np.cos(th)
            Z = fd.run_static(m.gpumd, m.work, "loop", text, r, types, lat, props="charge bec")["bec"] / m.sqrt_eps
            integrand = np.einsum("kba,ka->b", Z.reshape(len(pos), 3, 3), dr)
            I += integrand * 2 * np.pi / npts
            S += np.abs(integrand) * 2 * np.pi / npts
        out[name] = (I, S)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("gpumd")
    ap.add_argument("workdir")
    ap.add_argument("test", choices=["force", "nve", "loop"])
    ap.add_argument("--structure", default="thin_n12_L7.5")
    ap.add_argument("--rc", type=float, default=4.0)
    ap.add_argument("--mode", type=int, default=2)
    ap.add_argument("--triclinic", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    gpumd, work = os.path.abspath(a.gpumd), os.path.abspath(a.workdir)
    text, pos, types, lat, name = fd.setup(a.structure, 1, False, a.rc, a.rc, a.mode, a.triclinic)
    m = Model(gpumd, work, 1, a.rc, a.rc, 2, a.mode)
    E = [0.3, -0.2, 0.5]
    tag = "%s%s mode%d rc%g" % (name, " tri" if a.triclinic else "", a.mode, a.rc)
    if a.test == "force":
        atoms = list(np.linspace(0, len(pos) - 1, min(6, len(pos))).astype(int))
        print(tag, field_force_fd(m, pos, types, lat, E, atoms))
    elif a.test == "nve":
        for lab, EE in (("E=0", [0, 0, 0]), ("E=%s" % E, E)):
            etot, H = nve_field(m, pos, types, lat, EE, a.steps)
            print(tag, lab, "K+U: drift(fit) %.3e span %.3e | K+U-sqrt(eps)E.P: drift(fit) %.3e span %.3e" % (
                np.polyfit(np.arange(len(etot)), etot, 1)[0] * (len(etot) - 1), np.ptp(etot),
                np.polyfit(np.arange(len(H)), H, 1)[0] * (len(H) - 1), np.ptp(H)))
    else:
        old = mk.make_model_text(1, a.rc, 3.5, a.rc, False, 2, a.mode)
        res = closed_loop(m, pos, types, lat, old)
        for k, (I, S) in res.items():
            print(tag, k, "loop integral", I, "scale", S, "ratio", np.abs(I) / S)


if __name__ == "__main__":
    main()
