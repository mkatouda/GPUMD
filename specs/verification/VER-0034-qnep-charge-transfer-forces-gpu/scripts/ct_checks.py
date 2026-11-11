#!/usr/bin/env python3
"""DR-005 stage 2c (VER-0034): (a) static energy check against a float64 host reciprocal-space sum; (b) larger-N neutrality
and BEC run-to-run / translation-sum statistics.

(a) E(ct) - E(same weights without the charge_transfer line) = E_el[q_ct] - E_el[q_noct]: the network energy
    E_NN is identical (same descriptor, same weights; q_noct = h - mean h), so the difference is the
    electrostatic energy difference of the two charge sets.  E_el is recomputed on the host in float64 with the
    reciprocal-space Ewald sum as implemented by `kspace ewald` for qNEP mode 2 (cubic cell, K_C = 14.399645 eV A, charges from gpumd's own double dump) and compared with
    the difference of the two gpumd total energies.  This checks that the energy gpumd reports is E_el of the
    charges it dumps (and that q itself equals the host formula via ct_bec_fd, VER-0032), not the NN part.
(b) N ~ 3000 atoms, large-box kernels: |sum q|, bit identity / max |dZ| of the BEC of two identical static runs,
    max |sum_k Z_k| (translation invariance of P).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ct_field_tests as ft  # noqa: E402
import ct_forces_fd as fd  # noqa: E402
import make_ct_model as mk  # noqa: E402

K_C = 14.399645351950548  # e^2 / (4 pi eps0) in eV A


def recip_energy(q, pos, L, rc_radial):
    """what `kspace ewald` adds for qNEP mode 2 (src/force/ewald.cu): the reciprocal-space part only (the
    real-space and self parts are absorbed in the network), alpha = pi / rc_radial, k-vectors with
    |k|^2 <= (2 pi alpha)^2, E = K_C (2 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2."""
    alpha = np.pi / rc_radial
    kmax = 2 * np.pi * alpha
    kn = int(kmax * L / (2 * np.pi)) + 2
    ks = 2 * np.pi / L * np.array(np.meshgrid(*[np.arange(-kn, kn + 1)] * 3, indexing="ij")).reshape(3, -1).T
    k2 = np.sum(ks**2, axis=1)
    keep = (k2 > 1e-12) & (k2 <= kmax**2)
    ks, k2 = ks[keep], k2[keep]
    S = (q[None, :] * np.exp(1j * ks @ pos.T)).sum(axis=1)
    return K_C * 2 * np.pi / L**3 * np.sum(np.exp(-k2 / (4 * alpha**2)) / k2 * np.abs(S) ** 2)


def zero_phi(text, nt=2):
    """same model with all phi coefficients 0: q = 0 exactly, so E = E_NN (network energy only)"""
    lines = text.rstrip("\n").split("\n")
    n = nt * (nt + 1) // 2 * (mk.BASIS_RADIAL + 1)
    return "\n".join(lines[:-n] + ["0.0"] * n) + "\n"


def energy_check(gpumd, work, structure, rc):
    """E(ct) - E(ct with phi = 0) = E_el[q_ct] (the NN part cancels exactly); compare with host Ewald."""
    text, pos, types, lat, name = fd.setup(structure, 1, False, rc, rc)
    m = ft.Model(gpumd, work, 1, rc, rc)
    L = lat[0, 0]
    a = fd.run_static(gpumd, work, "e_ct", m.ct, pos, types, lat)
    z = fd.run_static(gpumd, work, "e_zero", zero_phi(m.ct), pos, types, lat)
    b = fd.run_static(gpumd, work, "e_no", m.noct, pos, types, lat)
    dE_gpu = a["energy"] - z["energy"]
    dE_host = recip_energy(a["charge"].astype(float), pos, L, rc)
    dEn_gpu = b["energy"] - z["energy"]
    dEn_host = recip_energy(b["charge"].astype(float), pos, L, rc)  # noct: q = h - mean h, same check on the old path
    q_host, _ = ft.host_P(m.phi, m.rc_phi, pos, types, lat, b["charge"], 2)
    return {"structure": name, "max_abs_q_zero_phi": float(np.max(np.abs(z["charge"]))), "E_el_gpumd_ct": dE_gpu,
            "E_el_host_ct": float(dE_host), "abs_diff_ct": float(abs(dE_gpu - dE_host)),
            "E_el_gpumd_noct": dEn_gpu, "E_el_host_noct": float(dEn_host), "abs_diff_noct": float(abs(dEn_gpu - dEn_host)),
            "max_dq_gpu_host": float(np.max(np.abs(a["charge"] - q_host)))}


def large_n(gpumd, work, natoms=3000, box=59.0, seed=7):
    text = mk.make_model_text(1, 4.0, 3.5, 4.0, True)
    pos, types = mk.make_structure(natoms, box, seed)
    lat = np.eye(3) * box
    r1 = fd.run_static(gpumd, work, "big1", text, pos, types, lat, props="charge bec")
    r2 = fd.run_static(gpumd, work, "big2", text, pos, types, lat, props="charge bec")
    Z = r1["bec"].reshape(natoms, 9)
    return {"N": natoms, "sum_q_double_of_float_dump": float(np.sum(r1["charge"])), "max_abs_q": float(np.max(np.abs(r1["charge"]))),
            "max_abs_dq_run_to_run": float(np.max(np.abs(r1["charge"] - r2["charge"]))),
            "max_abs_dZ_run_to_run": float(np.max(np.abs(r1["bec"] - r2["bec"]))),
            "max_abs_sum_k_Z": float(np.max(np.abs(Z.sum(axis=0)))), "rms_Z": ft.rms(Z)}


if __name__ == "__main__":
    g, w = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    what = sys.argv[3]
    if what == "energy":
        for st, rc in (("thin_n12_L7.5", 4.0), ("thick_n45_L14", 3.5), ("thick_n8_L13", 4.0)):
            print(energy_check(g, w, st, rc))
    else:
        print(large_n(g, w))
