"""Integrability test of the GPUMD qNEP BEC: compare the BEC printed by GPUMD (dump_xyz ... charge bec) with the finite-difference derivative of P = sum_a q_a r_a (q from the same model, positions as in the cell, atoms not crossing the boundary).
BEC index convention (add_efield: fx = b0 Ex + b3 Ey + b6 Ez): b[3*beta + alpha] = dP_beta / dr_alpha.
Usage: bec_integrability.py MODEL.xyz NEP.txt OUTDIR [natoms=8] [delta=0.005]"""
import os, sys, subprocess, shutil, numpy as np
from pathlib import Path
from ase.io import read, write
src, nep, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); na = int(sys.argv[4]) if len(sys.argv) > 4 else 8; dl = float(sys.argv[5]) if len(sys.argv) > 5 else 0.005
GP = os.environ.get("GPUMD_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'gpumd')); out.mkdir(parents=True, exist_ok=True); shutil.copy(nep, out / "nep.txt")
a0 = read(src); a0.set_pbc(True); a0.arrays["group"] = np.zeros(len(a0), dtype=int)
def run(a, tag):
    d = out / tag; d.mkdir(exist_ok=True); shutil.copy(out / "nep.txt", d / "nep.txt"); write(d / "model.xyz", a, format="extxyz")
    (d / "run.in").write_text("potential nep.txt\nkspace pppm\ntime_step 0.000001\nensemble nve\ndump_xyz 1 f.xyz precision double charge bec\nrun 1\n")
    r = subprocess.run(GP, cwd=d, capture_output=True, text=True)
    L = (d / "f.xyz").read_text().split("\n")[2:2 + len(a)]   # species x y z charge bec(9)
    M = np.array([l.split()[1:] for l in L], dtype=float)
    return M[:, 3], M[:, 4:13], a.get_positions()
q0, b0, _ = run(a0, "base")
print("sum q = %.2e; sum BEC (diag) = %.2e" % (q0.sum(), np.abs(b0.sum(0)).max()))
rng = np.random.default_rng(1); sym = np.array(a0.get_chemical_symbols())
idx = list(rng.choice(np.where(sym == "C")[0], na // 2, replace=False)) + list(rng.choice(np.where(sym == "F")[0], na - na // 2, replace=False))
rows = []
cell = a0.cell.array; pos0 = a0.get_positions(); inv = np.linalg.inv(cell)
def mic(v):   # minimum image of vectors v (n,3) for the (orthogonal) cell
    f = v @ inv; f -= np.round(f); return f @ cell
for k in idx:
    zfd = np.zeros(9)
    for al in range(3):
        Q = []
        for sg in (+1, -1):
            a = a0.copy(); p = a.get_positions(); p[k, al] += sg * dl; a.set_positions(p)
            q, _, _ = run(a, f"k{k}_{al}_{'p' if sg > 0 else 'm'}"); Q.append(q)
        dq = (Q[0] - Q[1]) / (2 * dl)                     # dq_a / dr_{k,al}
        rel = mic(pos0 - pos0[k])                          # r_a - r_k (minimum image)
        dP = (dq[:, None] * rel).sum(0)                    # sum_a (r_a - r_k) dq_a/dr_k  (valid because sum_a dq_a = 0)
        dP[al] += q0[k]
        for be in range(3): zfd[3 * be + al] = dP[be]
    rows.append((k, sym[k], b0[k], zfd))
    print(k, sym[k], "BEC(GPUMD) diag", np.round(b0[k][[0, 4, 8]], 3), " FD diag", np.round(zfd[[0, 4, 8]], 3), " max|diff| (9 comp) %.3f" % np.abs(b0[k] - zfd).max())
d = np.array([r[2] - r[3] for r in rows]); print("RMS(BEC_GPUMD - BEC_FD) over tested atoms = %.4f e; RMS BEC = %.3f e" % (np.sqrt((d ** 2).mean()), np.sqrt(np.mean([r[2] ** 2 for r in rows]))))
