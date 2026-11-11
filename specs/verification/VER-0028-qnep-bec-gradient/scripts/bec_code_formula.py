"""Test of premise 1: is the GPUMD qNEP BEC (nep_charge.cu find_bec_radial/angular) equal to the exact gradient of P = sum_a q_a r_a?
Full finite-difference Jacobian J[a,k,al] = dq_a/dr_{k,al} of all charges (3N central differences), then, with rel_{ak} = (r_a - r_k)_MIC:
  exact   Z_k[b,al] = q_k d_{b,al} + sum_a rel_{ak,b} J[a,k,al]
  code    Z_k[b,al] = q_k d_{b,al} + 0.5 sum_j rel_{jk,b} J[k,j,al] + 0.5 sum_a rel_{ak,b} J[a,k,al]
(code = pair term 0.5 r12 x f12 with f12 = dq_{n1}/dr_{n2}, r12 = r_{n2} - r_{n1}, +0.5 to n1 and -0.5 to n2, as read from the CUDA kernel).
Compares both with the BEC printed by GPUMD. Index: b[3*beta+alpha] = dP_beta/dr_alpha.  Usage: bec_code_formula.py MODEL.xyz NEP.txt OUTDIR [delta=0.005] [workers=6]"""
import os, sys, subprocess, shutil, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ase.io import read, write
src, nep, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); dl = float(sys.argv[4]) if len(sys.argv) > 4 else 0.005; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 6
GP = os.environ.get("GPUMD_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'gpumd')); out.mkdir(parents=True, exist_ok=True)
a0 = read(src); a0.set_pbc(True); a0.arrays["group"] = np.zeros(len(a0), dtype=int); N = len(a0)
def run(a, tag):
    d = out / tag; d.mkdir(exist_ok=True); shutil.copy(nep, d / "nep.txt"); write(d / "model.xyz", a, format="extxyz")
    (d / "run.in").write_text("potential nep.txt\nkspace pppm\ntime_step 0.000001\nensemble nve\ndump_xyz 1 f.xyz precision double charge bec\nrun 1\n")
    subprocess.run(GP, cwd=d, capture_output=True, text=True)
    M = np.array([l.split()[1:] for l in (d / "f.xyz").read_text().split("\n")[2:2 + N]], dtype=float)
    shutil.rmtree(d); return M[:, 3], M[:, 4:13]
q0, b0 = run(a0, "base")
def task(args):
    k, al, sg = args; a = a0.copy(); p = a.get_positions(); p[k, al] += sg * dl; a.set_positions(p); return run(a, f"t{k}_{al}_{sg}")[0]
jobs = [(k, al, sg) for k in range(N) for al in range(3) for sg in (1, -1)]
with ThreadPoolExecutor(nw) as ex: res = list(ex.map(task, jobs))
J = np.zeros((N, N, 3))   # J[a,k,al]
for (k, al, sg), q in zip(jobs, res): J[:, k, al] += sg * q / (2 * dl)
np.save(out / "J.npy", J); np.save(out / "q0.npy", q0); np.save(out / "b0.npy", b0)
cell = a0.cell.array; inv = np.linalg.inv(cell); pos = a0.get_positions()
d = pos[:, None, :] - pos[None, :, :]   # d[a,k] = r_a - r_k
f = d @ inv; f -= np.round(f); rel = f @ cell                 # MIC, rel[a,k,b]
ex_b = np.zeros((N, 9)); co_b = np.zeros((N, 9))
for k in range(N):
    t_exact = np.einsum("ab,aA->bA", rel[:, k, :], J[:, k, :])          # sum_a rel_ak,b J[a,k,al]
    t_first = np.einsum("jb,jA->bA", rel[:, k, :], J[k, :, :])          # sum_j rel_jk,b J[k,j,al]
    diag = q0[k] * np.eye(3)
    ex_b[k] = (diag + t_exact).reshape(9); co_b[k] = (diag + 0.5 * t_first + 0.5 * t_exact).reshape(9)
rm = lambda x: float(np.sqrt((x ** 2).mean()))
print("N=%d atoms; sum q %.1e; RMS BEC (GPUMD) %.3f e" % (N, q0.sum(), rm(b0)))
print("RMS(GPUMD - code-formula(from FD Jacobian)) = %.4f e   <- reading of the kernel verified if ~1e-3 or less" % rm(b0 - co_b))
print("RMS(GPUMD - exact gradient (FD))          = %.4f e" % rm(b0 - ex_b))
print("RMS(code-formula - exact gradient)        = %.4f e" % rm(co_b - ex_b))
for s in ("C", "F"):
    m = np.array(a0.get_chemical_symbols()) == s; print("  %s: RMS(GPUMD-code) %.4f, RMS(GPUMD-exact) %.4f" % (s, rm((b0 - co_b)[m]), rm((b0 - ex_b)[m])))
