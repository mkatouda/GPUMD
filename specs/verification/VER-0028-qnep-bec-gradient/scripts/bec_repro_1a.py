"""VER-0028 stage 1-a: reproduce the requester's numbers from J_neut.npy (Jacobian of the neutralized charges as seen by the real gpumd),
bec.npy/q.npy of the real binary, with the same formulas as bec_code_formula.py / bec_formula_fit.py (no sqrt(eps_inf) scaling, as in the requester's scripts).
Usage: bec_repro_1a.py MODEL.xyz WORKDIR"""
import sys, numpy as np
from ase.io import read
a = read(sys.argv[1]); w = sys.argv[2]; N = len(a)
J = np.load(w + "/J_neut.npy"); q = np.load(w + "/q.npy"); b = np.load(w + "/bec.npy")
cell = a.cell.array; inv = np.linalg.inv(cell); pos = a.get_positions()
f = (pos[:, None] - pos[None]) @ inv; f -= np.round(f); rel = f @ cell
T1 = np.zeros((N, 9)); T2 = np.zeros((N, 9)); Dg = np.zeros((N, 9))
for k in range(N):
    T2[k] = np.einsum("ab,aA->bA", rel[:, k], J[:, k]).reshape(9); T1[k] = np.einsum("jb,jA->bA", rel[:, k], J[k]).reshape(9); Dg[k] = (q[k] * np.eye(3)).reshape(9)
rm = lambda x: float(np.sqrt((x ** 2).mean()))
sym = np.array(a.get_chemical_symbols())
print("N=%d RMS BEC %.3f e" % (N, rm(b)))
print("RMS(GPUMD - exact gradient (FD)) = %.4f e   (C %.4f, F %.4f)" % (rm(b - Dg - T2), rm((b - Dg - T2)[sym == "C"]), rm((b - Dg - T2)[sym == "F"])))
print("RMS(GPUMD - code formula 0.5/0.5) = %.4f e" % rm(b - Dg - 0.5 * T1 - 0.5 * T2))
y = (b - Dg).ravel(); X = np.stack([T1.ravel(), T2.ravel()], 1); c = np.linalg.lstsq(X, y, rcond=None)[0]
print("fit b-D = a T1 + b T2: a,b = %.3f %.3f, resid RMS %.4f, y RMS %.4f" % (c[0], c[1], rm(y - X @ c), rm(y)))
