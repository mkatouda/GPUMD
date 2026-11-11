"""VER-0028 stage 1-b and 1-d: rebuild the qNEP BEC from the finite-difference Jacobian of the RAW charges
(J.npy, q0.npy from bec_code_formula.py run with a scratch binary in which zero_total_charge is disabled) and compare with the
BEC of the real gpumd (neutralized charge). rel_{ak} = (r_a - r_k)_MIC; index b[3*beta+alpha] = dP_beta/dr_alpha.
  T1[k] = sum_j rel_jk x dq_k/dr_j   T2[k] = sum_a rel_ak x dq_a/dr_k   (raw charges)
  H0 : s*(qn_k*1 + 0.5 T1 + 0.5 T2)                   (current kernel formula, qn = q - mean q)
  H1 : H0 + s*(-(1/N) sum_a rel_ak x dM/dr_k)         (adds the chain-rule term of the neutralization, M = sum raw q)
       equivalently the exact gradient of P' = sum_a (q_a - mean q) r_a with the neutralized charges (MIC reference at k).
  1-d candidates (all scaled by s = sqrt(eps_inf)):
    Zloc   = qn*1 + T2'          T2' with the neutralized Jacobian J_n = J - S/N   (per-pair MIC, reference at k, with neutralization)
    Zloc_r = q *1 + T2           (raw Jacobian, no neutralization term; the charge on the diagonal raw)
    Zloc_n = qn*1 + T2           (neutralized charge on the diagonal, raw Jacobian: neutralization only in the diagonal)
    Zsym   = H0 (the kernel)
Usage: bec_hypotheses.py MODEL.xyz RAWDIR GPUMD_BEC.npy SQRT_EPS [workdir]
 GPUMD_BEC.npy: (N,9) BEC of the real binary (1-step run), q of the real binary is taken as q_raw - mean(q_raw)."""
import sys, numpy as np
from ase.io import read

model, rawdir, becf, s = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
a = read(model); N = len(a)
J = np.load(rawdir + "/J.npy"); q = np.load(rawdir + "/q0.npy"); bg = np.load(becf)
cell = a.cell.array; inv = np.linalg.inv(cell); pos = a.get_positions()
d = pos[:, None, :] - pos[None, :, :]
f = d @ inv; f -= np.round(f); rel = f @ cell  # rel[a,k,b] = (r_a - r_k)_MIC
qn = q - q.mean()
S = J.sum(0)                 # S[k,al] = dM/dr_k,al (raw)
Jn = J - S[None, :, :] / N   # Jacobian of the neutralized charges
rm = lambda x: float(np.sqrt((x ** 2).mean()))
I3 = np.eye(3)
T1 = np.zeros((N, 9)); T2 = np.zeros((N, 9)); T2n = np.zeros((N, 9)); TS = np.zeros((N, 9))
for k in range(N):
    T2[k] = np.einsum("ab,aA->bA", rel[:, k, :], J[:, k, :]).reshape(9)
    T1[k] = np.einsum("jb,jA->bA", rel[:, k, :], J[k, :, :]).reshape(9)
    T2n[k] = np.einsum("ab,aA->bA", rel[:, k, :], Jn[:, k, :]).reshape(9)
    TS[k] = (-(rel[:, k, :].sum(0))[:, None] * S[k][None, :] / N).reshape(9)
D = lambda qq: np.stack([qq[k] * I3.reshape(9) for k in range(N)])
H0 = s * (D(qn) + 0.5 * T1 + 0.5 * T2)
H1 = H0 + s * TS   # TS = -(1/N)(sum_a rel_ak) x S_k
yy = (bg / s - D(qn)).ravel(); XX = np.stack([T1.ravel(), T2.ravel()], 1); cc = np.linalg.lstsq(XX, yy, rcond=None)[0]
print("fit (BEC/s - qn*1) = a*T1 + b*T2 with RAW-charge J: a, b = %.3f %.3f  resid RMS %.4f e (cf. 1-a: fit with neutralized J)" % (cc[0], cc[1], rm(yy - XX @ cc)))
print("N=%d  sum q_raw = %.4f  mean = %.5f  RMS BEC(GPUMD) = %.4f e  RMS|S| (dM/dr) = %.4f e/A" % (N, q.sum(), q.mean(), rm(bg), rm(S)))
print("RMS(GPUMD - H0) = %.5f e   max|.| %.5f" % (rm(bg - H0), np.abs(bg - H0).max()))
print("RMS(GPUMD - H1) = %.5f e   max|.| %.5f" % (rm(bg - H1), np.abs(bg - H1).max()))
print("RMS(H1 - H0)    = %.5f e" % rm(H1 - H0))
# 1-d candidates. raw J is strictly local (zero beyond rc); Jn = J - S/N is nonlocal (every pair), so its MIC form is ambiguous for far pairs.
T2p = np.zeros((N, 9))   # plain differences of the coordinates as given in the file (unwrapped) = exact gradient of sum qn_a r_a
for k in range(N):
    T2p[k] = np.einsum("ab,aA->bA", d[:, k, :], Jn[:, k, :]).reshape(9)
C1 = s * (D(qn) + T2)
cands = {
    "kernel (GPUMD)": bg,
    "H0 (formula from raw J)": H0,
    "C1 qn*1 + MIC-sum(raw J)": C1,
    "C2 q_raw*1 + MIC-sum(raw J)": s * (D(q) + T2),
    "C3 qn*1 + MIC-sum(Jn), nonlocal": s * (D(qn) + T2n),
    "C4 qn*1 + plain-sum(Jn) (grad sum qn r)": s * (D(qn) + T2p),
}
print("\n--- 1-d: candidate definitions (all x s) vs the current kernel BEC ---")
print("%-44s %12s %9s %14s %12s" % ("candidate", "RMS vs GPUMD", "RMS(Z)", "max|sum_k Z_k|", "tr/3 of sum"))
for n, Z in cands.items():
    tot = Z.reshape(N, 3, 3).sum(0)
    print("%-44s %12.4f %9.4f %14.4f %12.4f" % (n, rm(Z - bg), rm(Z), np.abs(tot).max(), np.trace(tot) / 3))
print("(s*sum q_raw = %.4f; s*sum qn = %.1e. Translation invariance of the dipole current needs sum_k Z_k = s*sum(q on the diagonal)*1)" % (s * q.sum(), s * qn.sum()))
print("RMS(C4 - C1) = %.4f e (nonlocal neutralization term, plain differences)" % rm(cands["C4 qn*1 + plain-sum(Jn) (grad sum qn r)"] - C1))
print("RMS(kernel - C1) = %.4f e;  RMS of 0.5*s*(T1 - T2) = %.4f e" % (rm(bg - C1), rm(0.5 * s * (T1 - T2))))
