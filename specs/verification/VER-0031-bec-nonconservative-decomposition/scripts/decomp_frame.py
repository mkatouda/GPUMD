"""VER-0031: per-frame decomposition of BEC/s - C4 into W (0.5/0.5 asymmetry) and S_term (non-local total-charge conservation).
Usage: decomp_frame.py TRAJ.xyz FRAME_INDEX NEP.txt RAW_GPUMD REAL_GPUMD OUTNPZ [delta=0.02] [workers=4] [WORKDIR]
TRAJ.xyz: dump_xyz with unwrapped_position, velocity (A/fs as written by gpumd: dump_xyz converts natural units to A/fs), charge.
RAW_GPUMD: scratch binary with both zero_total_charge launches disabled (raw charges); REAL_GPUMD: unmodified build.
Index convention: b[3*beta+alpha] = dP_beta/dr_alpha; X[k,beta] = sum_alpha Z[k,beta,alpha] v[k,alpha] (e*A/fs, per atom).
All quantities are WITHOUT the factor s=sqrt_epsilon_inf (BEC/s = qn*1 + 0.5 T1(J) + 0.5 T2(J))."""
import sys, subprocess, shutil, tempfile, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ase.io import read, write
traj, fi, nep, RAW, REAL, outf = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
dl = float(sys.argv[7]) if len(sys.argv) > 7 else 0.02
nw = int(sys.argv[8]) if len(sys.argv) > 8 else 4
base = Path(sys.argv[9]) if len(sys.argv) > 9 else Path(tempfile.mkdtemp())
fr = read(traj, index=fi); N = len(fr)
u = fr.arrays["unwrapped_position"].copy(); v = fr.arrays["vel"].copy()
a0 = fr.copy(); a0.set_pbc(True); a0.arrays["group"] = np.zeros(N, dtype=int)
for k in ("unwrapped_position", "vel", "charge", "initial_charges"):
    a0.arrays.pop(k, None)
ext = np.ptp(u, axis=0)  # intact-molecule check (plain differences = MIC if extent < L - rc)
def run(GP, a, tag):
    d = base / tag; d.mkdir(parents=True, exist_ok=True); shutil.copy(nep, d / "nep.txt"); write(d / "model.xyz", a, format="extxyz")
    (d / "run.in").write_text("potential nep.txt\nkspace pppm\ntime_step 0.000001\nensemble nve\ndump_xyz 1 f.xyz precision double charge bec\nrun 1\n")
    subprocess.run(GP, cwd=d, capture_output=True, text=True)
    M = np.array([l.split()[1:] for l in (d / "f.xyz").read_text().split("\n")[2:2 + N]], dtype=float)
    shutil.rmtree(d); return M[:, 3], M[:, 4:13]
q_real, bec_real = run(REAL, a0, "real")
q_raw, _ = run(RAW, a0, "rawbase")
def task(args):
    k, al, sg = args; a = a0.copy(); p = a.get_positions(); p[k, al] += sg * dl; a.set_positions(p); return run(RAW, a, f"t{k}_{al}_{sg}")[0]
jobs = [(k, al, sg) for k in range(N) for al in range(3) for sg in (1, -1)]
with ThreadPoolExecutor(nw) as ex: res = list(ex.map(task, jobs))
J = np.zeros((N, N, 3))  # J[a,k,al] = dq_a/dr_k,al (raw)
for (k, al, sg), q in zip(jobs, res): J[:, k, al] += sg * q / (2 * dl)
S = J.sum(0); Jn = J - S[None] / N
d = u[:, None, :] - u[None, :, :]  # d[a,k] = u_a - u_k (plain, unwrapped)
T1 = np.zeros((N, 9)); T2 = np.zeros((N, 9)); T2n = np.zeros((N, 9))
for k in range(N):
    T2[k] = np.einsum("ab,aA->bA", d[:, k, :], J[:, k, :]).reshape(9)
    T1[k] = np.einsum("jb,jA->bA", d[:, k, :], J[k, :, :]).reshape(9)  # sum_j rel_jk x dq_k/dr_j, rel_jk = u_j - u_k
    T2n[k] = np.einsum("ab,aA->bA", d[:, k, :], Jn[:, k, :]).reshape(9)
qn = q_raw - q_raw.mean(); I9 = np.eye(3).reshape(9)[None]
Z_ker = qn[:, None] * I9 + 0.5 * T1 + 0.5 * T2     # kernel formula, no s
C4 = qn[:, None] * I9 + T2n
Wm = 0.5 * (T1 - T2); Sm = T2 - T2n
work = lambda Z: np.einsum("kba,ka->kb", Z.reshape(N, 3, 3), v)  # (N,3)
meas = bec_real / 0.989073634 - C4                 # directly measured (real-binary BEC/s - C4)
np.savez(outf, J=J, q_raw=q_raw, q_real=q_real, bec_real=bec_real, u=u, v=v, T1=T1, T2=T2, T2n=T2n, delta=dl, extent=ext,
         xW=work(Wm), xS=work(Sm), xmeas=work(meas), xbec=work(bec_real / 0.989073634), xC4=work(C4), xker=work(Z_ker))
rm = lambda x: float(np.sqrt((x ** 2).mean()))
print("frame %d delta %.3f extent %s  max|q_real-qn| %.1e  RMS(BEC/s - kernel(J)) %.2e  RMS|S| %.3f" % (fi, dl, np.round(ext, 1), np.abs(q_real - qn).max(), rm(bec_real / 0.989073634 - Z_ker), rm(S)))
