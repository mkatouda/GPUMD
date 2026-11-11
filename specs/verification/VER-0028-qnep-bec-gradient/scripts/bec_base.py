"""Run the real gpumd (1 step, dump_xyz charge bec, double precision) and save q.npy (N,), bec.npy (N,9) and, from RAW J.npy
(Jacobian of the raw charges, scratch binary without zero_total_charge), the Jacobian of the neutralized charges J_neut.npy = J - (sum_a J[a])/N.
Usage: bec_base.py MODEL.xyz NEP.txt RAWDIR WORKDIR   (binary: $GPUMD_BIN or <repo>/src/gpumd)"""
import os, sys, subprocess, shutil, numpy as np
from pathlib import Path
from ase.io import read, write
model, nep, rawdir, work = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
GP = os.environ.get("GPUMD_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'gpumd'))
work.mkdir(parents=True, exist_ok=True); shutil.copy(nep, work / "nep.txt")
a = read(model); a.set_pbc(True); a.arrays["group"] = np.zeros(len(a), dtype=int); write(work / "model.xyz", a, format="extxyz")
(work / "run.in").write_text("potential nep.txt\nkspace pppm\ntime_step 0.000001\nensemble nve\ndump_xyz 1 f.xyz precision double charge bec\nrun 1\n")
subprocess.run(GP, cwd=work, capture_output=True, text=True)
N = len(a); M = np.array([l.split()[1:] for l in (work / "f.xyz").read_text().split("\n")[2:2 + N]], dtype=float)
q, b = M[:, 3], M[:, 4:13]; np.save(work / "q.npy", q); np.save(work / "bec.npy", b)
J = np.load(rawdir / "J.npy"); qr = np.load(rawdir / "q0.npy"); Jn = J - J.sum(0)[None] / N; np.save(work / "J_neut.npy", Jn)
print("sum q (real binary) = %.2e;  max|q_real - (q_raw - mean q_raw)| = %.2e;  max|sum_a Jn| = %.2e" % (q.sum(), np.abs(q - (qr - qr.mean())).max(), np.abs(Jn.sum(0)).max()))
