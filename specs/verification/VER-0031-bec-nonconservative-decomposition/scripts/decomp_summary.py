"""VER-0031 summary. Usage: decomp_summary.py RESDIR [delta_tag=d0.02]. Reads RESDIR/fNN_<tag>.npz from decomp_frame.py.
Per frame: RMS over atoms x 3 components of the per-atom work rates x[k,beta] (e*A/fs), and the cell sums (3-vector, e*A/fs)."""
import sys, glob, numpy as np
res, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "d0.02")
fs = sorted(glob.glob("%s/f??_%s.npz" % (res, tag))); rm = lambda x: float(np.sqrt((x ** 2).mean()))
print("frame | RMS(W.v) RMS(S.v) RMS((W+S).v) RMS(meas.v) | RMS(clos) clos/meas | RMS(BEC.v/s) RMS(C4.v) | corr(W.v,S.v) | sumW(x,y,z) sumS(x,y,z)")
rows = []; ok = 0
for f in fs:
    z = np.load(f); W, S, M, B, C = z["xW"], z["xS"], z["xmeas"], z["xbec"], z["xC4"]
    cl = rm(W + S - M); r = cl / rm(M); ok += r <= 0.10
    cor = float(np.corrcoef(W.ravel(), S.ravel())[0, 1]); sW, sS = W.sum(0), S.sum(0)
    rows.append((rm(W), rm(S), rm(W + S), rm(M), cl, r, rm(B), rm(C), cor, *sW, *sS, *(M.sum(0)), *((W + S).sum(0))))
    print("%s | %.4f %.4f %.4f %.4f | %.5f %.4f | %.4f %.4f | %+.3f | %+.4f %+.4f %+.4f  %+.4f %+.4f %+.4f" % (f.split("/")[-1][:3], *rows[-1][:9], *rows[-1][9:15]))
R = np.array(rows); n = len(R)
print("G1 (closure <=10%%): %d of %d frames pass; max ratio %.4f" % (ok, n, R[:, 5].max()))
print("mean RMS: W.v %.4f  S.v %.4f  (W+S).v %.4f  meas %.4f  BEC.v/s %.4f  C4.v %.4f ; mean corr %.3f" % tuple(R[:, [0, 1, 2, 3, 6, 7, 8]].mean(0)))
print("fraction of RMS^2 (mean over frames): W / (W^2+S^2) = %.3f" % np.mean(R[:, 0] ** 2 / (R[:, 0] ** 2 + R[:, 1] ** 2)))
for nm, c in (("sum W.v", 9), ("sum S.v", 12), ("sum meas", 15), ("sum (W+S)", 18)):
    m = R[:, c:c + 3].mean(0); se = R[:, c:c + 3].std(0, ddof=1) / np.sqrt(n)
    print("%-10s mean (x,y,z) = (%+.4f %+.4f %+.4f) +/- SE (%.4f %.4f %.4f) e*A/fs" % (nm, *m, *se))
