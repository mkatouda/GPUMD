"""Check of the D2 `table qe` / `trunc plain` quantities of docs_ja/dftd/A2_derivation_d2.md, A2.8-A2.9.

CPU, NumPy.  (1) pair parameters C-C, C-F, F-F read from src/force/dftd2qe_table.cuh (QE 7.2 values),
(2) the plain-truncation jumps |phi(rc)| and 6|phi(rc)|/rc, (3) dphi vs central FD, (4) emulation of the
float accumulation of the kernel: naive float, TwoSum-compensated float, and compensated float
flushed to double every 4 'cells' (src/force/dftd2.cu, CompensatedSum, FLUSH_CELLS = 4), against the
exact double sum of the SAME float terms (this isolates the accumulation; the rounding of the
individual terms is common to all variants and not part of this emulation).
Run:  python check_d2_qe_plain.py
"""
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
TAB = (REPO / "src" / "force" / "dftd2qe_table.cuh").read_text()
EH, BOHR = 27.211386245988, 0.529177210903          # CODATA 2018 as in QE 7.2 constants.f90
HB6 = EH * BOHR ** 6                                 # eV A^6 per (Eh bohr^6)
D, RS6, S6 = 20.0, 1.1, 0.75


def table(name):
    m = re.search(name + r"\[3741\] = \{(.*?)\};", TAB, re.S)
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()])


C6T, R0T = table("DFTD2QE_C6_EH_BOHR6"), table("DFTD2QE_R0AB_A")


def pair(z1, z2):
    zs, zl = sorted((z1 - 1, z2 - 1))
    k = zl * (zl + 1) // 2 + zs
    return C6T[k], R0T[k]


def phi(r, c6, R0):
    f = 1.0 / (1.0 + np.exp(-D * (r / (RS6 * R0) - 1.0)))
    return -S6 * c6 * HB6 * f / r ** 6


def dphi(r, c6, R0):
    R = RS6 * R0
    e = np.exp(-D * (r / R - 1.0))
    f = 1.0 / (1.0 + e)
    return -S6 * c6 * HB6 * ((D / R) * e * f * f / r ** 6 - 6.0 * f / r ** 7)


def check_params():
    print("[params] pair values from dftd2qe_table.cuh (Eh bohr^6, Angstrom, R0 = 1.1 r0ab):")
    for a, b, name, c6x, r0x in ((6, 6, "C-C", 49.113, 2.9103), (6, 9, "C-F", 20.2827, 2.4770),
                                 (9, 9, "F-F", 9.6916, 2.2996)):
        c6, r0 = pair(a, b)
        print(f"   {name}: C6 = {c6} (request {c6x}), r0ab = {r0} (request {r0x}), R0 = {RS6 * r0:.5f}")
        assert c6 == c6x and r0 == r0x
    print(f"   Eh*a0^6 (CODATA2018) = {HB6:.12f} eV A^6")


def check_plain_step(rc=12.0):
    c6, r0 = pair(6, 6)
    p, dp = phi(rc, c6, r0), dphi(rc, c6, r0)
    print(f"[plain] C-C, s6 = 0.75, rc = {rc} A: phi(rc) = {p:.4e} eV, 6|phi|/rc = {6 * abs(p) / rc:.4e} eV/A, "
          f"dphi(rc) = {dp:.4e} eV/A (ratio dphi*rc/(6|phi|) = {dp * rc / (6 * abs(p)):.6f})")
    # explicit asymptotic value without damping: 0.75 * C6 * HB6 / rc^6
    print(f"   undamped 0.75 C6 Eh a0^6 / rc^6 = {S6 * c6 * HB6 / rc ** 6:.4e} eV")
    worst = 0.0
    rng = np.random.default_rng(7)
    for _ in range(200):
        z1, z2 = rng.choice([1, 6, 7, 8, 9, 14, 17], 2)
        c6, r0 = pair(z1, z2)
        r = rng.uniform(0.7, 2.5) * RS6 * r0 * 2
        h = 1e-6 * r
        fd = (phi(r + h, c6, r0) - phi(r - h, c6, r0)) / (2 * h)
        worst = max(worst, abs(dphi(r, c6, r0) - fd) / abs(fd))
    print(f"   dphi vs central FD, 200 pairs/r: max rel err {worst:.2e}")
    return worst


def compensated(terms, flush_every=None, cell=64):
    """float32 Neumaier/TwoSum accumulation as CompensatedSum; flush to a double total every
    flush_every blocks of `cell` terms (None = never)"""
    s = np.float32(0)
    e = np.float32(0)
    tot = 0.0
    for k, x in enumerate(terms):
        t = np.float32(s + x)
        bb = np.float32(t - s)
        e = np.float32(e + np.float32(np.float32(s - np.float32(t - bb)) + np.float32(x - bb)))
        s = t
        if flush_every and (k + 1) % (flush_every * cell) == 0:
            tot += float(s) + float(e)
            s = np.float32(0)
            e = np.float32(0)
    return tot + float(s) + float(e)


def naive(terms):
    s = np.float32(0)
    for x in terms:
        s = np.float32(s + x)
    return float(s)


def check_accumulation():
    rng = np.random.default_rng(11)
    print("[accumulation] net force of 12000 float terms of both signs (cancelling, like a net force):")
    rows = []
    for trial in range(5):
        v = rng.normal(size=12000) * 10 ** rng.uniform(-6, 0, size=12000)
        v = (v - v.sum() / len(v)).astype(np.float32)   # almost cancelling
        exact = float(np.sum(v.astype(np.float64)))
        scale = float(np.sum(np.abs(v.astype(np.float64))))
        rows.append([abs(naive(v) - exact) / scale, abs(compensated(v) - exact) / scale,
                     abs(compensated(v, 4) - exact) / scale])
    r = np.array(rows).max(axis=0)
    print(f"   max |error| / sum|terms| over 5 trials: naive float {r[0]:.2e}, TwoSum {r[1]:.2e}, "
          f"TwoSum + flush every 4 cells {r[2]:.2e} (float eps = {np.finfo(np.float32).eps:.2e})")
    return r


if __name__ == "__main__":
    check_params()
    w = check_plain_step()
    r = check_accumulation()
    ok = w < 1e-7 and r[1] < r[0] and r[2] < r[0]
    print("PASS" if ok else "FAIL")
