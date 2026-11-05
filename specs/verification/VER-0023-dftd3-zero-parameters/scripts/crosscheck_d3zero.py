#!/usr/bin/env python3
"""Cross-check the generated D3(0) tables.

Usage: crosscheck_d3zero.py <probe binary> <data dir> [<results.txt>]
Needs numpy and python `dftd3` (simple-dftd3 1.7.0 bindings), run with e.g.
  /home/katouda/miniforge3/envs/py312-teacher-orb/bin/python -I crosscheck_d3zero.py ...

Checks
 A  generated r0ab == r0ab actually stored in the compiled dftd3-lib (probe 'R'), emulating the
    single-precision Fortran literals:   float32(x_A)/0.52917726  (exact equality expected)
 B  generated parameters == dftd3-lib runtime parameters (probe 'P'), same float32 emulation
 C  independent: simple-dftd3 ZeroDampingParam(method=name) energy == energy with the generated
    parameters passed explicitly (dimers, several r)        [parameter equality via energies]
 D  independent: R0 of all 4465 pairs recovered from simple-dftd3 dimer energies
    (E at two sr6 values, C6 cancels) vs generated r0ab / bohr conversion
 E  DR-003 two-body formula E=-(s6 C6 f6/r^6 + s8 C8 f8/r^8) evaluated with the generated tables
    and C6,C8 taken from simple-dftd3 (damping switched off) vs simple-dftd3 and dftd3-lib energies
"""
import subprocess
import sys

import numpy as np
from dftd3.interface import DispersionModel, ZeroDampingParam

probe, data = sys.argv[1], sys.argv[2]
resfile = sys.argv[3] if len(sys.argv) > 3 else None
AUTOANG_LIB = 0.52917726
AUTOANG_CODATA14 = 0.52917721067
AUTOANG_CODATA18 = 0.529177210903
NEL = 94
log = []


def say(s):
    print(s)
    log.append(s)


# ---------------- read generated tables
R = np.array([[float(x) for x in l.split(",")] for l in open(data + "/r0ab_angstrom.csv") if not l.startswith("#")])
assert R.shape == (NEL, NEL) and np.array_equal(R, R.T)
params = []  # name, variant, s6, sr6, s8, sr8, a6, a8 (strings kept)
for l in open(data + "/d3zero_params.csv"):
    if l.startswith("#") or not l.strip():
        continue
    params.append(l.strip().split(","))
f32 = lambda x: float(np.float32(x))


def run_probe(cmds):
    p = subprocess.run([probe], input="\n".join(cmds) + "\n", capture_output=True, text=True, check=True)
    return p.stdout.splitlines()


# ---------------- A
out = run_probe(["R"])
Rlib = np.array([[float(x) for x in l.split()] for l in out])
assert Rlib.shape == (NEL, NEL)
emul = np.vectorize(f32)(R) / AUTOANG_LIB
nA = int(np.sum(Rlib != emul))
decrel = np.max(np.abs(Rlib - R / AUTOANG_LIB) / (R / AUTOANG_LIB))
say("A  r0ab generated vs dftd3-lib runtime: entries %d, mismatches (float32-literal emulation, exact) = %d; "
    "max rel diff vs decimal value = %.2e (float32 literal rounding, <= 6e-8)" % (NEL * NEL, nA, decrel))
say("   units: table is Angstrom; lib bohr = table/0.52917726. Sanity: r0ab(H,H)=%.4f A = %.4f bohr, r0ab(C,C)=%.4f A, r0ab(Pu,Pu)=%.4f A"
    % (R[0, 0], R[0, 0] / AUTOANG_LIB, R[5, 5], R[93, 93]))

# ---------------- B
cmds = ["P %s %d" % (p[0], 1 if p[1] == "TZVPP" else 0) for p in params]
out = run_probe(cmds)
nB = 0
for p, l in zip(params, out):
    t = l.split()
    assert t[0] == p[0]
    lib = [float(x) for x in t[1:]]  # s6 rs6 s18 rs18 alp
    gen = [float(p[2]), float(p[3]), float(p[4]), float(p[5]), 14.0]
    # s6 default is a double literal 1.0d0, others float32 literals; compare with float32 emulation (1.0, 0.0 exact)
    emu = [f32(g) for g in gen]
    if lib != emu:
        nB += 1
        say("   B mismatch %s %s lib=%s gen=%s" % (p[0], p[1], lib, gen))
say("B  parameters generated vs dftd3-lib runtime setfuncpar(version=3): %d entries (%d functional/variant), mismatches = %d"
    % (len(params) * 5, len(params), nB))

# ---------------- simple-dftd3 helpers
BOHR_A = AUTOANG_LIB


def energy(z1, z2, r, par):
    m = DispersionModel(np.array([z1, z2]), np.array([[0.0, 0, 0], [0, 0, r]]))
    return float(m.get_dispersion(par, grad=False)["energy"])


def custom(s6, s8, sr6, sr8):
    return ZeroDampingParam(s6=s6, s8=s8, sr6=sr6, sr8=sr8, alpha6=14.0)


# ---------------- C
def sd3_names(n):
    cands = [n, n.replace("-", ""), {"b-p": "bp", "revpbe38": "revpbe38", "rpw86-pbe": "rpw86pbe"}.get(n, n)]
    for c in dict.fromkeys(cands):
        try:
            return c, ZeroDampingParam(method=c)
        except Exception:
            pass
    return None, None


probes = [(1, 1), (6, 6), (6, 8), (8, 1), (17, 17), (14, 9), (26, 30), (79, 79)]
rs = [3.0, 5.0, 6.5, 8.0, 12.0]  # bohr
nC = 0
worstC = 0.0
uncovered = []
covered = []
for p in params:
    if p[1] != "def2-QZVP":
        continue
    c, par = sd3_names(p[0])
    if par is None:
        uncovered.append(p[0])
        continue
    cp = custom(float(p[2]), float(p[4]), float(p[3]), float(p[5]))
    ok = True
    for z1, z2 in probes:
        for r in rs:
            e1, e2 = energy(z1, z2, r, par), energy(z1, z2, r, cp)
            if e1 == 0.0 and e2 == 0.0:
                continue
            d = abs(e1 - e2) / abs(e1)
            worstC = max(worstC, d)
            if d > 1e-10:
                ok = False
    covered.append((p[0], c, ok))
    nC += 0 if ok else 1
say("C  simple-dftd3 method=name vs explicit generated parameters: %d functionals covered, %d disagree (>1e-10 rel), "
    "worst rel diff %.1e over %d dimers x %d r" % (len(covered), nC, worstC, len(probes), len(rs)))
for n, c, ok in covered:
    if not ok:
        say("   C disagree: %s (simple-dftd3 name %s)" % (n, c))
say("   not known to simple-dftd3 by that name (NOT independently cross-checked, lib only): %s" % (", ".join(uncovered) or "none"))
stageA = ["pbe", "revpbe", "b-lyp", "b3-lyp", "tpss"]
say("   Stage A covered by simple-dftd3 and agreeing: %s" % ", ".join("%s(%s)" % (n, c) for n, c, ok in covered if n in stageA and ok))

# ---------------- D: recover R0 of every pair from dimer energies (C6 cancels)
#   E_x = E0 / (1 + 6 x^14 y),  y = (R0/r)^14   ->  y = (Ea-Eb) / (6 (Eb B - Ea A)),  A=a^14, B=b^14
a, b = 1.0, 0.7
A, B = a ** 14, b ** 14


def recover(Ea, Eb, r):
    den = 6.0 * (Eb * B - Ea * A)
    if Ea == 0.0 or Eb == 0.0 or den == 0.0:
        return np.nan
    return r * ((Ea - Eb) / den) ** (1.0 / 14.0)


pa, pb = custom(1.0, 0.0, a, 1.0), custom(1.0, 0.0, b, 1.0)
rec = np.full((NEL, NEL), np.nan)
rec_lib = np.full((NEL, NEL), np.nan)
cmds = []
idx = []
for i in range(NEL):
    for j in range(i + 1):
        r = 1.2 * R[i, j] / AUTOANG_LIB
        rec[i, j] = rec[j, i] = recover(energy(i + 1, j + 1, r, pa), energy(i + 1, j + 1, r, pb), r)
        cmds.append("Q 1.0 %.17g 0.0 1.0 %d %d %.17g" % (a, i + 1, j + 1, r))
        cmds.append("Q 1.0 %.17g 0.0 1.0 %d %d %.17g" % (b, i + 1, j + 1, r))
        idx.append((i, j, r))
outl = run_probe(cmds)
for k, (i, j, r) in enumerate(idx):
    rec_lib[i, j] = rec_lib[j, i] = recover(float(outl[2 * k]), float(outl[2 * k + 1]), r)

Z86 = np.zeros((NEL, NEL), bool)
Z86[:86, :86] = True
Zhi = ~Z86
Rf32 = np.vectorize(f32)(R)


def stat(rr, ref, m):
    rel = np.abs(rr[m] - ref[m]) / ref[m]
    return int(m.sum()), np.nanmax(rel), int(np.sum(~np.isfinite(rel)))


# the oracle (simple-dftd3) uses CODATA18 for bohr->A; dftd3-lib uses 0.52917726
n, mx, nn = stat(rec, R / AUTOANG_CODATA18, Z86)
say("D1 R0 recovered from simple-dftd3 dimers vs table/0.529177210903, Z<=86: %d pairs (full matrix entries), max rel %.2e, unusable %d" % (n, mx, nn))
n, mx, nn = stat(rec, R / AUTOANG_LIB, Z86)
say("D1b same but table/0.52917726 (dftd3-lib constant): max rel %.2e  (=> simple-dftd3 uses the CODATA2018 Angstrom/bohr; ratio %.3e)"
    % (mx, AUTOANG_LIB / AUTOANG_CODATA18 - 1))
n, mx, nn = stat(rec, R / AUTOANG_CODATA18, Zhi)
nbad = int(np.sum(np.abs(rec[Zhi] - (R / AUTOANG_CODATA18)[Zhi]) / (R / AUTOANG_CODATA18)[Zhi] > 1e-6))
say("D2 Z>=87 (any partner), simple-dftd3 vs generated: %d matrix entries, %d differ by >1e-6, max rel %.2e  ==> SOURCES DISAGREE"
    % (n, nbad, mx))
dd = np.abs(rec[:86, :86] - (R / AUTOANG_CODATA18)[:86, :86])
say("D3 R0 recovered from dftd3-lib itself (energies with explicit parameters, all Z<=94) vs float32 table/0.52917726: "
    "max rel %.2e over %d entries" % (np.nanmax(np.abs(rec_lib - Rf32 / AUTOANG_LIB) / (Rf32 / AUTOANG_LIB)), NEL * NEL))

# ---------------- E: DR-003 formula with the generated tables
cases = [("pbe", 6, 6), ("pbe", 1, 1), ("pbe", 6, 8), ("revpbe", 8, 1), ("b-lyp", 17, 17), ("b3-lyp", 14, 9),
         ("tpss", 26, 30), ("pbe", 79, 79), ("pbe", 86, 86), ("pbe", 94, 94), ("slater-dirac-exchange", 6, 6)]
pd = {p[0]: p for p in params if p[1] == "def2-QZVP"}
fcs = (0.5, 0.8, 1.0, 1.3, 2.0, 5.0)
worst = {"sd_dec": 0.0, "lib_f32": 0.0, "lib_dec": 0.0, "sd_libconst": 0.0}
worst_hi = {"sd_dec": 0.0}
cmds, rows = [], []
for name, z1, z2 in cases:
    p = pd[name]
    s6, sr6, s8, sr8 = float(p[2]), float(p[3]), float(p[4]), float(p[5])
    c, par = sd3_names(name)
    for f in fcs:
        r = f * R[z1 - 1, z2 - 1] / AUTOANG_CODATA18
        # C6, C8 from simple-dftd3 with damping switched off (sr -> 1e-3)
        e6 = energy(z1, z2, r, custom(1.0, 0.0, 1e-3, 1.0))
        e8 = energy(z1, z2, r, custom(0.0, 1.0, 1e-3, 1e-3))
        C6, C8 = -e6 * r ** 6, -e8 * r ** 8
        Esd = energy(z1, z2, r, par)

        def formula(s6_, sr6_, s8_, sr8_, R0):
            f6 = 1.0 / (1.0 + 6.0 * (sr6_ * R0 / r) ** 14)
            f8 = 1.0 / (1.0 + 6.0 * (sr8_ * R0 / r) ** 16)
            return -(s6_ * C6 * f6 / r ** 6 + s8_ * C8 * f8 / r ** 8)

        Ed = formula(s6, sr6, s8, sr8, R[z1 - 1, z2 - 1] / AUTOANG_CODATA18)
        rows.append((z1, z2, r, Esd, Ed, formula(s6, sr6, s8, sr8, R[z1 - 1, z2 - 1] / AUTOANG_LIB),
                     (name, z1, z2, s6, sr6, s8, sr8, r)))
for (z1, z2, r, Esd, Ed, Edl, meta) in rows:
    name = meta[0]
    cmds.append("D %s 0 %d %d %.17g" % (name, z1, z2, r))
outl = run_probe(cmds)
for (z1, z2, r, Esd, Ed, Edl, meta), l in zip(rows, outl):
    name, _, _, s6, sr6, s8, sr8, _ = meta
    El = float(l)
    d = abs(Esd - Ed) / abs(Esd)
    key = "sd_dec"
    if max(z1, z2) <= 86:
        worst["sd_dec"] = max(worst["sd_dec"], d)
        worst["sd_libconst"] = max(worst["sd_libconst"], abs(Esd - Edl) / abs(Esd))
        # lib: its C6/C8 equal the oracle's for Z<=86 (checked by the agreement itself); emulate float32 literals
        R0f = Rf32[z1 - 1, z2 - 1] / AUTOANG_LIB
        e6 = energy(z1, z2, r, custom(1.0, 0.0, 1e-3, 1.0))
        e8 = energy(z1, z2, r, custom(0.0, 1.0, 1e-3, 1e-3))
        C6, C8 = -e6 * r ** 6, -e8 * r ** 8
        f6 = 1.0 / (1.0 + 6.0 * (f32(sr6) * R0f / r) ** 14)
        f8 = 1.0 / (1.0 + 6.0 * (f32(sr8) * R0f / r) ** 16)
        Ef32 = -(f32(s6) * C6 * f6 / r ** 6 + f32(s8) * C8 * f8 / r ** 8)
        worst["lib_f32"] = max(worst["lib_f32"], abs(El - Ef32) / abs(El))
        worst["lib_dec"] = max(worst["lib_dec"], abs(El - Ed) / abs(El))
    else:
        worst_hi["sd_dec"] = max(worst_hi["sd_dec"], d)
say("E1 DR-003 formula (decimal tables, oracle bohr constant, C6/C8 from simple-dftd3) vs simple-dftd3, Z<=86 dimers: worst rel %.2e" % worst["sd_dec"])
say("E2 same formula with float32-literal tables (what dftd3-lib really uses) vs dftd3-lib energy, Z<=86: worst rel %.2e" % worst["lib_f32"])
say("E3 informative: decimal tables vs dftd3-lib energy: worst rel %.2e (float32 literals x alpha=14 amplification);"
    " decimal tables with 0.52917726 instead of CODATA18 vs simple-dftd3: worst rel %.2e" % (worst["lib_dec"], worst["sd_libconst"]))
say("E4 Pu-Pu (Z=94) dimer, formula vs simple-dftd3 (same C6/C8, differing r0ab): worst rel %.2e" % worst_hi["sd_dec"])

gate_le86 = (nA == 0 and nB == 0 and nC == 0 and np.nanmax(np.abs(rec[Z86] - (R / AUTOANG_CODATA18)[Z86]) / (R / AUTOANG_CODATA18)[Z86]) < 1e-9
             and worst["sd_dec"] < 1e-9 and worst["lib_f32"] < 1e-6)
say("GATE Z<=86 (A,B exact; C<=1e-10; D1,E1 <1e-9; E2 <1e-6): %s" % ("PASS" if gate_le86 else "FAIL"))
say("GATE Z=87..94 vs simple-dftd3: %s (%d entries differ >1e-6)" % ("PASS" if nbad == 0 else "FAIL", nbad))
if resfile:
    open(resfile, "w").write("\n".join(log) + "\n")
sys.exit(0 if (gate_le86 and nbad == 0) else 1)
