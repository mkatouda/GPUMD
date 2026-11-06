#!/usr/bin/env python3
"""Stage 1 comparison: QE 7.2 first-point C6 and r0ab vs GPUMD tables and VER-0023.

Usage: compare_qe_gpumd.py <qe72 csv dir (output of extract_qe_d2_params.py)> <repo root> <datadir-out>
Needs only numpy. Reads (read-only) <repo>/src/force/dftd3para.cuh, dftd3zero_table.cuh and
<repo>/specs/verification/VER-0023-dftd3-zero-parameters/data/r0ab_angstrom.csv.
"""
import re, sys, os
import numpy as np

def cuh_array(path, name):
    txt = open(path).read()
    m = re.search(name + r'\s*\[[^\]]*\]\s*=\s*\{(.*?)\};', txt, re.S)
    toks = [t.strip().rstrip('f') for t in m.group(1).replace('\n', ' ').split(',') if t.strip()]
    return toks

def load_csv(p):
    return np.loadtxt(p, delimiter=',', comments='#')

def main():
    qdir, repo, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    f32 = np.float32
    c6q = load_csv(os.path.join(qdir, 'qe_qe72_c6_first_point.csv'))
    cnq = load_csv(os.path.join(qdir, 'qe_qe72_c6_first_point_cn.csv'))
    r0q = load_csv(os.path.join(qdir, 'qe_qe72_r0ab_decimal.csv'))
    r0h = load_csv(os.path.join(qdir, 'qe_qeHEAD_r0ab_decimal.csv'))
    c6h = load_csv(os.path.join(qdir, 'qe_qeHEAD_c6_first_point.csv'))
    para = os.path.join(repo, 'src/force/dftd3para.cuh')
    c6tok = cuh_array(para, 'c6_ref_cpu'); assert len(c6tok) == 111625
    cntok = cuh_array(para, 'cn_ref'); assert len(cntok) == 5 * 94
    r0tok = cuh_array(os.path.join(repo, 'src/force/dftd3zero_table.cuh'), 'd3zero_r0ab'); assert len(r0tok) == 4465
    v23 = load_csv(os.path.join(repo, 'specs/verification/VER-0023-dftd3-zero-parameters/data/r0ab_angstrom.csv'))
    rep = []
    P = rep.append
    NZ = 86
    # ---- C6 first point ----
    mism = []; maxrel = 0.0; maxrel32 = 0.0
    cn_first_nonzero_gpumd = [Z + 1 for Z in range(94) if float(cntok[5 * Z]) != 0.0]
    cn_first_nonzero_qe = [Z + 1 for Z in range(94) if cnq[Z, Z] != 0.0]
    cn_mis = []
    for zs in range(94):
        for zl in range(zs, 94):
            z12 = zs * 94 - zs * (zs - 1) // 2 + (zl - zs)
            tok = c6tok[z12 * 25]
            g = float(tok)                      # decimal as written in the GPUMD table
            q = c6q[zs, zl]                     # decimal parsed from pars.f90 (double)
            assert q == c6q[zl, zs]
            rel = abs(g - q) / q
            rel32 = abs(float(f32(g)) - q) / q  # what the float table holds
            # CN of the first point (QE: cnq[i,j] = CN of atom i; GPUMD: first reference CN per element)
            if zl <= NZ - 1 or True:
                if not (float(cntok[5 * zs]) == cnq[zs, zl] and float(cntok[5 * zl]) == cnq[zl, zs]):
                    cn_mis.append((zs + 1, zl + 1, cnq[zs, zl], cnq[zl, zs], float(cntok[5 * zs]), float(cntok[5 * zl])))
            if zl < NZ:
                maxrel = max(maxrel, rel); maxrel32 = max(maxrel32, rel32)
                if rel != 0.0:
                    mism.append((zs + 1, zl + 1, q, g, rel))
            else:
                if rel != 0.0:
                    pass
    P(f'C6 first point, Z<=86: pairs {86 * 87 // 2}, decimal mismatches vs GPUMD c6_ref_cpu[z12*25]: {len(mism)}; '
      f'max rel (decimal) {maxrel:.3e}; max rel (float32 table value vs QE double) {maxrel32:.3e}')
    for m in mism[:50]:
        P('  MISMATCH C6 Z=%d,%d QE=%.8g GPUMD=%.8g rel=%.2e' % m)
    info = []
    for zs in range(94):
        for zl in range(zs, 94):
            if zl < NZ: continue
            z12 = zs * 94 - zs * (zs - 1) // 2 + (zl - zs)
            g = float(c6tok[z12 * 25]); q = c6q[zs, zl]
            if g != q: info.append((zs + 1, zl + 1, q, g))
    P(f'C6 first point, pairs with Z>=87 (information only): {int(94*95/2-86*87/2)} pairs, decimal mismatches vs GPUMD: {len(info)}')
    # ---- CN of first points ----
    P(f'First reference CN != 0 (GPUMD cn_ref[5*Z]), Z<=86: {[Z for Z in cn_first_nonzero_gpumd if Z <= NZ]}; '
      f'QE c6ab(Z,Z,1,1,2) != 0: {[Z for Z in cn_first_nonzero_qe if Z <= NZ]}')
    P(f'First-point CN mismatch QE vs GPUMD cn_ref (all 94 elements, all pairs): {len(cn_mis)}')
    # ---- C6 vs HEAD ----
    P(f'C6 first point QE 7.2 vs QE HEAD (7.6-515), all Z<=94: max abs diff {np.abs(c6q - c6h).max():.3e}')
    # ---- r0ab ----
    nm_v23 = 0; nm_gp = 0; nm_gp_f32 = 0; lst = []
    for zs in range(94):
        for zl in range(zs, 94):
            k = zl * (zl + 1) // 2 + zs
            q = r0q[zl, zs]
            assert r0q[zs, zl] == q
            v = v23[zl, zs]
            g = float(r0tok[k])
            inrange = zl < NZ
            if inrange:
                if q != v: nm_v23 += 1; lst.append(('VER-0023', zs + 1, zl + 1, q, v))
                if q != g: nm_gp += 1; lst.append(('GPUMD d3zero_r0ab', zs + 1, zl + 1, q, g))
                if f32(q) != f32(g): nm_gp_f32 += 1
    P(f'r0ab Z<=86: {86*87//2} pairs; decimal mismatches QE7.2 vs VER-0023 csv: {nm_v23}; vs GPUMD d3zero_r0ab literals: {nm_gp} (as float32: {nm_gp_f32})')
    for m in lst[:50]:
        P('  MISMATCH r0ab %s Z=%d,%d QE=%.4f other=%.4f' % m)
    hi = [(zs + 1, zl + 1, r0q[zl, zs], v23[zl, zs]) for zs in range(94) for zl in range(zs, 94) if zl >= NZ and r0q[zl, zs] != v23[zl, zs]]
    P(f'r0ab pairs with Z>=87 (information only): decimal differences QE7.2 vs VER-0023: {len(hi)}')
    P(f'r0ab QE 7.2 vs QE HEAD decimals (all 4465): differences {int((r0q != r0h).sum())}')
    # single precision effect: 7.2 real(4) literal -> float32 value; HEAD _wp literal -> double. Both then / autoang
    autoang = 0.52917720859
    r72 = np.array([float(f32(x)) for x in r0q.ravel()]) / autoang
    rH = r0h.ravel() / autoang
    rel = np.abs(r72 - rH) / rH
    P(f'r0ab bohr, QE 7.2 (float32 literal)/autoang vs HEAD (double literal)/autoang: max rel {rel.max():.3e}; '
      f'effect on exp(-20 (r/(1.1 R0) - 1)) argument at r=R0*1.1: up to {20 * rel.max():.2e} absolute')
    # d2 damping effect of float32 literals on f at r = 1.1*R0 (largest sensitivity ~ alp*dR0/R0)
    open(os.path.join(out, 'stage1_compare_report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    # ---- data files: Z<=86 first-point C6 (Eh bohr^6) and r0ab (A) as extracted from QE 7.2
    np.savetxt(os.path.join(out, 'qe72_c6_first_point_Eh_bohr6_Z86.csv'), c6q[:86, :86], fmt='%.8e', delimiter=',',
               header='QE 7.2 (b4c5e8de) dft-d3/pars.f90 c6ab(i,j,1,1,1), Hartree bohr^6, Z=1..86 (rows/cols)', comments='# ')
    np.savetxt(os.path.join(out, 'qe72_r0ab_angstrom_Z86.csv'), r0q[:86, :86], fmt='%.4f', delimiter=',',
               header='QE 7.2 (b4c5e8de) dft-d3/core.f90 setr0ab r0ab, Angstrom (decimal literal; real(4) in 7.2), Z=1..86', comments='# ')

if __name__ == '__main__':
    main()
