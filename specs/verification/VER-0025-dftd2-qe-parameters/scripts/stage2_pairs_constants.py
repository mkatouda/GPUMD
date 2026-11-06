#!/usr/bin/env python3
"""Stage 2: pair values C-C, C-F, F-F from the QE 7.2 extraction vs the request table, and the effect of
the Bohr/Hartree constants. Usage: stage2_pairs_constants.py <data dir with qe72_*_Z86.csv> [out report]"""
import sys, os
import numpy as np

d = sys.argv[1]
c6 = np.loadtxt(os.path.join(d, 'qe72_c6_first_point_Eh_bohr6_Z86.csv'), delimiter=',', comments='#')
r0 = np.loadtxt(os.path.join(d, 'qe72_r0ab_angstrom_Z86.csv'), delimiter=',', comments='#')
Z = {'C': 6, 'F': 9}
req = {('C', 'C'): (49.113, 2.9103, 3.2013), ('C', 'F'): (20.2827, 2.4770, 2.7247), ('F', 'F'): (9.6916, 2.2996, 2.5296)}
dec = {('C', 'C'): (3, 4, 4), ('C', 'F'): (4, 4, 4), ('F', 'F'): (4, 4, 4)}   # printed decimals in the request
out = []
P = out.append
ok = True
for (a, b), (rc6, rr0, rrs) in req.items():
    i, j = Z[a] - 1, Z[b] - 1
    v = (c6[i, j], r0[i, j], 1.1 * r0[i, j])
    good = all(round(x, n) == y for x, y, n in zip(v, (rc6, rr0, rrs), dec[(a, b)]))
    ok &= good
    P(f'{a}-{b}: C6 {v[0]:.10g} Eh bohr^6 (request {rc6}), r0ab {v[1]:.4f} A (request {rr0}), 1.1*r0ab {v[2]:.6f} A '
      f'(request {rrs}) -> {"OK" if good else "MISMATCH"}')
P('STAGE 2 pair values gate: ' + ('PASS' if ok else 'FAIL'))
# constants
H_cod = 27.211386245988; B_cod = 0.529177210903          # d2_reference.py / QE 7.2 Modules/constants.f90 (CODATA 2018)
H_qe = 4.3597447222071e-18 / 1.602176634e-19              # QE AUTOEV
HB6_ref = H_cod * B_cod ** 6
HB6_gpumd = 0.597527426643772                             # dftd3para.cuh HartreeBohr6
Bohr_gpumd = 0.5291772575069165                           # dftd3para.cuh Bohr
P(f'QE 7.2 AUTOEV = HARTREE_SI/ELECTRONVOLT_SI = {H_qe:.12f} (d2_reference.py {H_cod}); rel diff {H_qe / H_cod - 1:.2e}')
P(f'Hartree*Bohr^6 [eV A^6 per Eh bohr^6]: CODATA2018 (d2_reference.py, QE 7.2 constants.f90) {HB6_ref:.12f}; '
  f'GPUMD HartreeBohr6 {HB6_gpumd:.12f}; rel diff (GPUMD/ref - 1) = {HB6_gpumd / HB6_ref - 1:+.3e}')
P(f'GPUMD Bohr {Bohr_gpumd} vs CODATA2018 {B_cod}: rel {Bohr_gpumd / B_cod - 1:+.3e}; (ratio)^6 - 1 = {(Bohr_gpumd / B_cod) ** 6 - 1:+.3e}')
P(f'  GPUMD HartreeBohr6 if recomputed from its own Bohr and CODATA Hartree: {H_cod * Bohr_gpumd ** 6:.12f} '
  f'(rel to ref {H_cod * Bohr_gpumd ** 6 / HB6_ref - 1:+.3e}); float32 rounding of the literal: {float(np.float32(HB6_gpumd)) / HB6_gpumd - 1:+.2e}')
# QE-internal: r0ab divided by autoang=0.52917720859 (dft-d3/common.f90), positions converted with BOHR_RADIUS_SI
aq = 0.52917720859
P(f'QE dft-d3 autoang {aq} vs BOHR_RADIUS_SI {B_cod}: effective R0 (A) = r0ab * {B_cod / aq:.12f} (rel {B_cod / aq - 1:+.2e}); '
  f'effect on d2_reference.py (decimal r0ab, no such factor): argument of the exponential up to 20*{B_cod / aq - 1:.2e} = {20 * (B_cod / aq - 1):.1e}')
open(os.path.join(d, 'stage2_pairs_constants_report.txt'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
