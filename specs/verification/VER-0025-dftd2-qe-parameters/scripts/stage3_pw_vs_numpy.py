#!/usr/bin/env python3
"""Stage 3: QE pw.x (dftd3_version = 2) vs d2_reference.py (table qe, plain, rc = sqrt(9000) bohr).

Usage: stage3_pw_vs_numpy.py <workdir> [--pw PW_X] [--np N] [--run]
Runs two SCF jobs with identical geometry (vdw_corr='dft-d3', dftd3_version=2, dftd3_threebody=.false. vs no vdw_corr),
takes the 'DFT-D3 Dispersion Correction' energy directly, and forces/stress as (with D3) - (without D3), which equals the
D2 contribution because the dispersion term does not enter the Kohn-Sham potential. Environment (MPI/MKL) must be set up
by the caller (source /opt/intel/oneapi/setvars.sh).
"""
import argparse, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2_reference as d2r

BOHR = 0.529177210903; RYTOEV = 13.605693122994
PS = '/home/katouda/bin/qe/ps/sg15_oncv'
# slightly distorted CF4 + a second displaced F ion pair is unnecessary; one CF4 in a triclinic 9 A cell (5 atoms)
CELL = np.array([[9.0, 0.0, 0.0], [0.9, 8.6, 0.0], [-0.7, 0.5, 9.3]])
SPEC = ['C', 'F', 'F', 'F', 'F']
POS = np.array([[4.50, 4.30, 4.65], [5.60, 5.10, 5.30], [3.45, 5.20, 4.10], [3.90, 3.15, 5.55], [5.15, 3.60, 3.90]])

def write_input(path, d3):
    with open(path, 'w') as f:
        f.write(f"""&control
  calculation='scf', prefix='cf4', outdir='./tmp_{"d3" if d3 else "no"}', pseudo_dir='{PS}', tprnfor=.true., tstress=.true.
/
&system
  ibrav=0, nat=5, ntyp=2, ecutwfc=40.0, input_dft='pbe'
""")
        if d3:
            f.write("  vdw_corr='dft-d3', dftd3_version=2, dftd3_threebody=.false.\n")
        f.write("""/
&electrons
  conv_thr=1.0d-12, mixing_beta=0.5
/
ATOMIC_SPECIES
C 12.011 C_ONCV_PBE-1.2.upf
F 18.998 F_ONCV_PBE-1.2.upf
CELL_PARAMETERS angstrom
""")
        for v in CELL: f.write('%.10f %.10f %.10f\n' % tuple(v))
        f.write('ATOMIC_POSITIONS angstrom\n')
        for s, x in zip(SPEC, POS): f.write('%s %.10f %.10f %.10f\n' % ((s,) + tuple(x)))
        f.write('K_POINTS gamma\n')

def parse(path):
    """full-precision values from the XML next to the text output (Hartree units): etot, forces, stress.
    Also returns the printed 'DFT-D3 Dispersion' (Ry, 8 decimals) from the text output."""
    t = open(path).read()
    m = re.search(r'DFT-D3 Dispersion\s*=\s*(\S+)\s*Ry', t)
    ed3 = float(m.group(1)) if m else None
    prefix_dir = os.path.join(os.path.dirname(path), 'tmp_' + ('d3' if os.path.basename(path).startswith('d3') else 'no'))
    x = open(os.path.join(prefix_dir, 'cf4.save', 'data-file-schema.xml')).read()
    etot = float(re.search(r'<etot>\s*(\S+)\s*</etot>', x).group(1))
    F = np.array(re.search(r'<forces rank[^>]*>(.*?)</forces>', x, re.S).group(1).split(), float).reshape(-1, 3)
    S = np.array(re.search(r'<stress rank[^>]*>(.*?)</stress>', x, re.S).group(1).split(), float).reshape(3, 3)
    return ed3, F, S, etot

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('work'); ap.add_argument('--pw', default='/home/katouda/bin/qe/bin/pw.x')
    ap.add_argument('--np', type=int, default=4); ap.add_argument('--r0scale', type=float, default=1.0, help='scale r0ab of the numpy reference (QE autoang effect: 1 + 4.371e-9)'); ap.add_argument('--run', action='store_true'); a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for d3 in (True, False):
        nm = os.path.join(a.work, 'd3.in' if d3 else 'no.in'); write_input(nm, d3)
        if a.run:
            with open(nm.replace('.in', '.out'), 'w') as o:
                subprocess.run(['mpirun', '-np', str(a.np), a.pw, '-in', nm], cwd=a.work, stdout=o, stderr=subprocess.STDOUT, check=True)
    e1, F1, S1, et1 = parse(os.path.join(a.work, 'd3.out')); e0, F0, S0, et0 = parse(os.path.join(a.work, 'no.out'))
    HA = 27.211386245988                                  # QE AUTOEV
    rc = np.sqrt(9000.0) * BOHR
    d2r.QE_R0AB_A = {k: v * a.r0scale for k, v in d2r.QE_R0AB_A.items()}   # default 1.0: decimal r0ab as printed
    print(f'r0ab scale factor applied to the numpy reference: {a.r0scale!r}')
    E, F, W = d2r.d2(SPEC, POS, CELL, 'qe', 'plain', rc)
    vol = abs(np.linalg.det(CELL))                       # A^3
    EQE = (et1 - et0) * HA                               # total-energy difference (XML, Hartree) -> eV
    FQE = (F1 - F0) * HA / BOHR                          # Ha/bohr -> eV/A
    # QE sigma = -(1/V) dE/d(strain) = W / V; W in eV, V in A^3 -> eV/A^3
    SQE = (S1 - S0) * HA / BOHR ** 3
    SREF = W / vol
    print(f'rc = {rc:.6f} A, cell volume {vol:.4f} A^3')
    print(f'printed DFT-D3 Dispersion {e1} Ry = {e1 * HA / 2:.8f} eV (8 decimals in Ry only)')
    print(f'E_D3 QE(7.6-dev, XML etot difference) {EQE:.10f} eV   numpy {E:.10f} eV   diff {EQE - E:+.3e} eV   rel {abs(EQE - E) / abs(E):.3e}')
    print(f'forces: max|F| numpy {np.abs(F).max():.6e} eV/A, max|dF| {np.abs(FQE - F).max():.3e} eV/A  rel(max-norm) {np.abs(FQE - F).max() / np.abs(F).max():.3e}')
    print(f'stress (eV/A^3): max|S| numpy {np.abs(SREF).max():.6e}, max|dS| {np.abs(SQE - SREF).max():.3e} rel(max-norm) {np.abs(SQE - SREF).max() / np.abs(SREF).max():.3e}; in kbar: max|dS| {np.abs(SQE - SREF).max() * 1602.1766208:.5f} kbar, max|S| {np.abs(SREF).max() * 1602.1766208:.3f} kbar')

if __name__ == '__main__':
    main()
