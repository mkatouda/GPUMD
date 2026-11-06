#!/usr/bin/env python3
"""Compare two extended-xyz files written by d2_reference.py (energy, virial, forces).
Usage: compare_xyz_reference.py A.xyz B.xyz   -> prints max-norm relative differences
(relative = max|A-B| / max|B| per quantity; per-component relative error is meaningless for components
printed with fixed 1e-10 absolute resolution)."""
import re, sys
import numpy as np

def read(p):
    ln = open(p).read().splitlines(); n = int(ln[0]); h = ln[1]
    E = float(re.search(r'energy=(\S+)', h).group(1))
    W = np.array(re.search(r'virial="([^"]+)"', h).group(1).split(), float)
    F = np.array([l.split()[4:7] for l in ln[2:2 + n]], float)
    X = np.array([l.split()[1:4] for l in ln[2:2 + n]], float)
    return E, W, F, X

def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    assert np.array_equal(a[3], b[3])
    dE = abs(a[0] - b[0]) / abs(b[0])
    dW = np.abs(a[1] - b[1]).max() / np.abs(b[1]).max()
    dF = np.abs(a[2] - b[2]).max() / np.abs(b[2]).max()
    print(f'rel(E) {dE:.3e}  rel(W, max-norm) {dW:.3e}  rel(F, max-norm) {dF:.3e}  '
          f'abs(E) {abs(a[0]-b[0]):.2e} eV  max|dF| {np.abs(a[2]-b[2]).max():.2e} eV/A  max|dW| {np.abs(a[1]-b[1]).max():.2e} eV')
    ok = max(dE, dW, dF) <= 1e-10
    print('GATE (<=1e-10):', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
