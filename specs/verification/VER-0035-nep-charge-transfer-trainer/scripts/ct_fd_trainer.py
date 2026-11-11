#!/usr/bin/env python3
"""Oracle independent of gpumd (DR-005, VER-0035): the trainer's own forces and virial in `nep`
prediction mode against central finite differences of the trainer's own energy.

Usage: ct_fd_trainer.py NEP WORKDIR [--structure 3] [--atoms 12] [--no-ct] [--delta 0.005] [--strain 0.005]
All displaced / strained copies are written to ONE train.xyz (random-free dummy labels) and evaluated in
a single `nep` prediction run. Forces: F = -dE/dr (central difference, every atom up to --atoms).
Virial: W_ab = -dE/d eps_ab for r -> (1 + eps) r, lattice -> (1 + eps) lattice (six components
xx yy zz xy yz zx, symmetric strain: W_xy = -0.5 dE/d eps_xy^sym); the trainer prints the virial per atom.
`--no-ct` uses the same generator weights without charge_transfer (the "old-model floor").
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ct_trainer_inputs as mt  # noqa: E402

STRAIN_COMPONENTS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]


def write_cells(path, mk, cells):
    with open(path, "w") as f:
        for pos, types, lat in cells:
            n = len(pos)
            f.write("%d\n" % n)
            f.write('Lattice="%s" energy=0 virial="0 0 0 0 0 0 0 0 0" '
                    'Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T"\n'
                    % " ".join("%.12g" % v for v in lat.reshape(-1)))
            for a in range(n):
                f.write("%s %.12f %.12f %.12f 0 0 0\n" % (mk.SYMBOLS[types[a]], *pos[a]))


def evaluate(nep, d, mk, cells):
    write_cells(os.path.join(d, "train.xyz"), mk, cells)
    r = subprocess.run([os.path.abspath(nep)], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-1200:])
    n_atoms = np.array([len(c[0]) for c in cells])
    e = np.loadtxt(os.path.join(d, "energy_train.out"), ndmin=2)[:, 0] * n_atoms
    f = np.loadtxt(os.path.join(d, "force_train.out"), ndmin=2)[:, :3]
    v = np.loadtxt(os.path.join(d, "virial_train.out"), ndmin=2)[:, :6] * n_atoms[:, None]
    return e, f, v


def fd_check(nep, workdir, structure=3, atoms=12, with_ct=True, delta=0.005, strain=0.005, nt=2,
             charge_mode=2, rc_phi=4.0):
    mk = mt.load_mk()
    d = os.path.abspath(workdir)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    pos, types, box = mt.make_structures(mk, [structure], nt)[0]
    lat = np.eye(3) * box
    mt.write_model(os.path.join(d, "nep.txt"), mk, 1, 4.0, 3.5, rc_phi, with_ct, nt, charge_mode)
    mt.write_nep_in(os.path.join(d, "nep.in"), mk, 4.0, 3.5, rc_phi, with_ct, charge_mode, nt, 1)
    n = len(pos)
    sel = np.unique(np.linspace(0, n - 1, min(atoms, n)).round().astype(int))
    cells = [(pos, types, lat)]
    for a in sel:
        for k in range(3):
            for s in (+1, -1):
                p = pos.copy()
                p[a, k] += s * delta
                cells.append((p, types, lat))
    for (i, j) in STRAIN_COMPONENTS:
        for s in (+1, -1):
            S = np.zeros((3, 3))
            S[i, j] = 1.0
            S[j, i] = 1.0
            F = np.eye(3) + s * strain * S * (1.0 if i == j else 1.0)
            cells.append((pos @ F.T, types, lat @ F.T))
    e, f, v = evaluate(nep, d, mk, cells)
    F_an = f[:n][sel]
    F_fd = np.zeros((len(sel), 3))
    c = 1
    for ia, a in enumerate(sel):
        for k in range(3):
            F_fd[ia, k] = -(e[c] - e[c + 1]) / (2 * delta)
            c += 2
    W_an = v[0]
    W_fd = np.zeros(6)
    for m, (i, j) in enumerate(STRAIN_COMPONENTS):
        dE = (e[c] - e[c + 1]) / (2 * strain)
        # symmetric strain: diag -> W_ii = -dE/de ; off-diag -> W_ij = -dE/de / 2
        W_fd[m] = -dE if i == j else -0.5 * dE
        c += 2
    dF = F_an - F_fd
    dW = W_an - W_fd
    return dict(rms_F=float(np.sqrt(np.mean(F_an ** 2))), F_err_rms=float(np.sqrt(np.mean(dF ** 2))),
                F_err_max=float(np.abs(dF).max()), rms_W=float(np.sqrt(np.mean(W_an ** 2))),
                W_err_rms=float(np.sqrt(np.mean(dW ** 2))), W_err_max=float(np.abs(dW).max()),
                n=n, natoms_checked=len(sel))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("nep")
    ap.add_argument("workdir")
    ap.add_argument("--structure", type=int, default=3)
    ap.add_argument("--atoms", type=int, default=12)
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--strain", type=float, default=0.005)
    ap.add_argument("--charge-mode", type=int, default=2)
    a = ap.parse_args(argv)
    r = fd_check(a.nep, a.workdir, a.structure, a.atoms, not a.no_ct, a.delta, a.strain,
                 charge_mode=a.charge_mode)
    for k, v in r.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
