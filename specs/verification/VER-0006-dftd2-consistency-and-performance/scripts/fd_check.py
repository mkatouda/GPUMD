"""Finite-difference consistency of the GPU D2 term (AC-4).

usage: fd_check.py GPUMD_BIN NEP_TXT FIXTURE_DIR WORKDIR [s6] [rc]
D2 part := (run with dftd2) - (same run without), so the NEP part cancels.
Forces: 5-point central difference of E_D2 on selected atoms, scan over h.
Virial: W_ab = -dE/d(eps_ab), 5-point central difference of the deformation x -> (1 + eps) x.
Tolerances (fixed before the measurement, see README): nominal h_force = 0.01 A, h_strain = 1e-3;
force error <= 2e-4 * max|F_D2|, virial error <= 2e-4 * max|W_D2|.
"""
import re, subprocess, sys, shutil, itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ase.io import read, write

gpumd, nep, fixdir, work = sys.argv[1:5]
S6 = float(sys.argv[5]) if len(sys.argv) > 5 else 75.0
RC = float(sys.argv[6]) if len(sys.argv) > 6 else 12.0
H_FORCE = [0.0025, 0.005, 0.01, 0.02, 0.04]
H_STRAIN = [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
NOM_HF, NOM_HS = 0.01, 1e-3
TOL = 2e-4
_n = itertools.count()


def run(atoms, d2):
    d = Path(work) / f'r{next(_n)}'
    d.mkdir(parents=True)
    write(d / 'model.xyz', atoms, format='extxyz')
    L = [f'potential {nep}'] + ([f'dftd2 {S6} {RC}'] if d2 else []) + \
        ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    r = subprocess.run([gpumd], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    h = (d / 'dump.xyz').read_text().splitlines()[1]
    E = float(re.search(r'energy=([-0-9.eE+]+)', h).group(1))
    W = np.array(re.search(r'virial="([^"]+)"', h).group(1).split(), float).reshape(3, 3)
    F = read(d / 'dump.xyz', format='extxyz').get_forces()
    shutil.rmtree(d)
    return E, F, W


def d2_only(atoms):
    (e1, f1, w1), (e0, f0, w0) = run(atoms, True), run(atoms, False)
    return e1 - e0, f1 - f0, w1 - w0


def d2_energy(atoms):
    return d2_only(atoms)[0]


def disp(atoms, k, a, step):
    at = atoms.copy(); p = at.positions; p[k, a] += step; at.set_positions(p); return at


def strained(atoms, a, b, step):
    F = np.eye(3); F[a, b] += step
    at = atoms.copy(); pos = atoms.positions.copy(); cell = atoms.cell.array.copy()
    at.set_cell(cell @ F.T, scale_atoms=False); at.set_positions(pos @ F.T)
    return at


def stencil(fn, atoms, make, h):
    es = [fn(make(atoms, s)) for s in (-2 * h, -h, h, 2 * h)]
    return (es[0] - 8 * es[1] + 8 * es[2] - es[3]) / (12 * h)


def systems():
    w = read(Path(fixdir) / 'water27.xyz')
    wt = read(Path(fixdir) / 'water27_tri.xyz')
    cl = w.copy(); cl.pbc = False
    slab = w.copy(); slab.pbc = [True, True, False]
    return {'water27_cubic': w, 'water27_triclinic': wt, 'water27_open': cl, 'water27_slab_TTF': slab}


if __name__ == '__main__':
    atoms_sel = [0, 1, 40]  # O, H, and an O from the middle of the cell
    ex = ThreadPoolExecutor(4)
    print(f'# s6={S6} rc={RC}  tolerances: force {TOL}*max|F|, virial {TOL}*max|W|; nominal h {NOM_HF} A, {NOM_HS}')
    for name, at in systems().items():
        e, F, W = d2_only(at)
        # determinism of the on-off difference (NEP part cancels?)
        e_b, F_b, W_b = d2_only(at)
        print(f'## {name}: E_D2={e:.6f} eV  max|F|={np.abs(F).max():.4f} eV/A  max|W|={np.abs(W).max():.4f} eV; '
              f'repeat diff: dE={abs(e-e_b):.2e} dF={np.abs(F-F_b).max():.2e} dW={np.abs(W-W_b).max():.2e}')
        for h in H_FORCE:
            jobs = [(k, a) for k in atoms_sel for a in range(3)]
            res = list(ex.map(lambda ka: -stencil(d2_energy, at, lambda x, s, ka=ka: disp(x, ka[0], ka[1], s), h), jobs))
            err = max(abs(r - F[k, a]) for r, (k, a) in zip(res, jobs))
            tag = ' <- nominal' if h == NOM_HF else ''
            print(f'  force h={h:<7} max|F_fd-F_gpu|={err:.3e} eV/A  rel={err/np.abs(F).max():.2e}{tag}', flush=True)
        comps = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)] if at.pbc.all() else []
        # virial: only meaningful as a strain derivative when the system is periodic in all directions
        # (for open/mixed boundaries the deformation also moves non-periodic positions: still W = -dE/d eps
        # holds with x -> (1+eps)x; GPUMD virial is sum r_i F_i, which for open systems is the same quantity)
        comps = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
        for h in H_STRAIN:
            res = list(ex.map(lambda ab: -stencil(d2_energy, at, lambda x, s, ab=ab: strained(x, ab[0], ab[1], s), h), comps))
            err = max(abs(r - W[a, b]) for r, (a, b) in zip(res, comps))
            tag = ' <- nominal' if h == NOM_HS else ''
            print(f'  virial h={h:<7} max|W_fd-W_gpu|={err:.3e} eV  rel={err/np.abs(W).max():.2e}{tag}', flush=True)
