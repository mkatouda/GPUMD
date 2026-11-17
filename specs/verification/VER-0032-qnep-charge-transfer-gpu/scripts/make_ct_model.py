#!/usr/bin/env python3
"""Write a synthetic charge_transfer qNEP model (new NEP model format) and small test structures.

DR-005 stage 2b (VER-0032).  Synthetic data only: random reproducible weights, no trained model.

The legacy text format (nep4_charge2) was removed upstream.  The random weights are still drawn
in the legacy layout (same RNG sequence and values as before), converted by the upstream
potentials/nep/nep_model_converter.py (qnep2/qnep3 header, channel-major descriptor order), and
the two charge_transfer items are added, which the converter does not know:

  charge_transfer <rc_phi>            field (one line, before the first `@` block)
  @ phi_parameters <num_pairs*(basis_size_radial+1)>    block, at the end of the file

Reader: src/utilities/nep_io/nep_model_file_charge.cu (qnep2/qnep3 only; 0 < rc_phi <= rc_radial).
Legacy value order before conversion:
    1. ANN block, num_para_ann = (dim + 3) * num_neurons * num_types + 2 values:
         for each type t: w0[t] (num_neurons*dim), b0[t] (num_neurons), w1[t] (num_neurons*2)
         then sqrt_epsilon_inf (1), b1 (1)
    2. descriptor coefficients c (legacy basis-major order; the converter reorders them)
    3. q_scaler, dim values
    4. phi values (drawn last, after q_scaler).

phi pair ordering (symmetric, upper triangle row-major, t1 <= t2):
    p(t1, t2) = t1 * num_types - t1 * (t1 - 1) / 2 + (t2 - t1)
    2 types: p(0,0)=0, p(0,1)=1, p(1,1)=2.  phi_{t1 t2} = phi_{t2 t1}.
    Coefficients of pair p: c[p*(basis_size_radial+1) + k], k = 0 .. basis_size_radial, on the
    radial basis f_k(r) fc(r) with cutoff rc_phi: phi(r) = sum_k c_k f_k(r) fc(r).

Usage: make_ct_model.py OUTDIR [--seed 1] [--rc-radial 4.0] [--rc-phi 4.0] [--no-ct]
"""
import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

NUM_TYPES = 2
SYMBOLS = ["Na", "Cl", "Mg"]
N_MAX_RADIAL = 3
N_MAX_ANGULAR = 2
BASIS_RADIAL = 4
BASIS_ANGULAR = 4
L_MAX = 4
HAS_Q_222 = 0
HAS_Q_1111 = 0
NEURONS = 10
PHI_MAX = 0.3  # phi coefficients uniform in [0, PHI_MAX]; keeps |q| of the dense thin cells near 1 e


def pair_index(t1, t2, nt=NUM_TYPES):
    t1, t2 = min(t1, t2), max(t1, t2)
    return t1 * nt - t1 * (t1 - 1) // 2 + (t2 - t1)


def _load_converter():
    for parent in Path(__file__).resolve().parents:
        path = parent / "potentials" / "nep" / "nep_model_converter.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("nep_model_converter", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("potentials/nep/nep_model_converter.py not found")


def make_legacy_parts(seed, rc_radial, rc_angular, nt=NUM_TYPES, charge_mode=2):
    """Legacy-layout text (no charge_transfer) and the phi values (drawn after q_scaler)."""
    rng = np.random.default_rng(seed)
    num_L = L_MAX + HAS_Q_222 + HAS_Q_1111
    dim = (N_MAX_RADIAL + 1) + (N_MAX_ANGULAR + 1) * num_L
    values = []
    for _ in range(nt):
        values += list(rng.normal(0.0, 1.0 / np.sqrt(dim), NEURONS * dim))  # w0
        values += list(rng.normal(0.0, 0.1, NEURONS))  # b0
        values += list(rng.normal(0.0, 1.5, NEURONS * 2))  # w1 (energy, charge)
    values.append(1.0 + 0.2 * rng.random())  # sqrt_epsilon_inf
    values.append(rng.normal(0.0, 0.1))  # b1
    n_c = nt * nt * ((N_MAX_RADIAL + 1) * (BASIS_RADIAL + 1) + (N_MAX_ANGULAR + 1) * (BASIS_ANGULAR + 1))
    values += list(rng.uniform(-0.5, 0.5, n_c))
    values += list(rng.uniform(0.5, 2.0, dim))  # q_scaler
    num_pairs = nt * (nt + 1) // 2
    phi = list(rng.uniform(0.0, PHI_MAX, num_pairs * (BASIS_RADIAL + 1)))
    lines = [
        # the converter has no charge3 tag; qnep1/2/3 share one block layout, so write charge2
        # here and rename the model after conversion
        "nep4_charge2 %d %s" % (nt, " ".join(SYMBOLS[:nt])),
        "cutoff %g %g 100 50" % (rc_radial, rc_angular),
        "n_max %d %d" % (N_MAX_RADIAL, N_MAX_ANGULAR),
        "basis_size %d %d" % (BASIS_RADIAL, BASIS_ANGULAR),
        "l_max %d %d %d" % (L_MAX, HAS_Q_222, HAS_Q_1111),
        "ANN %d 0" % NEURONS,
    ]
    lines += ["%.15e" % v for v in values]
    return "\n".join(lines) + "\n", phi


def make_model_text(seed, rc_radial, rc_angular, rc_phi, with_ct=True, nt=NUM_TYPES, charge_mode=2):
    """New-format (qnep2/qnep3) model text; with_ct adds charge_transfer and phi_parameters."""
    legacy, phi = make_legacy_parts(seed, rc_radial, rc_angular, nt, charge_mode)
    conv = _load_converter()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy_nep.txt"
        path.write_text(legacy)
        text = conv.model_text(conv.read_model(path))
    text = text.replace("model qnep2\n", "model qnep%d\n" % charge_mode, 1)
    if with_ct:
        lines = text.split("\n")
        i = next(k for k, l in enumerate(lines) if l.startswith("@ "))
        lines.insert(i, "charge_transfer %g" % rc_phi)
        text = "\n".join(lines).rstrip("\n") + "\n"
        text += "@ phi_parameters %d\n" % len(phi) + "".join("%.15e\n" % v for v in phi)
    return text


def make_structure(natoms, box, seed, min_dist=1.6, nt=NUM_TYPES):
    rng = np.random.default_rng(seed)
    pos = []
    tries = 0
    while len(pos) < natoms:
        tries += 1
        if tries > 200000:
            raise RuntimeError("cannot place atoms")
        p = rng.random(3) * box
        ok = True
        for q in pos:
            d = p - q
            d -= box * np.round(d / box)
            if np.dot(d, d) < min_dist**2:
                ok = False
                break
        if ok:
            pos.append(p)
    types = rng.integers(0, nt, natoms)
    types[0], types[-1] = 0, 1  # all types always present
    if nt > 2:
        types[1] = 2
    return np.array(pos), types


def write_extxyz(path, pos, types, box):
    with open(path, "w") as f:
        f.write("%d\n" % len(pos))
        f.write(
            'Lattice="%g 0 0 0 %g 0 0 0 %g" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            % (box, box, box)
        )
        for p, t in zip(pos, types):
            f.write("%s %.10f %.10f %.10f\n" % (SYMBOLS[t], p[0], p[1], p[2]))


# (name, natoms, cubic box edge in A); rc_radial = 4: thick = box >= 2.5 rc, thin = box < 2.5 rc
STRUCTURES = [
    ("thick_n8_L13", 8, 13.0),
    ("thick_n45_L14", 45, 14.0),
    ("thick_n59_L16", 59, 16.0),
    ("thin_n12_L7.5", 12, 7.5),
    ("thin_n10_L6", 10, 6.0),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rc-radial", type=float, default=4.0)
    ap.add_argument("--rc-angular", type=float, default=3.5)
    ap.add_argument("--rc-phi", type=float, default=None)
    ap.add_argument("--no-ct", action="store_true", help="omit the charge_transfer field and phi block")
    a = ap.parse_args(argv)
    rc_phi = a.rc_phi if a.rc_phi is not None else a.rc_radial
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "nep.txt"), "w") as f:
        f.write(make_model_text(a.seed, a.rc_radial, a.rc_angular, rc_phi, not a.no_ct))
    for i, (name, n, box) in enumerate(STRUCTURES):
        pos, types = make_structure(n, box, a.seed * 100 + i)
        write_extxyz(os.path.join(a.outdir, name + ".xyz"), pos, types, box)
    print("wrote", a.outdir)


if __name__ == "__main__":
    sys.exit(main())
