#!/usr/bin/env python3
"""Synthetic inputs for the charge_transfer NEP trainer (DR-005 stage 3a, VER-0035).

Synthetic data only (random reproducible weights/structures; no external model or data).
Weights and structures come from the VER-0032 generator (make_ct_model.py); this script
(1) writes the model in the NEW nep.txt format that `nep` reads and gpumd converts
    (header fields + `@ parameters`, `@ q_scaler`, `@ phi_parameters` blocks),
(2) writes a train.xyz with energy / virial / forces / bec labels (random or from a given array),
(3) writes a matching nep.in.

new-format nep.txt for charge_transfer (qnep2/qnep3):
  nep nt El1 El2 ...; format_version 1; model qnep2; cutoff rcr rca MNr MNa;
  charge_transfer rc_phi        <- new field
  n_max, basis_size, l_max, ann, energy_reference
  @ parameters N   (ANN block, then descriptor c in type-pair-major order)
  @ q_scaler dim
  @ phi_parameters npairs*(basis_size_radial+1)   <- new block, pair-major (upper triangle)

Usage: make_ct_trainer_inputs.py OUTDIR [--seed 1] [--structures 1,3] [--rc-phi 4] [--no-ct]
       [--charge-mode 2] [--prediction 1] [--label-seed 7]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MK_PATH = os.path.join(
    HERE, "..", "..", "VER-0032-qnep-charge-transfer-gpu", "scripts", "make_ct_model.py"
)


def load_mk():
    spec = importlib.util.spec_from_file_location("make_ct_model", MK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legacy_to_new(mk, text, nt, rc_phi, with_ct, charge_mode):
    lines = text.strip().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("ANN"))
    vals = np.array([float(x) for x in lines[k + 1:]])
    dim = (mk.N_MAX_RADIAL + 1) + (mk.N_MAX_ANGULAR + 1) * (mk.L_MAX + mk.HAS_Q_222 + mk.HAS_Q_1111)
    n_ann = (dim + 3) * mk.NEURONS * nt + 2
    nb_r = (mk.N_MAX_RADIAL + 1) * (mk.BASIS_RADIAL + 1)
    nb_a = (mk.N_MAX_ANGULAR + 1) * (mk.BASIS_ANGULAR + 1)
    nsq = nt * nt
    ann = vals[:n_ann]
    c = vals[n_ann:n_ann + nsq * (nb_r + nb_a)]
    q_scaler = vals[n_ann + nsq * (nb_r + nb_a):n_ann + nsq * (nb_r + nb_a) + dim]
    phi = vals[n_ann + nsq * (nb_r + nb_a) + dim:]
    # legacy basis-major -> type-pair-major (separately radial and angular)
    c_r = c[:nsq * nb_r].reshape(nb_r, nsq).T.reshape(-1)
    c_a = c[nsq * nb_r:].reshape(nb_a, nsq).T.reshape(-1)
    params = np.concatenate([ann, c_r, c_a])
    return params, q_scaler, (phi if with_ct else None)


def write_model(path, mk, seed, rc_radial, rc_angular, rc_phi, with_ct=True, nt=2, charge_mode=2):
    text = mk.make_model_text(seed, rc_radial, rc_angular, rc_phi, with_ct, nt=nt, charge_mode=charge_mode)
    params, q_scaler, phi = legacy_to_new(mk, text, nt, rc_phi, with_ct, charge_mode)
    write_model_arrays(path, mk, params, q_scaler, phi, rc_radial, rc_angular, rc_phi, nt, charge_mode)


def select_types(mk, params, q_scaler, phi, nt, keep):
    """independent (numpy) sub-model of the types in `keep` from new-format arrays (the oracle of the
    gpumd species compaction)."""
    dim = len(q_scaler)
    per_type = (dim + 3) * mk.NEURONS
    nb_r = (mk.N_MAX_RADIAL + 1) * (mk.BASIS_RADIAL + 1)
    nb_a = (mk.N_MAX_ANGULAR + 1) * (mk.BASIS_ANGULAR + 1)
    n_ann = per_type * nt + 2
    out = [params[t * per_type:(t + 1) * per_type] for t in keep] + [params[per_type * nt:n_ann]]
    c_r = params[n_ann:n_ann + nt * nt * nb_r].reshape(nt * nt, nb_r)
    c_a = params[n_ann + nt * nt * nb_r:].reshape(nt * nt, nb_a)
    out += [c_r[t1 * nt + t2] for t1 in keep for t2 in keep]
    out += [c_a[t1 * nt + t2] for t1 in keep for t2 in keep]
    phi_sel = None
    if phi is not None:
        per = mk.BASIS_RADIAL + 1
        ph = phi.reshape(-1, per)
        pidx = lambda a, b: a * nt - a * (a - 1) // 2 + (b - a)
        phi_sel = np.concatenate([ph[pidx(keep[i], keep[j])] for i in range(len(keep)) for j in range(i, len(keep))])
    return np.concatenate(out), q_scaler, phi_sel


def write_model_arrays(path, mk, params, q_scaler, phi, rc_radial, rc_angular, rc_phi, nt, charge_mode):
    with_ct = phi is not None
    out =["nep %d %s" % (nt, " ".join(mk.SYMBOLS[:nt])), "format_version 1", "model qnep%d" % charge_mode,
           "cutoff %g %g 100 50" % (rc_radial, rc_angular)]
    if with_ct:
        out.append("charge_transfer %g" % rc_phi)
    out += ["n_max %d %d" % (mk.N_MAX_RADIAL, mk.N_MAX_ANGULAR),
            "basis_size %d %d" % (mk.BASIS_RADIAL, mk.BASIS_ANGULAR),
            "l_max %d %d %d" % (mk.L_MAX, mk.HAS_Q_222, mk.HAS_Q_1111),
            "ann %d 0" % mk.NEURONS,
            "energy_reference " + " ".join(["0"] * nt)]
    for name, arr in (("parameters", params), ("q_scaler", q_scaler), ("phi_parameters", phi)):
        if arr is None:
            continue
        out.append("@ %s %d" % (name, len(arr)))
        out += ["%.9g" % v for v in arr]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def write_nep_in(path, mk, rc_radial, rc_angular, rc_phi, with_ct, charge_mode, nt, prediction, extra=""):
    lines = ["type %d %s" % (nt, " ".join(mk.SYMBOLS[:nt])),
             "model qnep%d" % charge_mode,
             "cutoff %g %g" % (rc_radial, rc_angular),
             "n_max %d %d" % (mk.N_MAX_RADIAL, mk.N_MAX_ANGULAR),
             "basis_size %d %d" % (mk.BASIS_RADIAL, mk.BASIS_ANGULAR),
             "l_max %d %d %d" % (mk.L_MAX, mk.HAS_Q_222, mk.HAS_Q_1111),
             "neuron %d" % mk.NEURONS]
    if with_ct:
        lines.append("charge_transfer %g" % rc_phi)
    if prediction:
        lines.append("prediction 1")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n" + extra)


def make_structures(mk, indices, nt, seed_offset=0):
    out = []
    for index in indices:
        name, natoms, box = mk.STRUCTURES[index]
        pos, types = mk.make_structure(natoms, box, 100 + index + seed_offset, nt=nt)
        out.append((pos, types, box))
    return out


def write_train_xyz(path, mk, structures, labels=None, label_seed=7):
    """labels: list of dicts(energy, virial(9), forces(N,3), bec(N,9)) or None for random."""
    rng = np.random.default_rng(label_seed)
    with open(path, "w") as f:
        for i, (pos, types, box) in enumerate(structures):
            n = len(pos)
            if labels is None:
                lab = dict(energy=float(rng.normal(0, 1) * n), virial=rng.normal(0, 1, 9),
                           forces=rng.normal(0, 0.5, (n, 3)), bec=rng.normal(0, 0.5, (n, 9)))
            else:
                lab = labels[i]
            f.write("%d\n" % n)
            f.write('Lattice="%g 0 0 0 %g 0 0 0 %g" energy=%.9g virial="%s" '
                    'Properties=species:S:1:pos:R:3:forces:R:3:bec:R:9 pbc="T T T"\n'
                    % (box, box, box, lab["energy"], " ".join("%.9g" % v for v in lab["virial"])))
            for a in range(n):
                f.write("%s %.10f %.10f %.10f %s %s\n" % (
                    mk.SYMBOLS[types[a]], pos[a][0], pos[a][1], pos[a][2],
                    " ".join("%.9g" % v for v in lab["forces"][a]),
                    " ".join("%.9g" % v for v in lab["bec"][a])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--structures", default="1,3")
    ap.add_argument("--nt", type=int, default=2)
    ap.add_argument("--rc-radial", type=float, default=4.0)
    ap.add_argument("--rc-angular", type=float, default=3.5)
    ap.add_argument("--rc-phi", type=float, default=4.0)
    ap.add_argument("--charge-mode", type=int, default=2)
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--prediction", type=int, default=1)
    ap.add_argument("--label-seed", type=int, default=7)
    a = ap.parse_args(argv)
    mk = load_mk()
    os.makedirs(a.outdir, exist_ok=True)
    with_ct = not a.no_ct
    write_model(os.path.join(a.outdir, "nep.txt"), mk, a.seed, a.rc_radial, a.rc_angular, a.rc_phi,
                with_ct, a.nt, a.charge_mode)
    write_nep_in(os.path.join(a.outdir, "nep.in"), mk, a.rc_radial, a.rc_angular, a.rc_phi, with_ct,
                 a.charge_mode, a.nt, a.prediction)
    idx = [int(s) for s in a.structures.split(",")]
    write_train_xyz(os.path.join(a.outdir, "train.xyz"), mk, make_structures(mk, idx, a.nt),
                    label_seed=a.label_seed)
    print("wrote", a.outdir)


if __name__ == "__main__":
    sys.exit(main())
