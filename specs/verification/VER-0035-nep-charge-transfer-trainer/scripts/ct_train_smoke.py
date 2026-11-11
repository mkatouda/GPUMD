#!/usr/bin/env python3
"""Stage 3a-4 (DR-005, VER-0035): training smoke test of the charge_transfer qNEP in `nep`.

Usage: ct_train_smoke.py NEP GPUMD WORKDIR [--generations 40] [--charge-mode 2] [--seed 1]
Synthetic data only. A random teacher model (VER-0032 generator weights, seed 3) labels six small
periodic structures (energy, virial, forces, BEC from `nep` prediction of the teacher). A student with
`charge_transfer` is trained with SNES (population 30). Reported / checked:
  * loss.out total loss: first generation, minimum, last generation (decrease);
  * the written nep.txt (new format, `charge_transfer` field and `phi_parameters` block) is predicted by
    `nep` and by gpumd (A4 comparison on the trained model), and the prediction RMSE reproduces loss.out;
  * restart: a second run continues from nep.restart (phi tail included).
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ct_a4_compare as a4  # noqa: E402
import make_ct_trainer_inputs as mt  # noqa: E402

TRAIN_EXTRA = "population 30\ngeneration %d\nseed %d\nbatch 100000\noutput_interval 1\n"


def structures(mk, nt):
    out, idx = [], []
    for offset in (0, 1, 2):
        for index in (3, 4):
            out += mt.make_structures(mk, [index], nt, seed_offset=offset)
            idx.append(index * 10 + offset)
    return out, idx


def teacher_labels(nep, mk, workdir, structs, charge_mode, nt, rc_phi, seed=3):
    d = os.path.join(workdir, "teacher")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    mt.write_model(os.path.join(d, "nep.txt"), mk, seed, 4.0, 3.5, rc_phi, True, nt, charge_mode)
    mt.write_nep_in(os.path.join(d, "nep.in"), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 1)
    mt.write_train_xyz(os.path.join(d, "train.xyz"), mk, structs)
    a4.run([os.path.abspath(nep)], d)
    e = np.loadtxt(os.path.join(d, "energy_train.out"), ndmin=2)[:, 0]
    v = np.loadtxt(os.path.join(d, "virial_train.out"), ndmin=2)[:, :6]
    f = np.loadtxt(os.path.join(d, "force_train.out"), ndmin=2)[:, :3]
    b = np.loadtxt(os.path.join(d, "bec_train.out"), ndmin=2)[:, :9]
    labels, off = [], 0
    for k, (pos, types, box) in enumerate(structs):
        n = len(pos)
        vv = v[k] * n
        vir9 = [vv[0], vv[3], vv[5], vv[3], vv[1], vv[4], vv[5], vv[4], vv[2]]
        labels.append(dict(energy=float(e[k] * n), virial=vir9, forces=f[off:off + n], bec=b[off:off + n]))
        off += n
    return labels


def loss_rows(path):
    return np.loadtxt(path, comments="#", ndmin=2)


def run_smoke(nep, gpumd, workdir, generations=40, charge_mode=2, seed=1, rc_phi=4.0, nt=2):
    mk = mt.load_mk()
    W = os.path.abspath(workdir)
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(W)
    structs, idx = structures(mk, nt)
    labels = teacher_labels(nep, mk, W, structs, charge_mode, nt, rc_phi)
    sd = os.path.join(W, "student")
    os.makedirs(sd)
    mt.write_train_xyz(os.path.join(sd, "train.xyz"), mk, structs, labels)
    mt.write_nep_in(os.path.join(sd, "nep.in"), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 0,
                    TRAIN_EXTRA % (generations, seed))
    out = a4.run([os.path.abspath(nep)], sd)
    res = {}
    rows = loss_rows(os.path.join(sd, "loss.out"))
    res["rows"] = len(rows)
    # generation 1 only computes q_scaler (prepare_training) and reports the regularisation alone (all
    # models); the first evaluated generation is the second row
    res["loss_first"], res["loss_min"], res["loss_last"] = float(rows[1, 1]), float(rows[1:, 1].min()), float(rows[-1, 1])
    res["nparams_line"] = [l.strip() for l in out.split("\n") if "total number of parameters" in l][0]
    res["last_row"] = rows[-1]
    # prediction of the written nep.txt: nep and gpumd
    pd = os.path.join(W, "pred")
    os.makedirs(pd)
    shutil.copy(os.path.join(sd, "nep.txt"), pd)
    shutil.copy(os.path.join(sd, "train.xyz"), pd)
    mt.write_nep_in(os.path.join(pd, "nep.in"), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 1)
    res["a4_trained"] = a4.compare(nep, gpumd, pd, W, structs, idx, mk, verbose=False)
    # prediction RMSE of forces vs loss.out (column 5 = rmse_force_train, 8 = rmse_bec_train)
    f = np.loadtxt(os.path.join(pd, "force_train.out"), ndmin=2)
    res["force_rmse_pred"] = float(np.sqrt(np.mean((f[:, :3] - f[:, 3:6]) ** 2)))
    res["force_rmse_loss"] = float(rows[-1, 5])
    b = np.loadtxt(os.path.join(pd, "bec_train.out"), ndmin=2)
    res["bec_rmse_pred_per_comp"] = float(np.sqrt(np.mean((b[:, :9] - b[:, 9:18]) ** 2)))
    res["bec_rmse_loss"] = float(rows[-1, 8])
    # restart: continue from nep.restart written by the first run
    shutil.copy(os.path.join(sd, "nep.restart"), os.path.join(W, "first.restart"))
    shutil.copy(os.path.join(sd, "nep.txt"), os.path.join(W, "first.nep.txt"))
    mt.write_nep_in(os.path.join(sd, "nep.in"), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 0,
                    TRAIN_EXTRA % (5, seed))
    a4.run([os.path.abspath(nep)], sd)
    rows2 = loss_rows(os.path.join(sd, "loss.out"))
    # loss.out is appended by the second run: its generation 1 (regularisation only) is row len(rows),
    # its first evaluated generation is the next row
    res["restart_first"], res["restart_last"] = float(rows2[len(rows) + 1, 1]), float(rows2[-1, 1])
    # exact restart check. (a) normal run: the phi tail of nep.restart (mu after the last update) differs
    # from the phi block of nep.txt (best member of the last generation) by O(sigma) only;
    # (b) with all sigma set to 0 in nep.restart every population member equals mu, so the restarted run
    # must write exactly these mu (all 522 parameters and the 15 phi values) into nep.txt.
    mu, sigma = read_restart(os.path.join(W, "first.restart"))
    phi_txt = read_block(os.path.join(W, "first.nep.txt"), "phi_parameters")
    res["num_phi"] = len(phi_txt)
    res["restart_vs_txt_phi_maxdiff_normal"] = float(np.abs(mu[-len(phi_txt):] - phi_txt).max())
    res["sigma_phi_max"] = float(sigma[-len(phi_txt):].max())
    write_restart_sigma0(os.path.join(W, "first.restart"), os.path.join(sd, "nep.restart"))
    mt.write_nep_in(os.path.join(sd, "nep.in"), mk, 4.0, 3.5, rc_phi, True, charge_mode, nt, 0,
                    TRAIN_EXTRA % (3, seed))
    a4.run([os.path.abspath(nep)], sd)
    phi_new = read_block(os.path.join(sd, "nep.txt"), "phi_parameters")
    par_new = read_block(os.path.join(sd, "nep.txt"), "parameters")
    res["sigma0_phi_maxdiff"] = float(np.abs(mu[-len(phi_txt):] - phi_new).max())
    # the trainer adds the energy shift to the last bias of the energy network (index n_ann - 1) in the
    # written nep.txt only: that single entry is excluded from the comparison and reported separately
    dim = len(read_block(os.path.join(sd, "nep.txt"), "q_scaler"))
    ibias = (dim + 3) * mk.NEURONS * nt + 2 - 1
    diff = np.abs(mu[:len(par_new)] - par_new)
    res["sigma0_bias_shift"] = float(diff[ibias])
    diff[ibias] = 0.0
    res["sigma0_parameters_maxdiff"] = float(diff.max())
    return res


def read_restart(path):
    lines = open(path).read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("parameters"))
    arr = np.array([[float(x) for x in l.split()] for l in lines[k + 1:] if l.strip()])
    return arr[:, 0], arr[:, 1]


def write_restart_sigma0(src, dst):
    lines = open(src).read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("parameters"))
    out = lines[:k + 1]
    for l in lines[k + 1:]:
        if l.strip():
            out.append("%s 0" % l.split()[0])
    open(dst, "w").write("\n".join(out) + "\n")


def read_block(path, name):
    lines = open(path).read().split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("@ " + name + " "))
    n = int(lines[k].split()[2])
    return np.array([float(x) for x in lines[k + 1:k + 1 + n]])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("nep")
    ap.add_argument("gpumd")
    ap.add_argument("workdir")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--charge-mode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    r = run_smoke(a.nep, a.gpumd, a.workdir, a.generations, a.charge_mode, a.seed)
    for k, v in r.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
