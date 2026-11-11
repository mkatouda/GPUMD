#!/usr/bin/env python3
"""A4 gate (DR-005, VER-0035): `nep` prediction vs gpumd static runs on the same model and structures.

Usage: ct_a4_compare.py NEP GPUMD WORKDIR [--structures 0,1,3,4] [--no-ct] [--charge-mode 2]
       [--rc-phi 4] [--nt 2] [--seed 1] [--kspace ewald]
Writes nep.txt (new format), nep.in (prediction 1) and train.xyz with random labels (synthetic) into
WORKDIR/nep, runs `nep`, then one gpumd static run per structure in WORKDIR/gpumd_<i> (time_step 0,
`kspace ewald`, dump_xyz precision double charge force bec) and prints RMS / max differences of
charge, BEC, forces, energy per atom and virial per atom. The same new-format nep.txt is read by gpumd
(converted internally), which also tests the nep.txt writer/reader layout.
`compare()` can also be used on an existing prediction directory (e.g. a trained nep.txt).
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


def parse_xyz(path):
    L = open(path).read().split("\n")
    n = int(L[0])
    head = L[1]
    energy = float(head.split("energy=")[1].split()[0])
    vir = np.array([float(x) for x in head.split('virial="')[1].split('"')[0].split()]).reshape(3, 3)
    props = head.split("Properties=")[1].split()[0].split(":")
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    rows = [l.split() for l in L[2:2 + n]]
    get = lambda name: np.array([[float(x) for x in r[col[name][0]:col[name][0] + col[name][1]]] for r in rows])
    return dict(energy=energy, virial=vir, forces=get("forces"), charge=get("charge")[:, 0], bec=get("bec"))


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s failed in %s:\n%s" % (cmd, cwd, (r.stdout + r.stderr)[-1200:]))
    return r.stdout


def stats(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))


def compare(nep, gpumd, nd, W, structs, idx, mk, kspace="ewald", verbose=True):
    """nd: directory with nep.txt, nep.in (prediction 1) and train.xyz of `structs`; W: gpumd scratch."""
    run([os.path.abspath(nep)], nd)
    bec_n = np.loadtxt(os.path.join(nd, "bec_train.out"), ndmin=2)[:, :9]
    f_n = np.loadtxt(os.path.join(nd, "force_train.out"), ndmin=2)[:, :3]
    q_n = np.loadtxt(os.path.join(nd, "charge_train.out"), ndmin=1)
    e_n = np.loadtxt(os.path.join(nd, "energy_train.out"), ndmin=2)[:, 0]
    v_n = np.loadtxt(os.path.join(nd, "virial_train.out"), ndmin=2)[:, :6]
    off = 0
    rows = []
    allb, allf, allq, allv = [], [], [], []
    for k, (pos, types, box) in enumerate(structs):
        n = len(pos)
        d = os.path.join(W, "gpumd_%d" % idx[k])
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        shutil.copy(os.path.join(nd, "nep.txt"), d)
        mk.write_extxyz(os.path.join(d, "model.xyz"), pos, types, box)
        open(os.path.join(d, "run.in"), "w").write(
            "potential nep.txt\nkspace %s\nensemble nve\ntime_step 0\n"
            "dump_xyz 1 d.xyz precision double charge force bec\nrun 1\n" % kspace)
        run([os.path.abspath(gpumd)], d)
        g = parse_xyz(os.path.join(d, "d.xyz"))
        sl = slice(off, off + n)
        vg = g["virial"]
        vg6 = np.array([vg[0, 0], vg[1, 1], vg[2, 2], 0.5 * (vg[0, 1] + vg[1, 0]),
                        0.5 * (vg[1, 2] + vg[2, 1]), 0.5 * (vg[2, 0] + vg[0, 2])])
        rb = stats(bec_n[sl], g["bec"])
        rf = stats(f_n[sl], g["forces"])
        rq = stats(q_n[sl], g["charge"])
        re = abs(e_n[k] - g["energy"] / n)
        rv = stats(v_n[k], vg6 / n)
        rows.append((idx[k], n, rb, rf, rq, re, rv, float(np.sqrt(np.mean(g["bec"] ** 2))),
                     float(np.sqrt(np.mean(g["forces"] ** 2)))))
        allb.append(bec_n[sl] - g["bec"])
        allf.append(f_n[sl] - g["forces"])
        allq.append(q_n[sl] - g["charge"])
        allv.append(v_n[k] - vg6 / n)
        off += n
    if verbose:
        print("struct  N   BEC rms/max      force rms/max    charge rms/max   dE/atom      "
              "dvirial/atom rms/max   rmsBEC rmsF")
        for r in rows:
            print("%5d %3d  %.2e/%.2e  %.2e/%.2e  %.2e/%.2e  %.2e  %.2e/%.2e   %.3f %.3f" % (
                r[0], r[1], r[2][0], r[2][1], r[3][0], r[3][1], r[4][0], r[4][1], r[5], r[6][0], r[6][1],
                r[7], r[8]))
    rms = lambda l: float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in l]) ** 2)))
    res = dict(bec_rms=rms(allb), force_rms=rms(allf), charge_rms=rms(allq), virial_rms=rms(allv),
               max_dE_per_atom=max(r[5] for r in rows), rms_bec=max(r[7] for r in rows),
               rms_force=max(r[8] for r in rows),
               max_bec=max(r[2][1] for r in rows), max_force=max(r[3][1] for r in rows))
    if verbose:
        print("ALL: BEC rms %.3e  force rms %.3e  charge rms %.3e  virial/atom rms %.3e  max dE/atom %.3e" % (
            res["bec_rms"], res["force_rms"], res["charge_rms"], res["virial_rms"], res["max_dE_per_atom"]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("nep")
    ap.add_argument("gpumd")
    ap.add_argument("workdir")
    ap.add_argument("--structures", default="0,1,3,4")
    ap.add_argument("--no-ct", action="store_true")
    ap.add_argument("--charge-mode", type=int, default=2)
    ap.add_argument("--rc-phi", type=float, default=4.0)
    ap.add_argument("--rc-radial", type=float, default=4.0)
    ap.add_argument("--nt", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kspace", default="ewald")
    a = ap.parse_args(argv)
    mk = mt.load_mk()
    W = os.path.abspath(a.workdir)
    shutil.rmtree(W, ignore_errors=True)
    nd = os.path.join(W, "nep")
    os.makedirs(nd)
    idx = [int(s) for s in a.structures.split(",")]
    structs = mt.make_structures(mk, idx, a.nt)
    with_ct = not a.no_ct
    mt.write_model(os.path.join(nd, "nep.txt"), mk, a.seed, a.rc_radial, 3.5, a.rc_phi, with_ct, a.nt,
                   a.charge_mode)
    mt.write_nep_in(os.path.join(nd, "nep.in"), mk, a.rc_radial, 3.5, a.rc_phi, with_ct, a.charge_mode,
                    a.nt, 1)
    mt.write_train_xyz(os.path.join(nd, "train.xyz"), mk, structs)
    compare(a.nep, a.gpumd, nd, W, structs, idx, mk, a.kspace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
