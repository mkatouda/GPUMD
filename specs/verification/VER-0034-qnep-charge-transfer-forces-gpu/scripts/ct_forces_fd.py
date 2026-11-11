#!/usr/bin/env python3
"""DR-005 stage 2c (VER-0034): gpumd forces / virial of a charge_transfer qNEP vs finite differences
of gpumd's own total potential energy (static runs, `time_step 0`, `dump_xyz precision double`).

Model/structures: VER-0032 make_ct_model.py (synthetic random weights).  The "old" reference model is
the same generator without the charge_transfer line (an ordinary qNEP, nep4_charge2) on the same
structures: its FD error is the noise floor (float kernels, FD truncation) to compare with.
Forces: central FD of the energy, F = -dE/dr, step `delta` (A).  Virial: central FD of the energy
under the homogeneous strain F = 1 + e E_ab (r -> F r, lattice -> F lattice), W_ab = -dE/de (ab symmetric
in a static periodic cell), compared with the dumped total virial.
Usage: ct_forces_fd.py GPUMD WORKDIR [--structure NAME] [--delta 0.005] [--kspace ewald] [--old]
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "VER-0032-qnep-charge-transfer-gpu", "scripts"))
import make_ct_model as mk  # noqa: E402


def write_xyz(path, pos, types, lat, group=False):
    with open(path, "w") as f:
        f.write("%d\n" % len(pos))
        f.write(
            'Lattice="%s" Properties=species:S:1:pos:R:3%s pbc="T T T"\n'
            % (" ".join("%.15g" % x for x in np.asarray(lat).reshape(-1)), ":group:I:1" if group else "")
        )
        for p, t in zip(pos, types):
            f.write("%s %.15f %.15f %.15f%s\n" % (mk.SYMBOLS[t], p[0], p[1], p[2], " 0" if group else ""))


def run_static(gpumd, workdir, tag, model_text, pos, types, lat, kspace="ewald", extra_run="", steps=1,
               time_step=0, env_extra=None, props="charge force", efield=None):
    d = os.path.join(workdir, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "nep.txt"), "w").write(model_text)
    write_xyz(os.path.join(d, "model.xyz"), pos, types, lat, group=efield is not None)
    if efield is not None:
        extra_run += "\nadd_efield 0 0 %.15g %.15g %.15g bec" % tuple(efield)
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\nkspace %s\n%s\nensemble nve\ntime_step %s\n"
        "dump_xyz 1 d.xyz precision double %s\n"
        "dump_thermo 1\nrun %d\n" % (kspace, extra_run, time_step, props, steps)
    )
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run([gpumd], cwd=d, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError("gpumd failed in %s:\n%s" % (d, (r.stdout + r.stderr)[-800:]))
    return parse_xyz(os.path.join(d, "d.xyz"))


def parse_xyz(path, frame=0):
    L = open(path).read().split("\n")
    n = int(L[0])
    stride = n + 2
    head = L[frame * stride + 1]
    energy = float(head.split("energy=")[1].split()[0])
    vir = np.array([float(x) for x in head.split('virial="')[1].split('"')[0].split()]).reshape(3, 3)
    props = head.split("Properties=")[1].split()[0].split(":")
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    rows = [l.split() for l in L[frame * stride + 2: frame * stride + 2 + n]]
    get = lambda name: np.array([[float(x) for x in r[col[name][0]:col[name][0] + col[name][1]]] for r in rows])
    out = {"energy": energy, "virial": vir, "pos": get("pos")}
    for k in ("forces", "charge", "bec"):
        if k in col:
            out[k] = get(k)[:, 0] if k == "charge" else get(k)
    return out


def nve_drift(gpumd, workdir, text, pos, types, lat, steps=2000, dt_fs=0.5, temp=300, kspace="ewald",
              tag="nve"):
    """short NVE run (`velocity T`); thermo.out columns: T K U ...; returns drift of the linear fit of
    K + U over the run and max |E - E0| (eV)."""
    d = os.path.join(workdir, tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "nep.txt"), "w").write(text)
    write_xyz(os.path.join(d, "model.xyz"), pos, types, lat)
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\nkspace %s\nvelocity %g\nensemble nve\ntime_step %g\n"
        "dump_thermo 20\nrun %d\n" % (kspace, temp, dt_fs, steps)
    )
    r = subprocess.run([gpumd], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gpumd failed in %s:\n%s" % (d, (r.stdout + r.stderr)[-800:]))
    th = np.loadtxt(os.path.join(d, "thermo.out"))
    etot = th[:, 1] + th[:, 2]  # K + U (eV)
    t = np.arange(len(etot))
    slope = np.polyfit(t, etot, 1)[0] * (len(etot) - 1)
    return {"E0": float(etot[0]), "drift_fit": float(slope), "max_dev": float(np.max(np.abs(etot - etot[0]))),
            "kin_mean": float(np.mean(th[:, 1]))}


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def force_fd(gpumd, workdir, text, pos, types, lat, delta, atoms, kspace):
    base = run_static(gpumd, workdir, "base", text, pos, types, lat, kspace)
    F_fd = np.zeros((len(atoms), 3))
    for i, k in enumerate(atoms):
        for a in range(3):
            e = []
            for s in (+1, -1):
                p2 = pos.copy()
                p2[k, a] += s * delta
                e.append(run_static(gpumd, workdir, "fd", text, p2, types, lat, kspace)["energy"])
            F_fd[i, a] = -(e[0] - e[1]) / (2 * delta)
    err = base["forces"][atoms] - F_fd
    return base, {"rms_F": rms(F_fd), "rms_err": rms(err), "max_err": float(np.max(np.abs(err)))}


STRAINS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def virial_fd(gpumd, workdir, text, pos, types, lat, delta, kspace, base):
    err, ref = [], []
    for a, b in STRAINS:
        e = []
        for s in (+1, -1):
            Fm = np.eye(3)
            Fm[a, b] += s * delta
            # r -> F r ; lattice vectors are rows of lat
            e.append(run_static(gpumd, workdir, "fdv", text, pos @ Fm.T, types, lat @ Fm.T, kspace)["energy"])
        W_fd = -(e[0] - e[1]) / (2 * delta)
        W_gpu = 0.5 * (base["virial"][a, b] + base["virial"][b, a])
        err.append(W_gpu - W_fd)
        ref.append(W_fd)
    return {"rms_W": rms(ref), "rms_err": rms(err), "max_err": float(np.max(np.abs(err)))}


TILT = np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [-0.08, 0.10, 1.0]])  # rows = lattice vectors / box


def setup(structure, seed=1, old=False, rc_radial=4.0, rc_phi=4.0, charge_mode=2, triclinic=False, nt=2):
    idx, (name, n, box) = next((i, s) for i, s in enumerate(mk.STRUCTURES) if s[0] == structure)
    pos, types = mk.make_structure(n, box, seed * 100 + idx, nt=nt)
    text = mk.make_model_text(seed, rc_radial, 3.5, rc_phi, not old, nt, charge_mode)
    lat = np.eye(3) * box
    if triclinic:  # same fractional coordinates in a sheared cell
        lat = TILT * box
        pos = (pos / box) @ lat
    return text, pos, types, lat, name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gpumd")
    ap.add_argument("workdir")
    ap.add_argument("--structure", default="thin_n12_L7.5")
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--kspace", default="ewald")
    ap.add_argument("--max-atoms", type=int, default=0)
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--mode", type=int, default=2, help="charge_mode (nep4_charge2 / nep4_charge3 header)")
    ap.add_argument("--triclinic", action="store_true")
    ap.add_argument("--nve", type=int, default=0, help="if > 0: NVE run of this many steps instead of FD")
    ap.add_argument("--rc-radial", type=float, default=4.0)
    ap.add_argument("--rc-phi", type=float, default=4.0)
    a = ap.parse_args(argv)
    gpumd, work = os.path.abspath(a.gpumd), os.path.abspath(a.workdir)
    text, pos, types, lat, name = setup(a.structure, 1, a.old, a.rc_radial, a.rc_phi, a.mode, a.triclinic)
    n = len(pos)
    if a.nve:
        r = nve_drift(gpumd, work, text, pos, types, lat, a.nve, kspace=a.kspace)
        print("%s %s NVE %d steps: E0=%.6f drift_fit=%.3e max_dev=%.3e <K>=%.4f" % (
            name, "OLD" if a.old else "CT", a.nve, r["E0"], r["drift_fit"], r["max_dev"], r["kin_mean"]))
        return
    atoms = list(range(n))
    if a.max_atoms and a.max_atoms < n:
        atoms = [int(x) for x in np.linspace(0, n - 1, a.max_atoms)]
    base, fr = force_fd(gpumd, work, text, pos, types, lat, a.delta, atoms, a.kspace)
    vr = virial_fd(gpumd, work, text, pos, types, lat, a.delta, a.kspace, base)
    print("%s %s kspace=%s delta=%g rc=%g/%g atoms=%d" % (
        name + (" tri" if a.triclinic else "") + " mode%d" % a.mode, "OLD" if a.old else "CT", a.kspace, a.delta,
        a.rc_radial, a.rc_phi, len(atoms)))
    print("  force: rms_F=%.4g rms_err=%.3e max_err=%.3e" % (fr["rms_F"], fr["rms_err"], fr["max_err"]))
    print("  virial: rms_W=%.4g rms_err=%.3e max_err=%.3e" % (vr["rms_W"], vr["rms_err"], vr["max_err"]))
    print("  sum F = %s ; E = %.10f" % (np.array2string(base["forces"].sum(0), precision=2), base["energy"]))


if __name__ == "__main__":
    main()
