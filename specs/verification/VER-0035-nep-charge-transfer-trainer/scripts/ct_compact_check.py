#!/usr/bin/env python3
"""Species compaction of a charge_transfer nep.txt in gpumd (DR-005, VER-0035).

gpumd reduces a potential file to the species present in model.xyz (utilities/nep_io/compact_nep.cu for
legacy files, NEP_Model_File::read_and_compact for new-format files). Test: a 3-type charge_transfer
model (Na Cl Mg) evaluated on a Na/Cl-only structure (compaction active) against the 2-type model
that numpy selects from the same arrays (select_types, an oracle independent of the C++ compaction),
evaluated on the same structure (no compaction). Both file formats. Prints max differences of
energy, charge, forces, BEC; the gate is bit-level agreement up to float noise (<= 1e-6).
Usage: ct_compact_check.py GPUMD WORKDIR
"""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ct_a4_compare as a4  # noqa: E402
import make_ct_trainer_inputs as mt  # noqa: E402


def run_gpumd(gpumd, d, model_text_path, pos, types, box, mk):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(model_text_path, os.path.join(d, "nep.txt"))
    mk.write_extxyz(os.path.join(d, "model.xyz"), pos, types, box)
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\nkspace ewald\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 d.xyz precision double charge force bec\nrun 1\n")
    r = subprocess.run([os.path.abspath(gpumd)], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-800:])
    return a4.parse_xyz(os.path.join(d, "d.xyz"))


def check(gpumd, workdir, fmt="new", structure=1, keep=(0, 1), seed=1):
    mk = mt.load_mk()
    W = os.path.abspath(workdir)
    os.makedirs(W, exist_ok=True)
    nt = 3
    text = mk.make_model_text(seed, 4.0, 3.5, 4.0, True, nt=nt, charge_mode=2)
    params, q_scaler, phi = mt.legacy_to_new(mk, text, nt, 4.0, True, 2)
    full = os.path.join(W, "full_%s.txt" % fmt)
    if fmt == "legacy":
        open(full, "w").write(text)
    else:
        mt.write_model_arrays(full, mk, params, q_scaler, phi, 4.0, 3.5, 4.0, nt, 2)
    p2, q2, phi2 = mt.select_types(mk, params, q_scaler, phi, nt, list(keep))
    sub = os.path.join(W, "sub_%s.txt" % fmt)
    mt.write_model_arrays(sub, mk, p2, q2, phi2, 4.0, 3.5, 4.0, len(keep), 2)
    name, natoms, box = mk.STRUCTURES[structure]
    pos, types = mk.make_structure(natoms, box, 100 + structure, nt=len(keep))
    g_full = run_gpumd(gpumd, os.path.join(W, "run_full_" + fmt), full, pos, types, box, mk)
    g_sub = run_gpumd(gpumd, os.path.join(W, "run_sub_" + fmt), sub, pos, types, box, mk)
    res = dict(
        dE=abs(g_full["energy"] - g_sub["energy"]),
        dq=float(np.abs(g_full["charge"] - g_sub["charge"]).max()),
        dF=float(np.abs(g_full["forces"] - g_sub["forces"]).max()),
        dZ=float(np.abs(g_full["bec"] - g_sub["bec"]).max()),
        rms_Z=float(np.sqrt(np.mean(g_sub["bec"] ** 2))),
        rms_F=float(np.sqrt(np.mean(g_sub["forces"] ** 2))),
    )
    return res


def main(argv=None):
    gpumd, workdir = sys.argv[1], sys.argv[2]
    for fmt in ("new", "legacy"):
        print(fmt, check(gpumd, workdir, fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
