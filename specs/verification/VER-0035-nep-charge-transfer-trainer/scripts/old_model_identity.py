#!/usr/bin/env python3
"""Gate 3a-1 (DR-005, VER-0035): models/keywords without charge_transfer are unchanged.

Usage: old_model_identity.py NEP_BASE NEP_NEW WORKDIR [--generations 20]
Short training runs (seed fixed, population 20) of qnep1/qnep2/qnep3 and plain nep on a synthetic
dataset (random labels) with the base binary (twice, to measure its own run-to-run noise) and the new
binary. Compares the printed parameter counts, loss.out, nep.txt and nep.restart.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_ct_trainer_inputs as mt  # noqa: E402


def setup(d, mk, case, gens):
    os.makedirs(d)
    nt = 2
    structs = mt.make_structures(mk, [0, 1, 3, 4], nt)
    mt.write_train_xyz(os.path.join(d, "train.xyz"), mk, structs)
    extra = "population 20\ngeneration %d\nseed 1\nbatch 100000\noutput_interval 5\n" % gens
    if case == "nep":
        lines = ["type 2 Na Cl", "model nep", "cutoff 4 3.5", "n_max 3 2", "basis_size 4 4", "l_max 4 0 0", "neuron 10"]
        open(os.path.join(d, "nep.in"), "w").write("\n".join(lines) + "\n" + extra)
        # plain nep reads no bec column: keep the file (extra columns are ignored) as is
    else:
        mode = int(case[-1])
        mt.write_nep_in(os.path.join(d, "nep.in"), mk, 4.0, 3.5, 4.0, False, mode, nt, 0, extra)


def run(nep, d):
    r = subprocess.run([nep], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("nep failed in %s:\n%s" % (d, (r.stdout + r.stderr)[-1500:]))
    open(os.path.join(d, "stdout.txt"), "w").write(r.stdout)
    return r.stdout


def counts(text):
    return [l.strip() for l in text.split("\n") if "number of" in l or "total number" in l or "NN architecture" in l]


def load(path):
    return np.loadtxt(path, comments="#", ndmin=2)


def load_numbers(path):
    out = []
    for l in open(path):
        for t in l.split():
            try:
                out.append(float(t))
            except ValueError:
                pass
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("workdir")
    ap.add_argument("--generations", type=int, default=20)
    a = ap.parse_args(argv)
    mk = mt.load_mk()
    W = os.path.abspath(a.workdir)
    shutil.rmtree(W, ignore_errors=True)
    for case in ["qnep1", "qnep2", "qnep3", "nep"]:
        res = {}
        for tag, exe in (("base1", a.base), ("base2", a.base), ("new", a.new)):
            d = os.path.join(W, case, tag)
            setup(d, mk, case, a.generations)
            res[tag] = (d, run(os.path.abspath(exe), d))
        same_counts = counts(res["base1"][1]) == counts(res["new"][1])
        print("%s: parameter-count lines identical: %s (%s)" % (case, same_counts, "; ".join(counts(res["new"][1])[-1:])))
        for fname in ("loss.out", "nep.txt", "nep.restart"):
            row = []
            for pair in (("base1", "base2"), ("base1", "new")):
                x = load_numbers(os.path.join(res[pair[0]][0], fname))
                y = load_numbers(os.path.join(res[pair[1]][0], fname))
                if x.shape != y.shape:
                    row.append("%s vs %s: SHAPE %s %s" % (pair[0], pair[1], x.shape, y.shape))
                    continue
                bit = filecmp_same(os.path.join(res[pair[0]][0], fname), os.path.join(res[pair[1]][0], fname))
                row.append("%s vs %s: bytes identical %s, max|diff| %.3e, max rel %.3e" % (
                    pair[0], pair[1], bit, np.max(np.abs(x - y)), np.max(np.abs(x - y) / (np.abs(x) + 1e-12))))
            print("  %-12s %s" % (fname, " | ".join(row)))
        first = load(os.path.join(res["new"][0], "loss.out"))
        print("  loss.out: first row total loss %.6g, last row %.6g" % (first[0][1], first[-1][1]))
    return 0


def filecmp_same(p, q):
    import filecmp
    return filecmp.cmp(p, q, shallow=False)


if __name__ == "__main__":
    sys.exit(main())
