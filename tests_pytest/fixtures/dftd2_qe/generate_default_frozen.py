"""Freeze the output of a gpumd binary for the default-path tests (DR-004 AC-2; ADR-0005 item 5).

Writes default_frozen.json next to this file. Two generations exist:
  1. `pre-change` (develop af269060, double accumulation): now kept as default_frozen_pre_r3.json
     and used by the tolerance comparison of the R3 change (not regenerated).
  2. `r3` (R3 binary, compensated float accumulation): default_frozen.json, regenerated ONCE on
     2026-11-06 (ADR-0005 item 5: the accumulation precision changes the last digits); not a precedent.

    python generate_default_frozen.py <path/to/gpumd> <source-description> <dftd2.cu-sha256> [pre-change|r3]

For each structure and each default-form input (`dftd2 0.75`, `dftd2 0.75 12`) it stores the dump.xyz
lines 2.. (header with energy and virial, then species, positions and forces, `precision double`) of a
static run, and the stdout lines mentioning DFT-D2 (the header block). Every run is repeated 3 times
and must be bit-identical (the D2 kernels use no atomics). The potential is the all-zero NEP
`zero_nep_H_C_O_Cu.txt` (zero energy, force and virial exactly), so the output is the D2 contribution
alone and NEP's own run-to-run noise is absent. The `stress="..."` field of the header is removed: with
a zero potential it varies from run to run (also on the same binary).
Never regenerate it to make a failing test pass.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
NEP = HERE / 'zero_nep_H_C_O_Cu.txt'

CASES = {  # name: (structure file, note)
    'water27': ('water27.xyz', 'small box (9.3 A), cubic, periodic'),
    'graphite': ('graphite.xyz', 'small box, hexagonal'),
    'water27_tri': ('water27_tri.xyz', 'small box, triclinic'),
    'benzene_dimer': ('benzene_dimer.xyz', 'non-periodic, large-box path'),
    'tiny24': ('tiny24.xyz', '24 atoms in a 40 A cell, large-box path at rc = 12'),
}
INPUTS = ['dftd2 0.75', 'dftd2 0.75 12']


def run(exe, structure, line):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        read(structure).write(tmp / 'model.xyz', format='extxyz')
        lines = [f'potential {NEP}', line, 'ensemble nve', 'time_step 0',
                 'dump_xyz 1 dump.xyz force precision double', 'run 1']
        (tmp / 'run.in').write_text('\n'.join(lines) + '\n')
        r = subprocess.run([exe], cwd=tmp, capture_output=True, text=True, check=True)
        block = [ln for ln in r.stdout.splitlines() if re.search(r'dft-?d2', ln, re.I)]
        dump = (tmp / 'dump.xyz').read_text().splitlines()[1:]
        dump[0] = re.sub(r' stress="[^"]*"', '', dump[0])
        return dump, block


if __name__ == '__main__':
    exe, source, sha = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else 'pre-change'
    assert label in ('pre-change', 'r3')
    binary = {'pre-change': 'pre-change gpumd (dftd2 without table/trunc options)',
              'r3': 'R3 gpumd (compensated float accumulation, ADR-0005); regenerated once, 2026-11-06'}[label]
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    out = {'_provenance': {'binary': binary,
                           'source': source, 'dftd2_cu_sha256': sha, 'gpu_and_driver': gpu,
                           'repeats': 3, 'note': 'bitwise reference; see generate_default_frozen.py'}}
    for name, (fname, note) in CASES.items():
        entry = {'structure': fname, 'note': note, 'inputs': {}}
        for line in INPUTS:
            runs = [run(exe, HERE / fname, line) for _ in range(3)]
            assert all(r == runs[0] for r in runs), f'{name} / {line} is not bit-reproducible'
            entry['inputs'][line] = {'D2': runs[0][0], 'header': runs[0][1]}
        out[name] = entry
        print(name, 'ok (3 identical repeats x 2 inputs)')
    json.dump(out, open(HERE / 'default_frozen.json', 'w'), indent=0)
