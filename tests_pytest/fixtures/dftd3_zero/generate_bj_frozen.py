"""Freeze the output of the PRE-CHANGE binary for the BJ-compatibility test (DR-003 AC-3).

Writes bj_frozen.json (and tiny24.xyz) next to this file. Run it ONCE, with a gpumd binary built
from the unmodified develop HEAD (before any zero-damping change; the commit is recorded):

    python generate_bj_frozen.py <path/to/pre-change/gpumd> <description-of-source> <dftd3.cu-sha256>

For each case it stores the dump.xyz lines 2.. (header with energy and virial, then species,
positions and forces, `precision double`) of a static run with `dftd3 pbe rc rcn`, and
the 'Add DFT-D3:' block of the standard output. Each run is repeated 3 times and must be
bit-identical. The potential is the all-zero NEP of make_zero_nep.py (zero energy, force and
virial exactly), so the output is the D3 contribution alone: NEP89 itself is not bit-reproducible
run to run on this GPU, and the large-box cases are deterministic only with one warp (tiny24,
benzene_dimer). The comparison in the test is bitwise. Never regenerate it with a changed binary.

The `stress="..."` field of the dump header is removed: with a zero potential it varies from run
to run (also on the same binary) while energy, virial and forces do not.

Which binary: the dftd3 kernels of the develop HEAD (3b2c8db8) use atomicAdd, so even that binary
is not bit-reproducible run to run (virial and forces differ in the last digits). The reference
is therefore taken from the binary of the DR-001 dftd3 code (src/force/dftd3.cu with the
fixed-order reductions; the sha256 of dftd3.cu is recorded), where energy, virial and forces are
bit-reproducible.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).resolve().parent
BJ_DIR = HERE.parent / 'dftd3'
NEP = HERE / 'zero_nep_H_C_O_Cu.txt'     # all-zero NEP (make_zero_nep.py): the output is D3 alone


def tiny24():
    atoms = read(BJ_DIR / 'water27.xyz')[:24]
    atoms.set_cell([40.0, 40.0, 40.0])
    atoms.set_pbc(True)
    atoms.center()
    write(HERE / 'tiny24.xyz', atoms, format='extxyz')


CASES = {  # name: (structure, (rc, rc_cn), note)
    'water27': ('water27.xyz', (12, 6), 'small-box, cubic'),
    'graphite': ('graphite.xyz', (12, 6), 'small-box, hexagonal'),
    'water27_tri': ('water27_tri.xyz', (12, 6), 'small-box, triclinic'),
    'benzene_dimer': ('benzene_dimer.xyz', (12, 6), 'large-box path, non-periodic'),
    'tiny24': ('tiny24.xyz', (12, 6), 'large-box path, 24 atoms in a 40 A cell'),
}


def run(exe, structure, d3_line):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        read(structure).write(tmp / 'model.xyz', format='extxyz')
        lines = [f'potential {NEP}'] + ([d3_line] if d3_line else []) + \
            ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
        (tmp / 'run.in').write_text('\n'.join(lines) + '\n')
        r = subprocess.run([exe], cwd=tmp, capture_output=True, text=True, check=True)
        out = r.stdout.splitlines()
        block = []
        if d3_line:
            i = out.index('    Add DFT-D3:')
            while i < len(out) and out[i].strip():
                block.append(out[i])
                i += 1
        lines = (tmp / 'dump.xyz').read_text().splitlines()[1:]
        lines[0] = re.sub(r' stress="[^"]*"', '', lines[0])   # varies run to run (see docstring)
        return lines, block


if __name__ == '__main__':
    exe, commit, sha = sys.argv[1], sys.argv[2], sys.argv[3]
    tiny24()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    out = {'_provenance': {'binary': 'pre-change gpumd (DR-001 dftd3 code, no zero damping)',
                           'source': commit, 'dftd3_cu_sha256': sha, 'gpu_and_driver': gpu,
                           'note': 'bitwise reference; see generate_bj_frozen.py'}}
    for name, (fname, cut, note) in CASES.items():
        structure = HERE / fname if (HERE / fname).exists() else BJ_DIR / fname
        entry = {'structure': fname, 'cutoffs': list(cut), 'note': note}
        runs = [run(exe, structure, f'dftd3 pbe {cut[0]} {cut[1]}') for _ in range(3)]
        assert all(r == runs[0] for r in runs), f'{name} is not bit-reproducible'
        entry['D3'], entry['stdout_block'] = runs[0]
        out[name] = entry
        print(name, 'ok')
    json.dump(out, open(HERE / 'bj_frozen.json', 'w'), indent=0)
