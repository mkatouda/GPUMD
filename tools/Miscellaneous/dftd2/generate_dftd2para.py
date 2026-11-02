"""Generate src/force/dftd2para.cuh (built-in DFT-D2 C6/R0 table, Z = 1..86) from the CSV.

Input : tests_pytest/fixtures/dftd2/dftd2_params.csv  (itself generated from dftd3-lib by
        tests_pytest/fixtures/dftd2/generate_params.py; see DR-002 D5 and VER-0003)
Output: src/force/dftd2para.cuh

Usage (from the repository root):
    python3 tools/Miscellaneous/dftd2/generate_dftd2para.py

Nothing is typed by hand.  The values written are
    c6 [eV A^6]  = column c6_eV_A6 of the CSV, verbatim
    R0 [A]       = 1.1 * r0_A (column r0_A of the CSV), as double (published R0 = 1.1 * r0)
The sha256 of the CSV is recorded in the header of the generated file.
"""
import hashlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'tests_pytest/fixtures/dftd2/dftd2_params.csv'
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'src/force/dftd2para.cuh'
raw = csv_path.read_bytes()
text_lines = raw.decode().splitlines()
lines = [ln for ln in text_lines if ln and not ln.startswith('#')]
header = lines[0].split(',')
col = {name: header.index(name) for name in header}
c6, r0 = {}, {}
for row in lines[1:]:
    f = row.split(',')
    z = int(f[col['Z']])
    c6[z] = f[col['c6_eV_A6']]
    r0[z] = repr(1.1 * float(f[col['r0_A']]))
assert sorted(c6) == list(range(1, 87))

source_line = [ln for ln in text_lines if ln.startswith('# source:')][0][2:]
digest = hashlib.sha256(raw).hexdigest()

HEAD = '''/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

// GENERATED FILE - do not edit by hand.
// Generator : tools/Miscellaneous/dftd2/generate_dftd2para.py
// Input     : tests_pytest/fixtures/dftd2/dftd2_params.csv, sha256 {digest}
// {source_line}
// Reference : S. Grimme, J. Comput. Chem. 27, 1787 (2006).
// Units     : C6 in eV A^6, R0 (= 1.1 * r0 of the source table) in Angstrom. Index = Z (0 unused).
// Z = 55-86 are single-source group fills (e.g. Ce-Lu share one value).

#pragma once

#define DFTD2_MAX_Z 86

'''

with open(out_path, 'w') as fh:
    fh.write(HEAD.format(digest=digest, source_line=source_line))
    for name, d in (('DFTD2_C6_EV_A6', c6), ('DFTD2_R0_A', r0)):
        fh.write('static const double %s[DFTD2_MAX_Z + 1] = {\n  0.0,\n' % name)
        for z in range(1, 87):
            fh.write('  %s, // Z = %d\n' % (d[z], z))
        fh.write('};\n\n')
