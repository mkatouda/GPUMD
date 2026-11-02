"""The built-in DFT-D2 table in src/force/dftd2para.cuh is generated from the CSV (DR-002 AC-2).

CPU only.  The header must be byte-identical to the output of
tools/Miscellaneous/dftd2/generate_dftd2para.py and its values must equal the CSV
(C6 in eV A^6 verbatim, R0 = 1.1 * r0), for all 86 elements.
"""
import re
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

HEADER = REPO_ROOT / 'src' / 'force' / 'dftd2para.cuh'
CSV = REPO_ROOT / 'tests_pytest' / 'fixtures' / 'dftd2' / 'dftd2_params.csv'
GENERATOR = REPO_ROOT / 'tools' / 'Miscellaneous' / 'dftd2' / 'generate_dftd2para.py'


def _csv_rows():
    rows = [ln.split(',') for ln in CSV.read_text().splitlines() if ln and not ln.startswith('#')]
    return rows[1:]


def test_header_is_exactly_the_generator_output(tmp_path):
    out = tmp_path / 'dftd2para.cuh'
    subprocess.run([sys.executable, str(GENERATOR), str(CSV), str(out)], check=True)
    assert out.read_text() == HEADER.read_text()


@pytest.mark.parametrize('array, column', [('DFTD2_C6_EV_A6', 6), ('DFTD2_R0_A', 3)])
def test_header_values_equal_csv(array, column):
    text = HEADER.read_text()
    body = re.search(array + r'\[DFTD2_MAX_Z \+ 1\] = \{(.*?)\};', text, re.S).group(1)
    values = [float(v) for v in re.findall(r'^\s*([-+0-9.eE]+),', body, re.M)]
    assert len(values) == 87 and values[0] == 0.0
    rows = _csv_rows()
    assert len(rows) == 86
    for row in rows:
        z = int(row[0])
        expected = float(row[column]) * (1.1 if array == 'DFTD2_R0_A' else 1.0)
        assert values[z] == expected, f'Z={z}'
