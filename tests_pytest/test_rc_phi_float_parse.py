"""rc_phi float parsing: strtof versus double-then-float (CPU only, no gpumd launch).

DR-005 (charge_transfer). The model reader (src/utilities/nep_io/nep_model_file_charge.cu,
`read_float` in nep_model_file.cu) parses the token with NEP_File::parse_double (istringstream
>> double, i.e. a correctly rounded strtod) and then does static_cast<float>. The pre-merge fork
(commit 5de76eb5, nep_charge.cu) used get_double_from_token (strtod) and a float member, i.e. the
same double-then-float path. So the merge did not change the rc_phi value. This file pins the
numerical premise: for the rc_phi values used by the fork's tests, docs and generators, a direct
strtof and float(double) give bitwise the same float, hence it does not matter which of the two a
reader uses.

Caveat (double rounding): float(double(s)) can differ from strtof(s) by one float ulp when the
decimal string lies within half a double-ulp of a float rounding midpoint (probability about
2**-29 for an arbitrary decimal; 0 of 200000 random strings measured on 2026-11-18). Short
decimals such as 3.5 or 4.0 are exact or far from any midpoint. The last test constructs such a
string to show that the comparison below is sensitive.
"""
import ctypes
import ctypes.util
from decimal import Decimal

import numpy as np
import pytest

# rc_phi tokens used in tests_pytest, specs/verification/**/scripts and doc/ (3, 3.0, 3.5, 4, 4.0,
# 4.5, 6 literally; the rest are the neighbouring values a user would write in a nep.txt)
RC_PHI_VALUES = ['3', '3.0', '3.5', '4', '4.0', '4.5', '5', '5.0', '5.5', '6', '6.0', '7', '8', '2.5']


def _libc():
    lib = ctypes.CDLL(ctypes.util.find_library('c') or None)
    lib.strtof.restype = ctypes.c_float
    lib.strtof.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    return lib


def _strtof(lib, text):
    return np.float32(lib.strtof(text.encode(), None))


def _double_then_float(text):
    return np.float32(float(text))


@pytest.mark.parametrize('text', RC_PHI_VALUES)
def test_rc_phi_strtof_equals_double_then_float(text):
    a = _strtof(_libc(), text)
    b = _double_then_float(text)
    assert a.tobytes() == b.tobytes(), (text, a, b)
    # the value is also what the '%g' written by the generators gives
    assert _double_then_float('%g' % float(text)).tobytes() == b.tobytes()


def test_double_rounding_example_is_detected():
    # float 1.0 has an even mantissa; the midpoint to the next float is 1 + 2**-24. A decimal just
    # above it (by less than half a double ulp, 2**-53) rounds to the midpoint as a double, and
    # ties-to-even then gives 1.0 as a float, while strtof sees it above the midpoint.
    text = format(Decimal(1) + Decimal(2) ** -24 + Decimal(2) ** -54, '.40f')
    a = _strtof(_libc(), text)
    b = _double_then_float(text)
    assert a.tobytes() != b.tobytes(), 'this host does not show the double-rounding case'
    assert b == np.float32(1.0) and a == np.nextafter(np.float32(1.0), np.float32(2.0))
