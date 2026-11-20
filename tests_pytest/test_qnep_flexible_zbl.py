"""Flexible ZBL (`zbl 0 0` + `zbl_parameters`) for the charge models qnep1/2/3 in gpumd (DR-007 AC-2).

Models are built from the checked-in qNEP fixtures (BaTiO3 qnep1/qnep2, water qnep2) by text edits, so
the test does not depend on specs/.  gpumd static runs (time_step 0, kspace ewald, dump_xyz precision
double) are used throughout.  The ZBL contribution of a model is obtained as (model with ZBL) minus
(the same model with the `zbl` line removed): the NEP/charge/Ewald part cancels.

  a. equivalence: flexible parameters equal to the universal ones reproduce the universal-ZBL energy,
     forces, and virial, with and without charge_transfer, qnep1/qnep2/qnep3, 2 and 3 types;
  b. independent float64 numpy dimer reference of the ZBL energy and force for pair-specific
     parameters (all six pairs of three types, inside and outside the switching region);
  c. central finite differences of the force and the virial (gates of the charge_transfer tests
     test_qnep_charge_transfer_forces.py: force RMS 1e-3 eV/A, virial RMS 3e-3 |W|_RMS, no loosening);
  d. the large-box and the small-box kernels give the same ZBL contribution;
  e. the restrictions that remain are still errors.
"""
import subprocess

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, MODELS_DIR

pytestmark = pytest.mark.fast

UNI = [0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162]
# Equivalence tolerances.  Observed (RTX A6000, sm_86, 3 repeats each of the universal and the flexible
# model, 13 cases): flexible - universal is bit-identical in energy, force, and virial (force differences
# <= 3e-14 eV/A appear only as run-to-run double-atomic-add noise in the charge_transfer cases, the same
# size as the same-binary repeat difference).  The tolerances are ~30x that noise, ~12 orders below the
# ZBL contribution (186 eV, 114 eV/A for the compressed BaTiO3 bulk) and far below the DR-001 AC-1
# caps (E, W relative 2e-6, F 1e-5 eV/A).
EQ_E_REL = 1e-12
EQ_F_ABS = 1e-12
EQ_W_REL = 1e-12
FORCE_GATE = 1e-3  # eV/A (VER-0034 gate of the charge_transfer tests)
VIRIAL_REL_GATE = 3e-3
# FD step.  The existing charge_transfer tests use 0.005 A; the steep ZBL forces need 0.0025 A here: the
# error of the central difference is the truncation term (delta^2 / 6) F'': measured on a pure-ZBL
# 10-atom cluster, force RMS error 4.96e-3, 1.08e-3, 2.7e-4, 1.1e-4 eV/A for delta 0.01, 0.005, 0.0025,
# 0.001 (scales as delta^2 down to 0.0025, i.e. the kernel force is the derivative of the energy), while
# the float32 energy noise (same models without ZBL: 3.7e-4 -> 7.1e-4 -> 1.5e-3 for delta 0.005,
# 0.0025, 0.0015) grows as 1/delta.  delta = 0.003 balances both.  The gates themselves (1e-3 eV/A,
# 3e-3 |W|) are not changed.
DELTA = 0.003
LARGE_L = 18.5  # > 2.5 * (rc_radial 6 + 1) = 17.5: large-box kernels
SMALL_L = 17.0  # <= 17.5: small-box kernels

# pair-specific parameters for three types (Ba, Ti, O); pair order (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)
PAIR_RC = [(0.50, 1.10), (0.60, 1.30), (0.70, 1.50), (0.80, 1.40), (0.90, 1.60), (0.60, 1.20)]


# finite-difference structure: a = 3.0 A (Ti-O 1.3-1.7 A, Ba-O ~1.9-2.2 A, ...) with windows so that
# every pair type has neighbors inside its switching region; the compressed a = 2.6 A structure of
# the equivalence tests has ZBL forces of ~100 eV/A, too large for a float32 energy FD at 1e-3 eV/A
FD_A = 3.0
FD_CELLS = (2, 1, 1)  # 10 atoms: total energy ~ 100 eV, so float noise of the energy FD stays small
FD_CAPACITY = (500, 200)
FD_PAIR_RC = [(1.5, 3.1), (1.2, 2.55), (1.0, 2.1), (1.5, 3.1), (0.5, 1.4), (1.0, 2.15)]
FD_NEP_SCALE = 0.1  # smooth NEP part (fixture weights give force RMS ~10 eV/A and a noisy energy FD)


def pair_params(rc_list, scale=True):
    out = []
    for k, (a, b) in enumerate(rc_list):
        c = [x * (1.0 + 0.03 * k * (1 if i % 2 == 0 else -1)) if scale else x for i, x in enumerate(UNI)]
        out += [a, b] + c
    return out


def pair_index(t1, t2, nt):
    t1, t2 = min(t1, t2), max(t1, t2)
    return t1 * nt - t1 * (t1 - 1) // 2 + (t2 - t1)


def read_fixture(name):
    return (MODELS_DIR / name).read_text()


def model_text(base, zbl, ct=None, qnep=None, zbl_values=None, capacity=(120, 60)):
    """zbl: 'none' | 'universal' | 'flex'.  The fixture itself carries universal `zbl 0.75 1.5`
    (BaTiO3) or no zbl line (water; 'universal'/'flex' then insert one)."""
    text = base
    nt = int(text.split('\n')[0].split()[1])
    lines = text.split('\n')
    lines = [l for l in lines if not l.startswith('zbl ')]
    first_at = next(i for i, l in enumerate(lines) if l.startswith('@ '))
    header, body = lines[:first_at], lines[first_at:]
    if zbl == 'universal':
        header.append('zbl 0.75 1.5')
    elif zbl == 'flex':
        header.append('zbl 0 0')
    if ct is not None:
        header.append('charge_transfer %g' % ct)
    # the compressed cluster has up to 38 angular neighbors; the fixture capacity (26, enlarged to
    # 33) would overflow the neighbor list (illegal memory access), so use a larger capacity
    header = [('cutoff 6 4 %d %d' % capacity if l == 'cutoff 6 4 86 26' else l) for l in header]
    if qnep is not None:
        header = [('model qnep%d' % qnep) if l.startswith('model ') else l for l in header]
    out = '\n'.join(header + body).rstrip('\n') + '\n'
    if ct is not None:
        rng = np.random.default_rng(5)
        n = nt * (nt + 1) // 2 * 9  # basis_size_radial + 1 = 9
        out += '@ phi_parameters %d\n' % n + ''.join('%.9g\n' % v for v in rng.uniform(0.0, 0.3, n))
    if zbl == 'flex':
        vals = zbl_values
        if vals is None:
            vals = []
            for _ in range(nt * (nt + 1) // 2):
                vals += [0.75, 1.5] + UNI
        assert len(vals) == 10 * nt * (nt + 1) // 2
        out += '@ zbl_parameters %d\n' % len(vals) + ''.join('%.9g\n' % v for v in vals)
    return out


def scale_nep(text, factor):
    """Same model with all ANN and descriptor weights multiplied by `factor` (0 gives a model whose
    NEP energy and forces vanish), except sqrt_epsilon_inf (the first of the two trailing ANN
    globals, kept so that the Ewald part stays finite); b1, the energy bias, is set to zero.
    factor = 0: the total of a run is the ZBL contribution alone, without the float cancellation
    noise of a (with ZBL) - (without ZBL) subtraction.  A small factor gives a smooth model whose
    float32 energy noise is low enough for central finite differences at 1e-3 eV/A (the fixture
    weights give force RMS ~10 eV/A on compressed structures).  Layout of qnep1/2/3:
    nt * (dim + 3) * neurons + 2 ANN values, then the descriptor coefficients."""
    lines = text.split('\n')
    nt = int(lines[0].split()[1])
    neurons = int(next(l for l in lines if l.startswith('ann ')).split()[1])
    iq = next(k for k, l in enumerate(lines) if l.startswith('@ q_scaler '))
    dim = int(lines[iq].split()[2])
    ip = next(k for k, l in enumerate(lines) if l.startswith('@ parameters '))
    n = int(lines[ip].split()[2])
    n_ann = nt * (dim + 3) * neurons + 2
    for k in range(n):
        if k == n_ann - 2:
            continue
        v = 0.0 if k == n_ann - 1 else float(lines[ip + 1 + k]) * factor
        lines[ip + 1 + k] = '%.9g' % v
    return '\n'.join(lines)


def zero_nep(text):
    return scale_nep(text, 0.0)


def batio3_cluster(L=None, a=2.6, seed=1, sigma=0.08, cells=(2, 2, 2)):
    """2x2x2 BaTiO3, a = 2.6 A (short Ti-O, O-O contacts), rattled; periodic bulk (L=None, box 2a)
    or the same atoms as an isolated cluster in a cubic box L."""
    rng = np.random.default_rng(seed)
    basis = [('Ba', 0, 0, 0), ('Ti', .5, .5, .5), ('O', .5, .5, 0), ('O', .5, 0, .5), ('O', 0, .5, .5)]
    sym, pos = [], []
    for i in range(cells[0]):
        for j in range(cells[1]):
            for k in range(cells[2]):
                for s, x, y, z in basis:
                    sym.append(s)
                    pos.append(np.array([x + i, y + j, z + k]) * a + rng.normal(0, sigma, 3))
    pos = np.array(pos)
    ext = a * np.array(cells, float)
    if L is None:
        return sym, pos, np.diag(ext)
    return sym, pos + 0.5 * (L - ext), L * np.eye(3)


def water_structure(seed=3, L=8.0, n=20, min_dist=0.8):
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.random(3) * L
        if all(np.linalg.norm((p - q) - L * np.round((p - q) / L)) > min_dist for q in pos):
            pos.append(p)
    sym = ['O' if i % 2 == 0 else 'H' for i in range(n)]
    return sym, np.array(pos), L * np.eye(3)


def write_xyz(path, sym, pos, lat):
    with open(path, 'w') as f:
        f.write('%d\n' % len(pos))
        f.write('Lattice="%s" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                % ' '.join('%.15g' % x for x in np.asarray(lat).reshape(-1)))
        for s, p in zip(sym, pos):
            f.write('%s %.15f %.15f %.15f\n' % (s, p[0], p[1], p[2]))


def parse_xyz(path):
    L = open(path).read().split('\n')
    n = int(L[0])
    head = L[1]
    energy = float(head.split('energy=')[1].split()[0])
    vir = np.array([float(x) for x in head.split('virial="')[1].split('"')[0].split()]).reshape(3, 3)
    props = head.split('Properties=')[1].split()[0].split(':')
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    rows = [l.split() for l in L[2:2 + n]]
    c0, w = col['forces']
    forces = np.array([[float(x) for x in r[c0:c0 + w]] for r in rows])
    return {'energy': energy, 'virial': vir, 'forces': forces}


_counter = [0]


def run_static(tmp_path, text, sym, pos, lat, kspace='ewald'):
    _counter[0] += 1
    d = tmp_path / ('run%d' % _counter[0])
    d.mkdir()
    (d / 'nep.txt').write_text(text)
    write_xyz(d / 'model.xyz', sym, pos, lat)
    (d / 'run.in').write_text(
        'potential nep.txt\nkspace %s\nensemble nve\ntime_step 0\n'
        'dump_xyz 1 d.xyz force precision double\ndump_thermo 1\nrun 1\n' % kspace)
    r = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('gpumd failed in %s:\n%s' % (d, (r.stdout + r.stderr)[-800:]))
    return parse_xyz(d / 'd.xyz')


def zbl_part(tmp_path, with_text, without_text, sym, pos, lat):
    a = run_static(tmp_path, with_text, sym, pos, lat)
    b = run_static(tmp_path, without_text, sym, pos, lat)
    return {'energy': a['energy'] - b['energy'], 'forces': a['forces'] - b['forces'],
            'virial': a['virial'] - b['virial']}


# ---------------------------------------------------------------------------------------------
# a. equivalence with the universal ZBL
# ---------------------------------------------------------------------------------------------
EQ_CASES = [
    ('qnep1_BaTiO3', 'qnep_mode1_BaTiO3.txt', None, None),
    ('qnep2_BaTiO3', 'qnep_mode2_BaTiO3.txt', None, None),
    ('qnep3_BaTiO3', 'qnep_mode2_BaTiO3.txt', None, 3),
    ('qnep2_BaTiO3_ct', 'qnep_mode2_BaTiO3.txt', 4.0, None),
    ('qnep3_BaTiO3_ct', 'qnep_mode2_BaTiO3.txt', 4.0, 3),
    ('qnep2_water', 'qnep_mode2_water.txt', None, None),
    ('qnep2_water_ct', 'qnep_mode2_water.txt', 4.0, None),
]


def _structure_for(name, small=True):
    if 'water' in name:
        return water_structure()
    return batio3_cluster(None if small else LARGE_L)


@pytest.mark.parametrize('label, fixture, ct, qnep', EQ_CASES, ids=[c[0] for c in EQ_CASES])
@pytest.mark.parametrize('small', [True, False], ids=['small-box', 'large-box'])
def test_flexible_equals_universal(tmp_path, label, fixture, ct, qnep, small):
    if 'water' in label and not small:
        pytest.skip('water case runs in its own (small-box) structure only')
    base = read_fixture(fixture)
    sym, pos, lat = _structure_for(label, small)
    uni = run_static(tmp_path, model_text(base, 'universal', ct, qnep), sym, pos, lat)
    flex = run_static(tmp_path, model_text(base, 'flex', ct, qnep), sym, pos, lat)
    none = run_static(tmp_path, model_text(base, 'none', ct, qnep), sym, pos, lat)
    contrib = uni['energy'] - none['energy']
    # the ZBL contribution must be far above the tolerance, otherwise the comparison is void
    assert abs(contrib) > 1.0, (label, contrib)
    assert np.max(np.abs(uni['forces'] - none['forces'])) > 1.0
    assert abs(flex['energy'] - uni['energy']) <= EQ_E_REL * abs(uni['energy']), (
        label, flex['energy'], uni['energy'])
    assert np.max(np.abs(flex['forces'] - uni['forces'])) <= EQ_F_ABS
    wscale = np.max(np.abs(uni['virial']))
    assert np.max(np.abs(flex['virial'] - uni['virial'])) <= EQ_W_REL * wscale


# ---------------------------------------------------------------------------------------------
# b. independent float64 dimer reference
# ---------------------------------------------------------------------------------------------
Z = {'Ba': 56, 'Ti': 22, 'O': 8}
SYMS = ['Ba', 'Ti', 'O']
K_C = 14.399645  # eV A, 1/(4 pi eps0)


A_INV_CODE = float(np.float32(2.134563))  # literal in nep_utilities.cuh (1 / 0.46848 = 2.13456284...)


def zbl_reference(zi, zj, d, params, exact_a=False):
    """ZBL energy (eV) and dE/dr (eV/A) of one pair at distance d in float64:
    E = K_C zi zj phi(r/a) / r * fc(r), a = 0.46848 / (zi^0.23 + zj^0.23), phi = sum_k c_k exp(-e_k x),
    fc = 1 below inner, (1 + cos(pi (r - inner)/(outer - inner)))/2 between, 0 above outer."""
    inner, outer = params[0], params[1]
    coef, expo = np.array(params[2:10:2], float), np.array(params[3:10:2], float)
    # exact_a: a = 0.46848 / (zi^0.23 + zj^0.23); otherwise 1/a uses the float literal of the kernels
    # (relative difference 7.4e-8 in a, which enters the exponents e_k r / a and reaches 3e-6 in the
    # energy of heavy pairs near the outer cutoff; the same constant is in ordinary NEP and v5.9.1)
    s = zi ** 0.23 + zj ** 0.23
    a_inv = s / 0.46848 if exact_a else s * A_INV_CODE
    x = d * a_inv
    phi = np.sum(coef * np.exp(-expo * x))
    dphi = np.sum(-expo * coef * np.exp(-expo * x)) * a_inv
    pref = K_C * zi * zj
    f0 = pref * phi / d
    f0p = pref * (dphi / d - phi / d ** 2)
    if d < inner:
        fc, fcp = 1.0, 0.0
    elif d < outer:
        w = np.pi / (outer - inner)
        fc = 0.5 * np.cos(w * (d - inner)) + 0.5
        fcp = -0.5 * w * np.sin(w * (d - inner))
    else:
        fc, fcp = 0.0, 0.0
    return f0 * fc, f0p * fc + f0 * fcp


DIMER_PAIRS = [(t1, t2) for t1 in range(3) for t2 in range(t1, 3)]


@pytest.mark.parametrize('ct', [None, 4.0], ids=['no-ct', 'ct'])
@pytest.mark.parametrize('qnep', [2, 3])
def test_dimer_reference(tmp_path, qnep, ct):
    base = read_fixture('qnep_mode2_BaTiO3.txt')
    vals = pair_params(PAIR_RC)
    flex = zero_nep(model_text(base, 'flex', ct, qnep, vals))
    worst_e = worst_f = worst_ex = 0.0
    for (t1, t2) in DIMER_PAIRS:
        p = pair_index(t1, t2, 3)
        inner, outer = PAIR_RC[p]
        # below the inner cutoff, in the switching region, near the outer cutoff.  The last point is
        # ill-conditioned for a float32 kernel: fc -> 0 there, d ln E / dr = -37 / A for the Ba-Ba
        # window, so one float rounding of d12 (6e-8 A) gives 2e-6 relative; it uses the conditioned
        # bound tol = 1e-6 + 1.2e-7 |dE/dr| / |E| (A) instead of the strict AC-2 value
        for d, strict in ((0.6 * inner, True), (inner + 0.25 * (outer - inner), True),
                          (0.5 * (inner + outer), True), (outer - 0.1 * (outer - inner), False)):
            L = 30.0
            pos = np.array([[10.0, 12.0, 11.0], [10.0 + d * 0.6, 12.0 + d * 0.0, 11.0 + d * 0.8]])
            sym = [SYMS[t1], SYMS[t2]]
            r = run_static(tmp_path, flex, sym, pos, L * np.eye(3))  # NEP part is zero
            e_ref, de_ref = zbl_reference(Z[sym[0]], Z[sym[1]], d, vals[10 * p:10 * p + 10])
            e_ex, de_ex = zbl_reference(Z[sym[0]], Z[sym[1]], d, vals[10 * p:10 * p + 10], exact_a=True)
            # force on atom 2 along (0.6, 0, 0.8): F2 = -dE/dr * unit
            u = np.array([0.6, 0.0, 0.8])
            f2_ref = -de_ref * u
            assert abs(e_ref) > 1e-3, (t1, t2, d, e_ref)
            err_e = abs(r['energy'] - e_ref) / abs(e_ref)
            err_f = np.linalg.norm(r['forces'][1] - f2_ref) / np.linalg.norm(f2_ref)
            if strict:
                worst_e, worst_f = max(worst_e, err_e), max(worst_f, err_f)
            cond = abs(de_ref / e_ref)
            tol_e = 1e-6 if strict else 1e-6 + 1.2e-7 * cond
            tol_f = 1e-5 if strict else 1e-5 + 1.2e-7 * cond
            assert err_e <= tol_e, ('energy', t1, t2, d, r['energy'], e_ref, err_e, tol_e)
            assert err_f <= tol_f, ('force', t1, t2, d, r['forces'][1], f2_ref, err_f, tol_f)
            assert np.max(np.abs(r['forces'][0] + r['forces'][1])) <= 1e-5 * np.linalg.norm(f2_ref)
            # the exact a = 0.46848/(...) formula: differs only through the kernel constant
            dev_e = abs(r['energy'] - e_ex) / abs(e_ex)
            dev_f = np.linalg.norm(r['forces'][1] + de_ex * u) / abs(de_ex)
            worst_ex = max(worst_ex, dev_e, dev_f)
            assert dev_e <= 1e-5 + 1.2e-7 * cond and dev_f <= 1e-5 + 1.2e-7 * cond, ('exact-a', t1, t2, d, dev_e, dev_f)
    print('dimer worst relative error (strict points): energy %.2e force %.2e; exact-a formula %.2e' % (worst_e, worst_f, worst_ex))


# ---------------------------------------------------------------------------------------------
# c. finite differences
# ---------------------------------------------------------------------------------------------
def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def force_fd(tmp_path, text, sym, pos, lat, atoms):
    base = run_static(tmp_path, text, sym, pos, lat)
    F_fd = np.zeros((len(atoms), 3))
    for i, k in enumerate(atoms):
        for a in range(3):
            e = []
            for s in (+1, -1):
                p2 = pos.copy()
                p2[k, a] += s * DELTA
                e.append(run_static(tmp_path, text, sym, p2, lat)['energy'])
            F_fd[i, a] = -(e[0] - e[1]) / (2 * DELTA)
    err = base['forces'][atoms] - F_fd
    return base, {'rms_F': _rms(F_fd), 'rms_err': _rms(err), 'max_err': float(np.max(np.abs(err)))}


def virial_fd(tmp_path, text, sym, pos, lat, base):
    err, ref = [], []
    for a, b in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        e = []
        for s in (+1, -1):
            Fm = np.eye(3)
            Fm[a, b] += s * DELTA
            e.append(run_static(tmp_path, text, sym, pos @ Fm.T, lat @ Fm.T)['energy'])
        W_fd = -(e[0] - e[1]) / (2 * DELTA)
        W_gpu = 0.5 * (base['virial'][a, b] + base['virial'][b, a])
        err.append(W_gpu - W_fd)
        ref.append(W_fd)
    return {'rms_W': _rms(ref), 'rms_err': _rms(err)}


FD_CASES = [
    ('qnep2 small-box', None, None, True, [0, 1, 3, 6, 9]),
    ('qnep2 large-box', None, None, False, [0, 1, 3, 6, 9]),
    ('qnep3 ct small-box', 4.0, 3, True, [0, 1, 3, 6, 9]),
    ('qnep2 ct large-box', 4.0, None, False, [0, 1, 3, 6, 9]),
]


@pytest.mark.parametrize('label, ct, qnep, small, atoms', FD_CASES, ids=[c[0] for c in FD_CASES])
def test_force_and_virial_finite_difference(tmp_path, label, ct, qnep, small, atoms):
    base_text = read_fixture('qnep_mode2_BaTiO3.txt')
    text = scale_nep(
        model_text(base_text, 'flex', ct, qnep, pair_params(FD_PAIR_RC), FD_CAPACITY), FD_NEP_SCALE)
    sym, pos, lat = batio3_cluster(None if small else LARGE_L, a=FD_A, cells=FD_CELLS)
    base, fres = force_fd(tmp_path, text, sym, pos, lat, atoms)
    assert fres['rms_F'] > 0.1, (label, fres)
    assert fres['rms_err'] <= FORCE_GATE, (label, fres)
    assert fres['max_err'] <= 3 * FORCE_GATE, (label, fres)
    vres = virial_fd(tmp_path, text, sym, pos, lat, base)
    print(label, 'force FD', fres, 'virial FD', vres)
    assert vres['rms_W'] > 1.0, (label, vres)
    assert vres['rms_err'] <= VIRIAL_REL_GATE * vres['rms_W'], (label, vres)


# ---------------------------------------------------------------------------------------------
# d. large-box and small-box kernels
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize('ct', [None, 4.0], ids=['no-ct', 'ct'])
def test_large_box_equals_small_box(tmp_path, ct):
    base = read_fixture('qnep_mode2_BaTiO3.txt')
    vals = pair_params(PAIR_RC)
    flex = zero_nep(model_text(base, 'flex', ct, None, vals))
    none = zero_nep(model_text(base, 'none', ct, None))
    res = []
    for L in (SMALL_L, LARGE_L):
        sym, pos, lat = batio3_cluster(L)  # isolated cluster: no interaction across the images
        res.append(zbl_part(tmp_path, flex, none, sym, pos, lat))
    s, l = res
    assert abs(s['energy']) > 1.0
    assert abs(s['energy'] - l['energy']) <= 1e-6 * abs(s['energy'])
    # observed (RTX A6000): E 2.3e-7 relative, F 3.1e-4 eV/A = 1.8e-6 of max |F| = 171 eV/A, W 2.1e-7;
    # repeats of one kernel are bit-identical, so this is float summation-order noise
    assert np.max(np.abs(s['forces'] - l['forces'])) <= 1e-5 * np.max(np.abs(s['forces']))
    assert np.max(np.abs(s['virial'] - l['virial'])) <= 1e-6 * np.max(np.abs(s['virial']))


# ---------------------------------------------------------------------------------------------
# e. restrictions that remain
# ---------------------------------------------------------------------------------------------
def _expect_error(tmp_path, text, fragment):
    sym, pos, lat = batio3_cluster()
    with pytest.raises(RuntimeError) as exc:
        run_static(tmp_path, text, sym, pos, lat)
    assert fragment in str(exc.value), str(exc.value)


def test_flexible_zbl_with_typewise_factor_is_rejected(tmp_path):
    base = read_fixture('qnep_mode2_BaTiO3.txt')
    text = model_text(base, 'flex', None, None, pair_params(PAIR_RC))
    text = text.replace('\nzbl 0 0\n', '\nzbl 0 0 0.65\n')
    _expect_error(tmp_path, text, 'typewise')


def test_more_than_ten_active_elements_is_rejected(tmp_path):
    # 11 types, all present in the structure (an inactive type would be dropped by the compact
    # path): 11*12/2*10 = 660 > 550 values.  Layout: (dim + 3) * neurons per type + 2 globals, and
    # nt^2 * (9 * 9 + 7 * 9) descriptor coefficients (fixture: n_max 8 6, basis 8 8, l_max 4).
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na']
    nt = len(elements)
    dim = 37
    neurons = 40
    n_par = (dim + 3) * neurons * nt + 2 + nt * nt * (9 * 9 + 7 * 9)
    rng = np.random.default_rng(0)
    lines = ['nep %d %s' % (nt, ' '.join(elements)), 'format_version 1', 'model qnep2', 'zbl 0 0',
             'cutoff 6 4 86 26', 'n_max 8 6', 'basis_size 8 8', 'l_max 4 0 0', 'ann 40 0',
             'energy_reference ' + ' '.join(['0'] * nt)]
    text = '\n'.join(lines) + '\n@ parameters %d\n' % n_par
    text += ''.join('%.6g\n' % v for v in rng.normal(0, 0.1, n_par))
    text += '@ q_scaler %d\n' % dim + ''.join('1.0\n' for _ in range(dim))
    vals = []
    for _ in range(nt * (nt + 1) // 2):
        vals += [0.75, 1.5] + UNI
    text += '@ zbl_parameters %d\n' % len(vals) + ''.join('%.9g\n' % v for v in vals)
    sym = elements
    pos = np.array([[2.0 * (i % 4) + 1.0, 2.0 * (i // 4) + 1.0, 1.0] for i in range(nt)])
    _counter[0] += 1
    d = tmp_path / ('run%d' % _counter[0])
    d.mkdir()
    (d / 'nep.txt').write_text(text)
    write_xyz(d / 'model.xyz', sym, pos, 20.0 * np.eye(3))
    (d / 'run.in').write_text('potential nep.txt\nkspace ewald\nensemble nve\ntime_step 0\nrun 1\n')
    r = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=d, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode != 0, out[-500:]
    assert 'at most 10 active elements' in out, out[-500:]
