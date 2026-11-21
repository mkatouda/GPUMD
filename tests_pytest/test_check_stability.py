"""DR-008: run-time stability guard `check_stability`.

AC-2: for each criterion a constructed case violates the threshold at a known step. The oracle is
an independent run of the same input without the keyword: a double-precision dump at the check
interval gives the temperature, |U - U0|/N, shortest distance and largest force component at every
evaluation step; the threshold is put half-way between the 2nd and the 3rd evaluation, so the
guard must stop at the 3rd evaluation (within one interval), with 3 frames (default keep_frames) in
stability_frames.xyz whose last frame carries the violating value.
AC-3: thresholds above the observed values leave thermo and dump bit-identical to the run without
the keyword (the guard only reads state).
AC-4: a collapse that the thermostat has already cooled is seen by max_potential_change, not by
max_temperature evaluated only at the end.
AC-6: refusals.
Charges (max_charge) use the qNEP water fixture.
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, MODELS_DIR

STRUCTURES = Path(__file__).parent / 'fixtures' / 'structures'
LJ_TEXT = 'lj 1 C\n0.5 3.4 8.5\n'
INTERVAL = 10
K_B = 8.617343e-5


def lj_pair_xyz(r):
    # zero velocities are given explicitly (without them the initial velocities are random)
    return ('2\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3:vel:R:3 pbc="T T T"\n'
            'C 10 10 10 0 0 0\nC %.10f 10 10 0 0 0\n' % (10.0 + r))


def parse_extxyz(path):
    """Frames of an extended xyz file: list of dicts with step, lattice and the columns."""
    lines = Path(path).read_text().splitlines()
    frames = []
    n = 0
    while n < len(lines):
        count = int(lines[n])
        header = lines[n + 1]
        props = re.search(r'Properties=(\S+)', header).group(1).split(':')
        names = []
        for k in range(0, len(props), 3):
            names += [(props[k], int(props[k + 2]))]
        rows = [l.split() for l in lines[n + 2:n + 2 + count]]
        frame = {'header': header, 'n': count}
        col = 0
        for name, width in names:
            if name == 'species':
                frame[name] = [r[col] for r in rows]
            else:
                frame[name] = np.array([[float(x) for x in r[col:col + width]] for r in rows])
            col += width
        m = re.search(r'step=(\d+)', header)
        frame['step'] = int(m.group(1)) if m else None
        lat = re.search(r'Lattice="([^"]+)"', header).group(1).split()
        frame['lattice'] = np.array([float(x) for x in lat]).reshape(3, 3)
        frames.append(frame)
        n += 2 + count
    return frames


def min_distance(frame):
    """Brute-force minimum-image shortest distance (independent of the guard kernel)."""
    pos = frame['pos']
    cell = frame['lattice']  # rows are lattice vectors
    inv = np.linalg.inv(cell)
    best = np.inf
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            s = (pos[j] - pos[i]) @ inv
            s -= np.round(s)
            best = min(best, np.linalg.norm(s @ cell))
    return best


def run_gpumd(workdir, run_in, files):
    workdir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (workdir / name).write_text(text)
    (workdir / 'run.in').write_text('\n'.join(run_in) + '\n')
    result = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=workdir, capture_output=True, text=True)
    return result


def lj_case(workdir, guard=None, r0=5.2, steps=100, ensemble='ensemble nve', eps=0.5,
            interval=INTERVAL, thermo_every=INTERVAL, dump=True):
    lines = ['potential lj.txt', ensemble, 'time_step 1', 'dump_thermo %d' % thermo_every]
    if dump:
        lines += ['dump_xyz %d dump.xyz velocity force potential precision double' % interval]
    if guard:
        lines += [guard]
    lines += ['run %d' % steps]
    return run_gpumd(workdir, lines, {'model.xyz': lj_pair_xyz(r0),
                                      'lj.txt': LJ_TEXT.replace('0.5', str(eps))})


def text_of(result):
    return result.stdout + result.stderr


def oracle_series(tmp_path):
    """Per-evaluation temperature, |U-U0|/N, shortest distance, max |F| of the unguarded pair."""
    result = lj_case(tmp_path / 'oracle')
    assert result.returncode == 0, text_of(result)[-1500:]
    d = tmp_path / 'oracle'
    frames = parse_extxyz(d / 'dump.xyz')
    thermo = np.loadtxt(d / 'thermo.out')[:, 0:1]
    series = {'temperature': [], 'potential_change': [], 'distance': [], 'force': []}
    u0 = None
    for k, f in enumerate(frames):
        u = f['energy_atom'].sum()
        u0 = u if u0 is None else u0
        series['temperature'].append(thermo[k, 0])
        series['potential_change'].append(abs(u - u0) / f['n'])
        series['distance'].append(min_distance(f))
        series['force'].append(np.abs(f['forces']).max())
    return {k: np.array(v) for k, v in series.items()}


@pytest.fixture(scope='module')
def series(tmp_path_factory):
    s = oracle_series(tmp_path_factory.mktemp('oracle'))
    assert s['temperature'][0] > 0.0
    for key in ('temperature', 'potential_change', 'force'):
        assert np.all(np.diff(s[key][:4]) > 0.0), (key, s[key][:4])
    assert np.all(np.diff(s['distance'][:4]) < 0.0), s['distance'][:4]
    return s


CRITERIA = {
    'temperature': ('max_temperature', 'temperature', 'above'),
    'potential_change': ('max_potential_change', 'potential_change', 'above'),
    'distance': ('min_distance', 'distance', 'below'),
    'force': ('max_force', 'force', 'above'),
}


@pytest.mark.parametrize('name', list(CRITERIA))
def test_criterion_stops_at_third_evaluation(tmp_path, series, name):
    keyword, key, _ = CRITERIA[name]
    values = series[key]
    threshold = 0.5 * (values[1] + values[2])
    result = lj_case(tmp_path, guard='check_stability %d %s %.12g' % (INTERVAL, keyword, threshold))
    out = text_of(result)
    assert result.returncode != 0, out[-1500:]
    # stops within one interval of the first violation: at the 3rd evaluation, step 30
    assert 'step %d:' % (3 * INTERVAL) in out and keyword in out, out[-1500:]
    assert ('step %d:' % (4 * INTERVAL)) not in out
    # the observed value and the threshold are in the message
    assert ('%.10g' % threshold) in out
    frames = parse_extxyz(tmp_path / 'stability_frames.xyz')
    assert [f['step'] for f in frames] == [INTERVAL, 2 * INTERVAL, 3 * INTERVAL]
    last = frames[-1]
    ref = parse_extxyz(tmp_path / 'dump.xyz')[2]
    # the violating frame equals the unguarded trajectory at the same step
    assert np.allclose(last['pos'], ref['pos'], atol=1e-9)
    assert np.allclose(last['vel'], ref['vel'], atol=1e-9, rtol=1e-9)
    assert np.allclose(last['forces'], ref['forces'], atol=1e-9, rtol=1e-9)
    if name == 'distance':
        assert min_distance(last) < threshold <= min_distance(frames[1])
        assert 'atoms 0 and 1' in out
    if name == 'force':
        assert np.abs(last['forces']).max() > threshold >= np.abs(frames[1]['forces']).max()
        assert 'atom 0 ' in out or 'atom 1 ' in out
    if name == 'temperature':
        m = re.search(r'temperature ([0-9.eE+-]+) K exceeds', out)
        # consistency with the temperature of thermo.out (same step)
        assert float(m.group(1)) == pytest.approx(series['temperature'][2], rel=1e-6)
    if name == 'potential_change':
        m = re.search(r'= ([0-9.eE+-]+) eV/atom exceeds', out)
        assert float(m.group(1)) == pytest.approx(series['potential_change'][2], rel=1e-9)
    # no restart file
    assert not (tmp_path / 'restart.xyz').exists()


def test_keep_frames_zero_and_larger(tmp_path, series):
    threshold = 0.5 * (series['force'][1] + series['force'][2])
    guard = 'check_stability %d max_force %.12g keep_frames %%d' % (INTERVAL, threshold)
    r0 = lj_case(tmp_path / 'k0', guard=guard % 0)
    assert r0.returncode != 0
    assert [f['step'] for f in parse_extxyz(tmp_path / 'k0' / 'stability_frames.xyz')] == [30]
    r5 = lj_case(tmp_path / 'k5', guard=guard % 5)
    assert r5.returncode != 0
    assert [f['step'] for f in parse_extxyz(tmp_path / 'k5' / 'stability_frames.xyz')] == [10, 20, 30]
    r2 = lj_case(tmp_path / 'k2', guard=guard % 2)
    assert [f['step'] for f in parse_extxyz(tmp_path / 'k2' / 'stability_frames.xyz')] == [20, 30]


def test_first_violated_criterion_named_and_all_reported(tmp_path, series):
    thr_t = 0.5 * (series['temperature'][1] + series['temperature'][2])
    thr_f = 0.5 * (series['force'][1] + series['force'][2])
    result = lj_case(
        tmp_path, guard='check_stability 10 max_force %.12g max_temperature %.12g' % (thr_f, thr_t))
    out = text_of(result)
    assert result.returncode != 0
    assert 'max_temperature' in out and 'max_force' in out


# ---------------------------------------------------------------- AC-3 no false alarm

def run_outputs(workdir):
    return ((workdir / 'thermo.out').read_text(), (workdir / 'dump.xyz').read_text())


def test_no_false_alarm_identical_output(tmp_path):
    base = lj_case(tmp_path / 'base1', steps=200)
    base2 = lj_case(tmp_path / 'base2', steps=200)
    guard = ('check_stability 10 max_temperature 1e9 max_potential_change 1e9 min_distance 0.01 '
             'max_force 1e9')
    guarded = lj_case(tmp_path / 'guard', guard=guard, steps=200)
    assert base.returncode == 0 and guarded.returncode == 0, text_of(guarded)[-1500:]
    ref = run_outputs(tmp_path / 'base1')
    if ref != run_outputs(tmp_path / 'base2'):
        pytest.skip('the unguarded run is not repeatable on this GPU')
    assert run_outputs(tmp_path / 'guard') == ref
    assert not (tmp_path / 'guard' / 'stability_frames.xyz').exists()


# ---------------------------------------------------------------- AC-4 the s3 scenario

def test_collapse_cooled_by_thermostat_seen_by_potential_change(tmp_path):
    ens = 'ensemble nvt_bdp 300 300 20 seed 7'
    # deep well (5 eV): the pair falls in, heats up and is cooled back by the thermostat
    base = lj_case(tmp_path / 'base', ensemble=ens, eps=5.0, steps=3000, dump=True, interval=3000)
    assert base.returncode == 0, text_of(base)[-1500:]
    thermo = np.loadtxt(tmp_path / 'base' / 'thermo.out')
    frames = parse_extxyz(tmp_path / 'base' / 'dump.xyz')
    assert thermo[:, 0].max() > 5000.0                      # the collapse heated the pair
    assert thermo[-50:, 0].mean() < 1500.0                  # ... and it is cooled back
    # U dropped by >= 1 eV/atom between the start (5.2 A) and the end
    u_end = frames[-1]['energy_atom'].sum()
    assert u_end / 2 < -1.0
    # max_temperature evaluated only at the end does not see it
    late = lj_case(tmp_path / 'late', ensemble=ens, eps=5.0, steps=3000, interval=3000,
                   guard='check_stability 3000 max_temperature 3000')
    assert late.returncode == 0, text_of(late)[-1500:]
    # max_potential_change sees it
    early = lj_case(tmp_path / 'early', ensemble=ens, eps=5.0, steps=3000,
                    guard='check_stability 10 max_potential_change 1.0')
    out = text_of(early)
    assert early.returncode != 0 and 'max_potential_change' in out, out[-1500:]
    m = re.search(r'step (\d+): \|U - U0\|/N', out)
    assert int(m.group(1)) < 3000


# ---------------------------------------------------------------- charges (qNEP)

WATER = STRUCTURES / 'water-nat63-from-md.xyz'


def qnep_case(workdir, guard=None, steps=1, interval=1):
    lines = ['potential nep.txt', 'kspace ewald', 'velocity 300 seed 1', 'ensemble nve', 'time_step 1',
             'dump_thermo %d' % interval,
             'dump_xyz %d dump.xyz velocity force charge precision double' % interval]
    if guard:
        lines += [guard]
    lines += ['run %d' % steps]
    return run_gpumd(workdir, lines, {
        'nep.txt': (MODELS_DIR / 'qnep_mode1_water.txt').read_text(),
        'model.xyz': WATER.read_text()})


def test_max_charge_stops_below_static_value(tmp_path):
    base = qnep_case(tmp_path / 'static')
    assert base.returncode == 0, text_of(base)[-1500:]
    q = parse_extxyz(tmp_path / 'static' / 'dump.xyz')[-1]['charge'][:, 0]
    qmax = np.abs(q).max()
    imax = int(np.abs(q).argmax())
    result = qnep_case(tmp_path / 'guard', guard='check_stability 1 max_charge %.9g' % (0.5 * qmax))
    out = text_of(result)
    assert result.returncode != 0 and 'max_charge' in out, out[-1500:]
    assert 'step 1:' in out and ('atom %d ' % imax) in out
    frames = parse_extxyz(tmp_path / 'guard' / 'stability_frames.xyz')
    assert len(frames) == 1 and 'charge' in frames[0]
    assert np.abs(frames[0]['charge']).max() > 0.5 * qmax
    assert np.abs(frames[0]['charge'][:, 0]).argmax() == imax
    m = re.search(r'\|q\| = ([0-9.eE+-]+) e of atom', out)
    assert float(m.group(1)) == pytest.approx(qmax, rel=1e-6)
    assert not (tmp_path / 'guard' / 'restart.xyz').exists()


def test_qnep_no_false_alarm_identical_output(tmp_path):
    guard = ('check_stability 5 max_temperature 1e9 max_potential_change 1e9 min_distance 0.1 '
             'max_force 1e9 max_charge 1e9')
    b1 = qnep_case(tmp_path / 'b1', steps=20, interval=5)
    b2 = qnep_case(tmp_path / 'b2', steps=20, interval=5)
    g = qnep_case(tmp_path / 'g', guard=guard, steps=20, interval=5)
    assert b1.returncode == 0 and g.returncode == 0, text_of(g)[-1500:]
    ref = run_outputs(tmp_path / 'b1')
    if ref != run_outputs(tmp_path / 'b2'):
        pytest.skip('the unguarded qNEP run is not repeatable on this GPU')
    assert run_outputs(tmp_path / 'g') == ref


def test_min_distance_periodic_triclinic_and_small_box(tmp_path):
    """The distance kernel against a brute-force oracle: triclinic box, atoms across the
    periodic boundary, a box thinner than 2 * min_distance, and an open direction."""
    rng = np.random.default_rng(3)
    cases = [
        ('"12 0 0 3 11 0 2 1.5 10"', 'T T T', 1.0),
        ('"4 0 0 0 4 0 0 0 4"', 'T T T', 2.5),
        ('"15 0 0 0 15 0 0 0 15"', 'T T F', 1.2),
        ('"2.2 0 0 0 20 0 0 0 20"', 'T T T', 1.0),
    ]
    for idx, (lattice, pbc, rmin) in enumerate(cases):
        lat = np.array([float(x) for x in lattice.strip('"').split()]).reshape(3, 3)
        n = 40
        s = rng.random((n, 3))
        pos = s @ lat
        if idx == 0:  # an atom pair across the boundary
            pos[1] = pos[0] + np.array([0.0, 0.0, 0.0])
            s[1] = np.array([0.999, 0.5, 0.5])
            s[0] = np.array([0.001, 0.5, 0.5])
            pos = s @ lat
        flags = [p == 'T' for p in pbc.split()]
        # brute force over images
        best = np.inf
        for i in range(n):
            for j in range(i, n):
                for na in range(-3 if flags[0] else 0, 4 if flags[0] else 1):
                    for nb in range(-3 if flags[1] else 0, 4 if flags[1] else 1):
                        for nc in range(-3 if flags[2] else 0, 4 if flags[2] else 1):
                            if i == j and na == nb == nc == 0:
                                continue
                            d = pos[j] + np.array([na, nb, nc]) @ lat - pos[i]
                            best = min(best, np.linalg.norm(d))
        xyz = ['%d' % n, 'Lattice=%s Properties=species:S:1:pos:R:3 pbc="%s"' % (lattice, pbc)]
        xyz += ['C %.12f %.12f %.12f' % tuple(p) for p in pos]
        d = tmp_path / ('c%d' % idx)
        # zero-strength LJ (no forces), no motion: the guard sees the initial geometry
        result = run_gpumd(
            d,
            ['potential lj.txt', 'ensemble nve', 'time_step 0', 'check_stability 1 min_distance %.12g'
             % (best + 1e-6), 'run 1'],
            {'model.xyz': '\n'.join(xyz) + '\n', 'lj.txt': 'lj 1 C\n0.0 3.4 2.0\n'})
        out = text_of(result)
        assert result.returncode != 0, (idx, best, out[-800:])
        m = re.search(r'shortest distance ([0-9.eE+-]+) A between atoms', out)
        assert float(m.group(1)) == pytest.approx(best, rel=1e-9), (idx, out[-800:])
        # a threshold just below the shortest distance does not trigger
        result = run_gpumd(
            d / 'ok',
            ['potential lj.txt', 'ensemble nve', 'time_step 0', 'check_stability 1 min_distance %.12g'
             % (best - 1e-6), 'run 1'],
            {'model.xyz': '\n'.join(xyz) + '\n', 'lj.txt': 'lj 1 C\n0.0 3.4 2.0\n'})
        assert result.returncode == 0, (idx, best, text_of(result)[-800:])


def brute_force_shortest(pos, lat, flags, rmin):
    """Shortest distance over all periodic images (and the self image) by brute force.
    Image range per periodic direction: +-(ceil(rmin / perpendicular width) + 1) lattice vectors,
    which contains every image closer than rmin."""
    vol = abs(np.linalg.det(lat))
    widths = [vol / np.linalg.norm(np.cross(lat[(a + 1) % 3], lat[(a + 2) % 3])) for a in range(3)]
    rng_a = [int(np.ceil(rmin / w)) + 1 if f else 0 for w, f in zip(widths, flags)]
    shifts = np.array([[i, j, k] for i in range(-rng_a[0], rng_a[0] + 1)
                       for j in range(-rng_a[1], rng_a[1] + 1)
                       for k in range(-rng_a[2], rng_a[2] + 1)]) @ lat
    best = np.inf
    for i in range(len(pos)):
        d = pos[i:, None, :] + shifts[None, :, :] - pos[i]
        dist = np.linalg.norm(d, axis=2)
        if True:
            dist[0][np.all(shifts == 0, axis=1)] = np.inf  # the atom itself
        best = min(best, dist.min())
    return best


# (id, lattice, number of atoms, seeds, min_distance). Thin periodic cells (perpendicular width
# 1.0-1.1 A, below min_distance). The skew_two case (2 atoms, seed 3) needs the image (-2, ...) of
# the lattice vectors: the shortest distance is 2.2336 A, whereas scanning only the shifts
# -1, 0, +1 of the lattice vectors gives 2.4495 A, so a threshold of 2.3 A is missed (false
# negative) by a +-1 scan. (The LJ neighbour list of gpumd does not run in cells thinner than
# about 1 A, so thinner cells cannot be tested through the executable.)
THIN = [
    ('orth_x1', '"1.0 0 0 0 6 0 0 0 6"', 40, [0, 1, 2], 2.5),
    ('skew_dense', '"8 0 0 7.5 1.0 0 0 0 8"', 40, [0, 1, 2], 2.5),
    ('skew_two', '"8 0 0 7 1 2 0 0 8"', 2, [3], 2.3),
]


@pytest.mark.parametrize('lattice,n,seeds,rmin', [t[1:] for t in THIN], ids=[t[0] for t in THIN])
def test_min_distance_box_thinner_than_threshold(tmp_path, lattice, n, seeds, rmin):
    """A periodic cell thinner than min_distance (also skewed): all images closer than
    min_distance count, so the reported shortest distance equals the brute-force one.
    Brute-force image range: +-(ceil(rmin / perpendicular width) + 1) lattice vectors."""
    lat = np.array([float(x) for x in lattice.strip('"').split()]).reshape(3, 3)
    for seed in seeds:
        pos = np.random.default_rng(seed).random((n, 3)) @ lat
        best = brute_force_shortest(pos, lat, [True] * 3, rmin)
        xyz = ['%d' % n, 'Lattice=%s Properties=species:S:1:pos:R:3 pbc="T T T"' % lattice]
        xyz += ['C %.12f %.12f %.12f' % tuple(p) for p in pos]
        result = run_gpumd(
            tmp_path / ('s%d' % seed),
            ['potential lj.txt', 'ensemble nve', 'time_step 0',
             'check_stability 1 min_distance %.12g' % rmin, 'run 1'],
            {'model.xyz': '\n'.join(xyz) + '\n', 'lj.txt': 'lj 1 C\n0.0 3.4 0.9\n'})
        out = text_of(result)
        assert best < rmin
        assert result.returncode != 0, out[-800:]
        m = re.search(r'shortest distance ([0-9.eE+-]+) A between atoms', out)
        assert float(m.group(1)) == pytest.approx(best, rel=1e-9), (seed, best, out[-500:])


def test_min_distance_huge_ratio_does_not_overflow(tmp_path):
    """width / min_distance far above INT_MAX must not overflow the cell count."""
    xyz = ('2\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
           'C 10 10 10\nC 11 10 10\n')
    result = run_gpumd(
        tmp_path,
        ['potential lj.txt', 'ensemble nve', 'time_step 0',
         'check_stability 1 min_distance 1e-15', 'run 1'],
        {'model.xyz': xyz, 'lj.txt': 'lj 1 C\n0.0 3.4 2.0\n'})
    assert result.returncode == 0, text_of(result)[-500:]


# ---------------------------------------------------------------- AC-6 refusals

BAD = [
    ('check_stability 10', 'at least one criterion'),
    ('check_stability', 'at least one criterion'),
    ('check_stability 0 max_force 1', 'interval'),
    ('check_stability -5 max_force 1', 'interval'),
    ('check_stability x max_force 1', 'interval'),
    ('check_stability 10 max_force 0', 'positive'),
    ('check_stability 10 max_force -1', 'positive'),
    ('check_stability 10 max_temperature abc', 'positive'),
    ('check_stability 10 max_potential_change 0', 'positive'),
    ('check_stability 10 min_distance -0.5', 'positive'),
    ('check_stability 10 max_force', 'missing value'),
    ('check_stability 10 max_force 1 max_force 2', 'twice'),
    ('check_stability 10 bogus 1', 'unknown token'),
    ('check_stability 10 keep_frames 3', 'at least one criterion'),
    ('check_stability 10 max_force 1 keep_frames -1', 'keep_frames'),
    ('check_stability 10 max_force 1 keep_frames 1.5', 'keep_frames'),
    ('check_stability 10 max_charge 1', 'no charges'),
]


@pytest.mark.parametrize('line,expected', BAD)
def test_bad_input_refused(tmp_path, line, expected):
    result = lj_case(tmp_path, guard=line, steps=20)
    out = text_of(result)
    assert result.returncode != 0
    assert 'check_stability' in out and expected in out, out[-800:]
    assert not (tmp_path / 'stability_frames.xyz').exists()


def test_two_keywords_in_one_run_refused(tmp_path):
    lines = ['potential lj.txt', 'ensemble nve', 'time_step 1',
             'check_stability 10 max_force 1e9', 'check_stability 10 max_temperature 1e9', 'run 20']
    result = run_gpumd(tmp_path, lines, {'model.xyz': lj_pair_xyz(5.2), 'lj.txt': LJ_TEXT})
    assert result.returncode != 0 and 'multiple check_stability' in text_of(result)


def test_refused_with_minimize(tmp_path):
    lines = ['potential lj.txt', 'check_stability 10 max_force 1e9', 'minimize sd 1e-6 10',
             'ensemble nve', 'time_step 1', 'run 20']
    result = run_gpumd(tmp_path, lines, {'model.xyz': lj_pair_xyz(5.2), 'lj.txt': LJ_TEXT})
    assert result.returncode != 0 and 'cannot be combined with minimize' in text_of(result)


def test_refused_with_multi_replica(tmp_path):
    lines = ['potential lj.txt', 'multi_replica 2', 'ensemble nve', 'time_step 1',
             'check_stability 10 max_force 1e9', 'run 20']
    result = run_gpumd(tmp_path, lines, {'model.xyz': lj_pair_xyz(5.2), 'lj.txt': LJ_TEXT})
    assert result.returncode != 0
    assert 'multi_replica run supports' in text_of(result)


def test_guard_applies_per_run_only(tmp_path):
    """The action ends with its run: a second run without the keyword is not guarded."""
    lines = ['potential lj.txt', 'ensemble nve', 'time_step 1', 'check_stability 10 max_force 1e9',
             'run 20', 'ensemble nve', 'run 20']
    result = run_gpumd(tmp_path, lines, {'model.xyz': lj_pair_xyz(5.2), 'lj.txt': LJ_TEXT})
    assert result.returncode == 0, text_of(result)[-800:]


# ---------------------------------------------------------------- DR-008 review items C and E

TUC = 10.18051  # time unit conversion of GPUMD (natural -> fs)


def test_temperature_uses_3N_dof_with_fixed_group(tmp_path):
    """T = 2 KE / (3 N kB) of all atoms (no fixed-atom correction), as stated in the manual;
    the oracle is an independent computation from a double-precision dump of the same step."""
    n = 8
    rng = np.random.default_rng(5)
    pos = np.array([[5.0 * (k % 2) + 8, 5.0 * ((k // 2) % 2) + 8, 5.0 * (k // 4) + 8]
                    for k in range(n)])
    vel = rng.normal(0.0, 0.003, (n, 3))  # A/fs
    group = [1 if k < 3 else 0 for k in range(n)]  # 3 fixed atoms
    xyz = ['%d' % n, 'Lattice="30 0 0 0 30 0 0 0 30" '
           'Properties=species:S:1:pos:R:3:vel:R:3:group:I:1 pbc="T T T"']
    xyz += ['C %.6f %.6f %.6f %.8f %.8f %.8f %d' % (*p, *v, g) for p, v, g in zip(pos, vel, group)]
    result = run_gpumd(
        tmp_path,
        ['potential lj.txt', 'ensemble nve', 'fix 1', 'time_step 1', 'dump_thermo 10',
         'dump_xyz 10 dump.xyz mass velocity precision double',
         'check_stability 10 max_temperature 1e-6', 'run 100'],
        {'model.xyz': '\n'.join(xyz) + '\n', 'lj.txt': 'lj 1 C\n0.0 3.4 2.0\n'})
    out = text_of(result)
    assert result.returncode != 0, out[-800:]
    m = re.search(r'step 10: temperature ([0-9.eE+-]+) K', out)
    frame = parse_extxyz(tmp_path / 'dump.xyz')[0]
    v = frame['vel'] * TUC  # natural units
    ke = 0.5 * np.sum(frame['mass'][:, 0] * np.sum(v * v, axis=1))
    t_oracle = 2.0 * ke / (3.0 * n * K_B)
    assert float(m.group(1)) == pytest.approx(t_oracle, rel=1e-6)
    assert t_oracle > 0.0 and (np.abs(frame['vel'][:3]).max() == 0.0)  # the fixed atoms rest


@pytest.fixture(scope='module')
def nep_water_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('water')


def md_with_guard(workdir, nep_text, extra_potential_lines, kspace_lines):
    """63-atom water, NVE, guard that does not trigger at the first evaluation but at the second
    (max_potential_change tiny): the message reports U and U0, compared with thermo.out."""
    lines = ['potential nep.txt'] + extra_potential_lines + kspace_lines + [
        'velocity 300 seed 1', 'ensemble nve', 'time_step 0.5', 'dump_thermo 5',
        'check_stability 5 max_potential_change 1e-9', 'run 20']
    return run_gpumd(workdir, lines, {'nep.txt': nep_text, 'model.xyz': WATER.read_text()})


@pytest.mark.parametrize('name,model,extra,kspace', [
    ('qnep_ewald', 'qnep_mode1_water.txt', [], ['kspace ewald']),
    ('nep_dftd3', 'nep_water.txt', ['dftd3 pbe 12 6'], []),
], ids=['qnep_ewald', 'nep_dftd3'])
def test_guard_potential_energy_equals_thermo(tmp_path, name, model, extra, kspace):
    """potential_per_atom is filled every step: the guard's U and U0 equal the potential energy
    of thermo.out at the steps of the two evaluations."""
    result = md_with_guard(tmp_path, (MODELS_DIR / model).read_text(), extra, kspace)
    out = text_of(result)
    assert result.returncode != 0, out[-800:]
    m = re.search(r'step 10: \|U - U0\|/N = ([0-9.eE+-]+) eV/atom .*\(U = ([-0-9.eE+]+) eV, '
                  r'U0 = ([-0-9.eE+]+) eV\)', out)
    assert m, out[-800:]
    thermo = np.loadtxt(tmp_path / 'thermo.out')
    pe = thermo[:, 2]  # columns: T KE PE ...
    u, u0 = float(m.group(2)), float(m.group(3))
    assert u == pytest.approx(pe[1], rel=1e-8, abs=1e-8)   # thermo step 10
    assert u0 == pytest.approx(pe[0], rel=1e-8, abs=1e-8)  # thermo step 5
    assert u != u0


LJ_FCC_A = 5.3


def fcc_xyz(rep, seed):
    rng = np.random.default_rng(seed)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pos = np.array([(np.array([i, j, k]) + b) * LJ_FCC_A
                    for i in range(rep) for j in range(rep) for k in range(rep) for b in basis])
    pos += rng.normal(0.0, 0.05, pos.shape)
    box = rep * LJ_FCC_A
    xyz = ['%d' % len(pos), 'Lattice="%g 0 0 0 %g 0 0 0 %g" Properties=species:S:1:pos:R:3 '
           'pbc="T T T"' % (box, box, box)]
    xyz += ['C %.8f %.8f %.8f' % tuple(p) for p in pos]
    return '\n'.join(xyz) + '\n', len(pos)


NPT_LJ = 'ensemble npt_scr 80 80 100 30 30 30 50 50 50 50 seed 3'


def npt_case(workdir, guard, steps=100, dump_every=20):
    xyz, _ = fcc_xyz(4, 1)
    lines = ['potential lj.txt', 'velocity 80 seed 2', 'time_step 2', NPT_LJ, 'dump_thermo 20',
             'dump_xyz %d dump.xyz velocity force precision double' % dump_every]
    if guard:
        lines += [guard]
    lines += ['run %d' % steps]
    return run_gpumd(workdir, lines, {'model.xyz': xyz, 'lj.txt': 'lj 1 C\n0.0104 3.4 8.5\n'})


def test_npt_min_distance_and_force_use_current_box(tmp_path):
    """Box shrinking under npt_scr (30 GPa target): the frame written on the violation carries
    the current box; min_distance and max_force are compared with brute-force values of that
    frame (box inverse is refreshed every step)."""
    result = npt_case(tmp_path, 'check_stability 40 min_distance 100 max_force 1e-9', steps=100)
    out = text_of(result)
    assert result.returncode != 0, out[-800:]
    frame = parse_extxyz(tmp_path / 'stability_frames.xyz')[-1]
    assert frame['step'] == 40
    initial = 4 * LJ_FCC_A
    assert frame['lattice'][0, 0] < 0.999 * initial  # the box has shrunk
    d = min_distance(frame)
    m = re.search(r'shortest distance ([0-9.eE+-]+) A between atoms', out)
    assert float(m.group(1)) == pytest.approx(d, rel=1e-9)
    m = re.search(r'force component ([0-9.eE+-]+) eV/A of atom', out)
    assert float(m.group(1)) == pytest.approx(np.abs(frame['forces']).max(), rel=1e-12)


def test_npt_no_false_alarm_identical_output(tmp_path):
    guard = ('check_stability 20 max_temperature 1e9 max_potential_change 1e9 min_distance 0.1 '
             'max_force 1e9')
    b1 = npt_case(tmp_path / 'b1', None)
    b2 = npt_case(tmp_path / 'b2', None)
    g = npt_case(tmp_path / 'g', guard)
    assert b1.returncode == 0 and g.returncode == 0, text_of(g)[-800:]
    ref = run_outputs(tmp_path / 'b1')
    if ref != run_outputs(tmp_path / 'b2'):
        pytest.skip('the unguarded NPT run is not repeatable on this GPU')
    assert run_outputs(tmp_path / 'g') == ref


def test_refused_with_pimd(tmp_path):
    lines = ['potential lj.txt', 'ensemble pimd 4 300 300 100', 'time_step 0.5',
             'check_stability 10 max_force 1e9', 'run 20']
    result = run_gpumd(tmp_path, lines, {'model.xyz': lj_pair_xyz(5.2), 'lj.txt': LJ_TEXT})
    assert result.returncode != 0 and 'PIMD' in text_of(result), text_of(result)[-600:]
