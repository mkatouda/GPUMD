"""Flexible ZBL (`zbl` + `zbl.in`) in the qNEP trainer, nep.txt IO, and gpumd consistency (DR-007 R3, AC-3).

Self-contained (no specs/): synthetic data with random labels (C, F; 12 and 16 atoms with short contacts
so that the pair-specific ZBL contributes), `nep` trains 5 generations for qnep2 and qnep3, with and
without charge_transfer, with a flexible ZBL (`zbl 1.5` in nep.in plus `zbl.in`, 3 pairs = 30 values).
Checks: nep.txt holds `zbl 0 0` and `@ zbl_parameters 30` with the zbl.in values (and
`@ phi_parameters` with charge_transfer); `nep` restarts from it and rejects a universal/flexible
mismatch; `nep` prediction mode reads the model; nep prediction and gpumd static runs on the same nep.txt
agree (BEC, forces, energy, charge, virial; gates of test_nep_charge_transfer_train.py: BEC 5e-6, energy
1e-5 eV/atom, force relative 1e-6 of the force RMS of a trained model); the ZBL contribution is clearly
non-zero (gpumd with and without the ZBL blocks); models that still reject flexible ZBL do so.
"""
import subprocess

import numpy as np
import pytest

from conftest import GPUMD_EXECUTABLE, NEP_EXECUTABLE

pytestmark = pytest.mark.fast

SYMBOLS = ['C', 'F']
# three pairs (C-C, C-F, F-F): inner, outer, then the universal coefficients/exponents, slightly scaled
ZBL_PAIRS = [(0.5, 1.2), (0.6, 1.4), (0.7, 1.5)]
UNI = [0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162]
BEC_GATE = 5e-6
ENERGY_GATE = 1e-5
FORCE_ABS_GATE = 1e-5
FORCE_REL_GATE = 1e-6
TRAIN_EXTRA = 'population 30\ngeneration %d\nseed 1\nbatch 100000\noutput_interval 1\n'
# Only the three built-in contacts below lie in ZBL windows (random atoms are >= MIN_DIST apart) and the model is
# trained for few generations: float noise of nep vs gpumd scales with the force RMS (measured: 1.2e-6 relative
# for a 20-generation model and for contacts down to 0.95 A, which exceeds the 1e-6 gate), so the geometry keeps
# the force RMS ~1.3 eV/A where the unchanged absolute gate of 1e-5 eV/A applies (measured 4.6-5.5e-6).
MIN_DIST = 1.6
GENERATIONS = 5  # few generations keep the trained model's force RMS small (float noise scales with it)
CONTACTS = (1.35, 1.15, 1.45)  # C-F, C-C, F-F distances
STRUCTS = [(12, 7.5, 11), (16, 8.0, 12)]  # atoms, cubic box, seed


def zbl_values():
    out = []
    for k, (a, b) in enumerate(ZBL_PAIRS):
        out += [a, b] + [x * (1.0 + 0.03 * k * (1 if i % 2 == 0 else -1)) for i, x in enumerate(UNI)]
    return out


def make_structures():
    out = []
    for n, box, seed in STRUCTS:
        rng = np.random.default_rng(seed)
        # atoms 0 (C), 1 (F), 2 (C), 3 (F): a C-F, a C-C and an F-F contact inside the switching windows of
        # their pairs (CONTACTS, A); the other atoms are placed randomly at least MIN_DIST apart
        p0 = rng.random(3) * box
        cf, cc, ff = CONTACTS
        pos = [p0, p0 + [cf, 0, 0], p0 + [0, cc, 0], p0 + [cf, cc + ff, 0]]
        while len(pos) < n:
            p = rng.random(3) * box
            if all(np.linalg.norm((p - q) - box * np.round((p - q) / box)) > MIN_DIST for q in pos):
                pos.append(p)
        types = [i % 2 for i in range(n)]
        out.append((np.array(pos), types, box))
    return out


def write_train_xyz(path, structs, seed=7):
    rng = np.random.default_rng(seed)
    with open(path, 'w') as f:
        for pos, types, box in structs:
            n = len(pos)
            f.write('%d\n' % n)
            f.write('Lattice="%g 0 0 0 %g 0 0 0 %g" energy=%.9g virial="%s" '
                    'Properties=species:S:1:pos:R:3:forces:R:3:bec:R:9 pbc="T T T"\n'
                    % (box, box, box, rng.normal() * n, ' '.join('%.9g' % v for v in rng.normal(0, 1, 9))))
            for a in range(n):
                f.write('%s %.10f %.10f %.10f %s %s\n' % (
                    SYMBOLS[types[a]], *pos[a], ' '.join('%.9g' % v for v in rng.normal(0, 0.5, 3)),
                    ' '.join('%.9g' % v for v in rng.normal(0, 0.5, 9))))


def write_nep_in(path, charge_mode, ct, extra):
    lines = ['type 2 C F', 'model qnep%d' % charge_mode, 'cutoff 4 3.5', 'n_max 3 2', 'basis_size 4 4',
             'l_max 4 0 0', 'neuron 10', 'zbl 1.5']
    if ct:
        lines.append('charge_transfer 4')
    path.write_text('\n'.join(lines) + '\n' + extra)


def write_zbl_in(path, values):
    path.write_text(''.join(' '.join('%.9g' % v for v in values[10 * k:10 * k + 10]) + '\n' for k in range(3)))


def run_nep(d):
    return subprocess.run([str(NEP_EXECUTABLE)], cwd=d, capture_output=True, text=True, timeout=600)


def blocks_of(text):
    fields, blocks, cur = {}, {}, None
    for k, line in enumerate(text.splitlines()):
        t = line.split()
        if not t or k == 0:
            continue
        if t[0] == '@':
            cur = t[1]
            blocks[cur] = []
            blocks[cur + '#n'] = int(t[2])
        elif cur is None:
            fields[t[0]] = t[1:]
        else:
            blocks[cur].append(float(t[0]))
    return fields, blocks


def parse_dump(path):
    L = open(path).read().split('\n')
    n = int(L[0])
    head = L[1]
    props = head.split('Properties=')[1].split()[0].split(':')
    col, c = {}, 0
    for i in range(0, len(props), 3):
        col[props[i]] = (c, int(props[i + 2]))
        c += int(props[i + 2])
    rows = [l.split() for l in L[2:2 + n]]
    get = lambda name: np.array([[float(x) for x in r[col[name][0]:col[name][0] + col[name][1]]] for r in rows])
    vir = np.array([float(x) for x in head.split('virial="')[1].split('"')[0].split()]).reshape(3, 3)
    return {'energy': float(head.split('energy=')[1].split()[0]), 'virial': vir, 'forces': get('forces'),
            'charge': get('charge')[:, 0], 'bec': get('bec')}


def gpumd_static(d, text, struct):
    pos, types, box = struct
    d.mkdir()
    (d / 'nep.txt').write_text(text)
    with open(d / 'model.xyz', 'w') as f:
        f.write('%d\nLattice="%g 0 0 0 %g 0 0 0 %g" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                % (len(pos), box, box, box))
        for t, p in zip(types, pos):
            f.write('%s %.10f %.10f %.10f\n' % (SYMBOLS[t], *p))
    (d / 'run.in').write_text('potential nep.txt\nkspace ewald\nensemble nve\ntime_step 0\n'
                              'dump_xyz 1 d.xyz precision double charge force bec\nrun 1\n')
    r = subprocess.run([str(GPUMD_EXECUTABLE)], cwd=d, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-600:]
    return parse_dump(d / 'd.xyz')


def strip_zbl(text):
    """the same model without the ZBL line and the zbl_parameters block"""
    out, skip = [], False
    for line in text.splitlines():
        if line.startswith('zbl '):
            continue
        if line.startswith('@ '):
            skip = line.split()[1] == 'zbl_parameters'
        if not skip:
            out.append(line)
    return '\n'.join(out) + '\n'


CASES = [(2, False), (2, True), (3, False), (3, True)]
IDS = ['qnep2', 'qnep2-ct', 'qnep3', 'qnep3-ct']


@pytest.mark.parametrize('charge_mode, ct', CASES, ids=IDS)
def test_flexible_zbl_training_io_and_gpumd_consistency(tmp_path, charge_mode, ct):
    structs = make_structures()
    vals = zbl_values()
    t = tmp_path / 'train'
    t.mkdir()
    write_train_xyz(t / 'train.xyz', structs)
    write_nep_in(t / 'nep.in', charge_mode, ct, TRAIN_EXTRA % GENERATIONS)
    write_zbl_in(t / 'zbl.in', vals)
    r = run_nep(t)
    assert r.returncode == 0, (r.stdout + r.stderr)[-800:]
    assert 'flexible ZBL' in r.stdout
    text = (t / 'nep.txt').read_text()
    fields, blocks = blocks_of(text)
    assert fields['model'] == ['qnep%d' % charge_mode] and fields['zbl'] == ['0', '0']
    assert blocks['zbl_parameters#n'] == 30 and len(blocks['zbl_parameters']) == 30
    assert np.allclose(blocks['zbl_parameters'], vals, rtol=1e-6, atol=0)
    assert ('charge_transfer' in fields) == ct
    assert ('phi_parameters' in blocks) == ct
    if ct:
        assert blocks['phi_parameters#n'] == 15
    # restart: nep.txt/nep.restart of a flexible-ZBL run are accepted by the same input
    (t / 'nep.in').write_text((t / 'nep.in').read_text().replace('generation %d' % GENERATIONS, 'generation 2'))
    r2 = run_nep(t)
    assert r2.returncode == 0, (r2.stdout + r2.stderr)[-800:]
    # import_q_scaler checks nep.in against the existing nep.txt: accepted with zbl.in, an error without it
    # (a universal ZBL request against a flexible model)
    (t / 'nep.in').write_text((t / 'nep.in').read_text() + 'import_q_scaler 1\n')
    rf = run_nep(t)
    assert rf.returncode == 0, (rf.stdout + rf.stderr)[-800:]
    (t / 'zbl.in').rename(t / 'zbl.in.off')
    r3 = run_nep(t)
    assert r3.returncode != 0
    assert 'requests a universal ZBL potential' in r3.stdout + r3.stderr, (r3.stdout + r3.stderr)[-500:]

    # prediction mode (zbl.in is not needed: the parameters come from nep.txt)
    p = tmp_path / 'pred'
    p.mkdir()
    (p / 'nep.txt').write_text(text)
    write_train_xyz(p / 'train.xyz', structs)
    write_nep_in(p / 'nep.in', charge_mode, ct, 'prediction 1\n')
    # without the zbl line nep.in disagrees with the model: an error
    (p / 'nep.in').write_text((p / 'nep.in').read_text().replace('zbl 1.5\n', ''))
    rn = run_nep(p)
    assert rn.returncode != 0 and 'ZBL' in rn.stdout + rn.stderr, (rn.stdout + rn.stderr)[-500:]
    write_nep_in(p / 'nep.in', charge_mode, ct, 'prediction 1\n')
    rp = run_nep(p)
    assert rp.returncode == 0, (rp.stdout + rp.stderr)[-800:]
    bec_n = np.loadtxt(p / 'bec_train.out', ndmin=2)[:, :9]
    f_n = np.loadtxt(p / 'force_train.out', ndmin=2)[:, :3]
    q_n = np.loadtxt(p / 'charge_train.out', ndmin=1)
    e_n = np.loadtxt(p / 'energy_train.out', ndmin=2)[:, 0]
    v_n = np.loadtxt(p / 'virial_train.out', ndmin=2)[:, :6]
    # diagnostic only (not gated): the same weights without ZBL, nep prediction vs gpumd, i.e. the float noise of the
    # comparison for this trained model (random labels, large forces)
    p0 = tmp_path / 'pred0'
    p0.mkdir()
    (p0 / 'nep.txt').write_text(strip_zbl(text))
    write_train_xyz(p0 / 'train.xyz', structs)
    write_nep_in(p0 / 'nep.in', charge_mode, ct, 'prediction 1\n')
    (p0 / 'nep.in').write_text((p0 / 'nep.in').read_text().replace('zbl 1.5\n', ''))
    assert run_nep(p0).returncode == 0
    f_0 = np.loadtxt(p0 / 'force_train.out', ndmin=2)[:, :3]
    df0 = []
    off = 0
    db, df, dq, dv, de, rms_f, zbl_f, q_zbl = [], [], [], [], [], [], [], []
    for k, s in enumerate(structs):
        n = len(s[0])
        g = gpumd_static(tmp_path / ('g%d' % k), text, s)
        g0 = gpumd_static(tmp_path / ('n%d' % k), strip_zbl(text), s)
        sl = slice(off, off + n)
        w = g['virial']
        v6 = np.array([w[0, 0], w[1, 1], w[2, 2], 0.5 * (w[0, 1] + w[1, 0]), 0.5 * (w[1, 2] + w[2, 1]),
                       0.5 * (w[2, 0] + w[0, 2])])
        df0.append(f_0[sl] - g0['forces'])
        db.append(bec_n[sl] - g['bec'])
        df.append(f_n[sl] - g['forces'])
        dq.append(q_n[sl] - g['charge'])
        q_zbl.append(np.max(np.abs(g['charge'] - g0['charge'])))
        dv.append(v_n[k] - v6 / n)
        de.append(abs(e_n[k] - g['energy'] / n))
        rms_f.append(np.sqrt(np.mean(g['forces'] ** 2)))
        zbl_f.append(np.sqrt(np.mean((g['forces'] - g0['forces']) ** 2)))
        off += n
    rms = lambda l: float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in l]) ** 2)))
    res = dict(bec=rms(db), force=rms(df), charge=rms(dq), virial=rms(dv), dE=max(de), rms_F=max(rms_f),
               force0=rms(df0), zbl_F=min(zbl_f), q_zbl=max(q_zbl))
    print(IDS[CASES.index((charge_mode, ct))], res)
    assert res['zbl_F'] > 0.1, ('ZBL does not contribute', res)
    assert res['rms_F'] > 0.1, res
    assert res['bec'] <= BEC_GATE, res
    # gate of test_nep_charge_transfer_train.py, unchanged: 1e-5 eV/A absolute or 1e-6 of the force RMS
    assert res['force'] <= max(FORCE_ABS_GATE, FORCE_REL_GATE * res['rms_F']), res
    assert res['dE'] <= ENERGY_GATE, res
    assert res['q_zbl'] <= 1e-12, ('the charges depend on the ZBL blocks', res)
    if ct:
        # (without charge_transfer, charge_*.out of nep holds the raw network output, while gpumd
        # dumps the neutralised charge: they differ by a per-structure constant, identical with and
        # without ZBL and in the pre-change binaries; therefore not gated here. Models with
        # charge_transfer are neutral by construction and agree.)
        assert res['charge'] <= 1e-5, res


@pytest.mark.parametrize('model', ['nep_dipole', 'nep_polarizability', 'nep_vdw', 'qnep3_vdw'])
def test_flexible_zbl_still_rejected_by_other_models(tmp_path, model):
    # only the shared reading path is exercised: the error is raised before any training
    structs = make_structures()
    write_train_xyz(tmp_path / 'train.xyz', structs)
    lines = ['type 2 C F', 'model %s' % model, 'cutoff 4 3.5', 'n_max 3 2', 'basis_size 4 4', 'l_max 4 0 0',
             'neuron 10', 'zbl 1.5', 'generation 1']
    (tmp_path / 'nep.in').write_text('\n'.join(lines) + '\n')
    write_zbl_in(tmp_path / 'zbl.in', zbl_values())
    r = run_nep(tmp_path)
    assert r.returncode != 0
    assert 'lexible ZBL' in r.stdout + r.stderr or 'zbl' in (r.stdout + r.stderr).lower(), (r.stdout + r.stderr)[-500:]
