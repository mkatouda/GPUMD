"""S1: extreme coordination numbers (underflow of the factorized C6 weights, DR-001 Phase 7 item (v)).
Structures are built here, the independent reference is simple-dftd3 (PBE, BJ, two-body, same cutoffs as GPUMD,
exactly generate_reference.py of tests_pytest/fixtures/dftd3), and GPUMD's D3 contribution is (with D3) - (without D3), NEP89.
usage: extreme_cn.py BINARY... --work DIR [--ref-python PY]   (ase/numpy env; reference computed in a subprocess of
the conda env that has dftd3).  Prints per case and binary: finite?, |dE|, max|dF|, max|dV| against the reference,
the AC-1 caps (rel 2e-6 energy / virial, 1e-5 eV/A forces) and the NEP-noise estimate (two runs without D3)."""
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write, read

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
REL_CAP, FORCE_CAP = 2e-6, 1e-5


def cases():
    rng = np.random.default_rng(777)
    c = {}
    # CN far above every reference CN: compressed lattices (large-box path, box > 2.5 rc), rc = 8 / 4
    h = bulk('H', 'fcc', a=1.9, cubic=True).repeat((11, 11, 11)); h.rattle(0.02, seed=1)
    c['H_fcc_compressed'] = (h, (8, 4), 'compressed H fcc a=1.9 A, 5324 atoms, huge CN, large box')
    cu = bulk('Cu', 'fcc', a=2.2, cubic=True).repeat((10, 10, 10)); cu.rattle(0.02, seed=2)
    c['Cu_fcc_compressed'] = (cu, (8, 4), 'compressed Cu fcc a=2.2 A, 4000 atoms, CN >> max reference CN, large box')
    # same, small-box path (periodic thickness <= 2.5 rc)
    hs = bulk('H', 'fcc', a=1.9, cubic=True).repeat((4, 4, 4)); hs.rattle(0.02, seed=3)
    c['H_fcc_compressed_smallbox'] = (hs, (12, 6), 'compressed H fcc a=1.9 A, 256 atoms, small-box path')
    # CN near 0: dilated crystals (CN exactly 0 when the nearest neighbor is beyond rc_cn, and tiny otherwise)
    for name, el, a, n in (('Cu_dilated_a5.2', 'Cu', 5.2, 7), ('Cu_dilated_a7.5', 'Cu', 7.5, 6),
                           ('Au_dilated_a9', 'Au', 9.0, 5), ('C_dilated_diamond_a9', 'C', 9.0, 5)):
        s = bulk(el, 'fcc' if el != 'C' else 'diamond', a=a, cubic=True).repeat((n, n, n)); s.rattle(0.02, seed=4)
        c[name] = (s, (8, 4), f'dilated {el}, a={a} A, CN ~ 0, large box')
    # isolated pairs at very short distance and a dense cluster (non-periodic, one cell)
    p = Atoms('H2Cu2', positions=[[0, 0, 0], [0, 0, 0.30], [10, 0, 0], [10, 0, 0.55]], cell=[40, 40, 40], pbc=True)
    p.center()
    c['short_pairs_periodic'] = (p, (12, 6), 'H2 at 0.30 A and Cu2 at 0.55 A in a 40 A box (large-box path)')
    pts = []
    while len(pts) < 70:
        x = rng.uniform(-1.6, 1.6, 3)
        if np.linalg.norm(x) < 1.6 and all(np.linalg.norm(x - y) > 0.45 for y in pts):
            pts.append(x)
    cl = Atoms('H70', positions=np.array(pts)); cl.center(vacuum=20.0); cl.pbc = False
    c['H70_dense_cluster'] = (cl, (12, 6), '70 H atoms within 1.6 A radius (CN ~ 40), non-periodic')
    pos = []
    while len(pos) < 70:
        x = rng.uniform(-2.2, 2.2, 3)
        if np.linalg.norm(x) < 2.2 and all(np.linalg.norm(x - y) > 0.8 for y in pos):
            pos.append(x)
    cm = Atoms(['Cu'] * 35 + ['H'] * 35, positions=np.array(pos)); cm.center(vacuum=20.0); cm.pbc = False
    c['CuH70_dense_cluster'] = (cm, (12, 6), '35 Cu + 35 H within 2.2 A radius, non-periodic')
    return c


def reference_main(out_json, xyz_dir):
    """run inside the env that has dftd3"""
    sys.path.insert(0, str(REPO / 'tests_pytest/fixtures/dftd3'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('gen', REPO / 'tests_pytest/fixtures/dftd3/generate_reference.py')
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    res = {}
    for name, (atoms, cut, note) in cases().items():
        write(Path(xyz_dir) / f'{name}.xyz', atoms, format='extxyz')
        e, f, v = gen.reference(atoms, cut)
        res[name] = dict(cutoffs=cut, note=note, natoms=len(atoms), energy=e, forces=f.tolist(), virial=v.tolist())
        print('ref', name, len(atoms), e, flush=True)
    json.dump(res, open(out_json, 'w'))


def run(binary, work, name, cut, d3, tag):
    d = Path(work) / f'{tag}_{name}_{int(bool(d3))}'
    d.mkdir(parents=True, exist_ok=True)
    atoms = read(Path(work) / 'xyz' / f'{name}.xyz')
    atoms.write(d / 'model.xyz', format='extxyz')
    lines = [f'potential {NEP}']
    if d3:
        lines.append(f'dftd3 pbe {cut[0]} {cut[1]}')
    lines += ['ensemble nve', 'time_step 0', 'dump_xyz 1 dump.xyz force precision double', 'run 1']
    (d / 'run.in').write_text('\n'.join(lines) + '\n')
    r = subprocess.run([str(binary)], cwd=d, capture_output=True, text=True)
    if r.returncode:
        return None
    txt = (d / 'dump.xyz').read_text().splitlines()
    e = float(re.search(r'energy=([-0-9.eE+]+)', txt[1]).group(1))
    v = np.array(re.search(r'virial="([^"]+)"', txt[1]).group(1).split(), float).reshape(3, 3)
    f = read(d / 'dump.xyz', format='extxyz').get_forces()
    return e, f, v


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('binaries', nargs='*')
    ap.add_argument('--work', required=True)
    ap.add_argument('--ref-python')
    ap.add_argument('--make-reference', action='store_true')
    a = ap.parse_args()
    work = Path(a.work); (work / 'xyz').mkdir(parents=True, exist_ok=True)
    ref_file = work / 'reference.json'
    if a.make_reference:
        reference_main(ref_file, work / 'xyz'); sys.exit()
    REF = json.load(open(ref_file))
    out = {}
    for name, ref in REF.items():
        cut = ref['cutoffs']
        rf = np.array(ref['forces']); rv = np.array(ref['virial'])
        row = {}
        for b in a.binaries:
            tag = Path(b).parent.parent.name + '_' + Path(b).parent.name
            n0 = run(b, work, name, cut, False, tag + 'a')
            n0b = run(b, work, name, cut, False, tag + 'b')
            d1 = run(b, work, name, cut, True, tag)
            if n0 is None or d1 is None or n0b is None:
                row[tag] = 'FAILED'; continue
            e, f, v = d1[0] - n0[0], d1[1] - n0[1], d1[2] - n0[2]
            finite = bool(np.isfinite([e]).all() and np.isfinite(f).all() and np.isfinite(v).all())
            res = dict(finite=finite, dE=abs(e - ref['energy']), dF=float(np.abs(f - rf).max()),
                       dV=float(np.abs(v - rv).max()),
                       capE=REL_CAP * abs(ref['energy']), capF=FORCE_CAP, capV=REL_CAP * float(np.abs(rv).max()),
                       noiseE=abs(n0b[0] - n0[0]), noiseF=float(np.abs(n0b[1] - n0[1]).max()),
                       noiseV=float(np.abs(n0b[2] - n0[2]).max()),
                       maxF_ref=float(np.abs(rf).max()), E_ref=ref['energy'],
                       contrib=(e, f.tolist(), v.tolist()))
            row[tag] = res
        out[name] = row
        for tag, r in row.items():
            if isinstance(r, str):
                print(f'{name:28s} {tag}: {r}'); continue
            ok = r['finite'] and r['dE'] <= r['capE'] and r['dF'] <= r['capF'] and r['dV'] <= r['capV']
            print(f"{name:28s} {tag:24s} {'OK ' if ok else 'OVER'} finite={r['finite']} dE {r['dE']:.2e}/{r['capE']:.2e} "
                  f"dF {r['dF']:.2e}/{r['capF']:.0e} dV {r['dV']:.2e}/{r['capV']:.2e} | NEP noise E {r['noiseE']:.1e} "
                  f"F {r['noiseF']:.1e} V {r['noiseV']:.1e} | E_D3 {r['E_ref']:.4g} max|F| {r['maxF_ref']:.3g}", flush=True)
        tags = [t for t in row if not isinstance(row[t], str)]
        if len(tags) == 2:
            x, y = row[tags[0]]['contrib'], row[tags[1]]['contrib']
            print(f"{'':28s} {tags[1]} - {tags[0]}: dE {abs(x[0]-y[0]):.2e} dF {np.abs(np.array(x[1])-np.array(y[1])).max():.2e} "
                  f"dV {np.abs(np.array(x[2])-np.array(y[2])).max():.2e}")
    for r in out.values():
        for t in r.values():
            if isinstance(t, dict):
                t.pop('contrib')
    json.dump(out, open(work / 'extreme_cn_result.json', 'w'), indent=1)
