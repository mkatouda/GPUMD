"""Run the VER-0012 measurements.  usage: run_study.py SCRATCH STAGE [...]
SCRATCH holds bin/gpumd_<variant> (build_variants.sh) and xyz/ (make_systems.py).
Stages: scan (a)+(f), variants (d)(e), empty (f), count (c).  Results: JSON lines on stdout and results/*.json.
Alternation: for variant stages the base and the variant binary are run alternately, 3 times each (base first).
"""
import json, subprocess, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
PY = sys.executable
S = Path(sys.argv[1])
stages = sys.argv[2:]
SCAN = ['w81', 'scan_162', 'scan_324', 'w648', 'scan_972', 'scan_1458', 'scan_2187', 'scan_2916', 'scan_4374',
        'scan_5832', 'scan_8748']


BIG = ['scan_17496', 'scan_34992']


def bench(binary, xyz, rc='12', tmp=None):
    out = S / 'tmp.json'
    r = subprocess.run([PY, str(HERE / 'bench.py'), str(S / 'bin' / f'gpumd_{binary}'), str(S / 'xyz' / f'{xyz}.xyz'),
                        str(S), '--rc', rc, '--json', str(out)], capture_output=True, text=True)
    try:
        return json.load(open(out)), r.stdout
    except Exception:
        print('FAILED', binary, xyz, r.stdout[-500:], r.stderr[-500:])
        return None, r.stdout


def kern(res, key):
    for k, v in res['kernels'].items():
        if key in k and 'extra' not in k or (key == 'extra' and 'extra' in k):
            return v['median_ms']


out = {}
for stage, lst, fn in [('scan', SCAN, 'scan_base.json'), ('scanbig', BIG, 'scan_big.json')]:
  if stage in stages:
    rows = []
    for x in lst:
        runs = []
        for rep in range(3):
            r, _ = bench('base', x)
            runs.append(r)
        row = dict(system=x, N=int(x.split('_')[-1]) if x.startswith('scan') else int(x[1:]))
        for name, key in [('nl', 'find_neighbor_list'), ('cn', 'coordination'), ('force', 'add_dftd3_force_small_box'),
                          ('extra', 'extra')]:
            vals = [kern(r, key) for r in runs]
            row[name] = statistics.median(vals)
            row[name + '_runs'] = vals
        row['total'] = statistics.median(r['d3_total_ms_median'] for r in runs)
        row['gap'] = statistics.median(r['d3_gap_ms_median'] for r in runs)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith('_runs')}), flush=True)
    json.dump(rows, open(RES / fn, 'w'), indent=1)
if 'variants' in stages:
    rows = []
    for x in ['w81', 'w648', 'scan_2187', 'scan_8748']:
        for v in ['noc6', 'f32atomic', 'nlf32', 'n1only', 'regacc']:
            tb, tv, fb, fv, nb, nv = [], [], [], [], [], []
            for rep in range(3):
                rb, _ = bench('base', x)
                rv, _ = bench(v, x)
                tb.append(rb['d3_total_ms_median']); tv.append(rv['d3_total_ms_median'])
                fb.append(kern(rb, 'add_dftd3_force_small_box')); fv.append(kern(rv, 'add_dftd3_force_small_box'))
                nb.append(kern(rb, 'find_neighbor_list')); nv.append(kern(rv, 'find_neighbor_list'))
            row = dict(system=x, variant=v, force_base=statistics.median(fb), force_var=statistics.median(fv),
                       nl_base=statistics.median(nb), nl_var=statistics.median(nv),
                       total_base=statistics.median(tb), total_var=statistics.median(tv),
                       force_runs_base=fb, force_runs_var=fv)
            row['force_speedup'] = row['force_base'] / row['force_var']
            row['nl_speedup'] = row['nl_base'] / row['nl_var']
            row['total_speedup'] = row['total_base'] / row['total_var']
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if not k.endswith('_runs_base') and 'runs' not in k}), flush=True)
    json.dump(rows, open(RES / 'variants.json', 'w'), indent=1)
if 'empty' in stages:
    rows = []
    for x in ['w81', 'w648']:
        for rep in range(3):
            r, _ = bench('empty', x)
            rows.append(dict(system=x, rep=rep, total=r['d3_total_ms_median'], span=r['d3_span_ms_median'],
                             gap=r['d3_gap_ms_median'], gaps_us=r['gaps_us_median'],
                             kernels={k: v['median_ms'] for k, v in r['kernels'].items()}))
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(RES / 'empty.json', 'w'), indent=1)
if 'large' in stages:
    def kl(res, key):
        for k, v in res['kernels'].items():
            if key in k and ('extra' not in k or key == 'extra'):
                return v['median_ms']
    rows = []
    for x in ['clus972', 'w41472']:
        for v in ['noc6L', 'xf32L']:
            tb, tv, fb, fv = [], [], [], []
            for rep in range(3):
                rb, _ = bench('base', x)
                rv, _ = bench(v, x)
                tb.append(rb['d3_total_ms_median']); tv.append(rv['d3_total_ms_median'])
                fb.append(kl(rb, 'force_large_box')); fv.append(kl(rv, 'force_large_box'))
            row = dict(system=x, variant=v, force_base=statistics.median(fb), force_var=statistics.median(fv),
                       total_base=statistics.median(tb), total_var=statistics.median(tv), force_runs_base=fb,
                       force_runs_var=fv)
            row['force_speedup'] = row['force_base'] / row['force_var']
            row['total_speedup'] = row['total_base'] / row['total_var']
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(RES / 'variants_large.json', 'w'), indent=1)
if 'count' in stages:
    lines = []
    for x, rc in [('w81', '12'), ('w648', '12'), ('w648', '20'), ('scan_2187', '12'), ('scan_8748', '12'),
                  ('cu864', '12'), ('cu864', '20'), ('c1728', '12'), ('c1728', '20')]:
        r = subprocess.run([PY, str(HERE / 'bench.py'), str(S / 'bin/gpumd_count'), str(S / 'xyz' / f'{x}.xyz'), str(S),
                            '--steps', '5', '--rc', rc], capture_output=True, text=True)
        l = [l for l in r.stdout.splitlines() if 'D3COUNT' in l]
        lines.append(f'{x} rc={rc}: ' + (l[0] if l else 'NO OUTPUT ' + r.stdout[-300:] + r.stderr[-300:]))
        print(lines[-1], flush=True)
    open(RES / 'count.txt', 'w').write('\n'.join(lines) + '\n')
