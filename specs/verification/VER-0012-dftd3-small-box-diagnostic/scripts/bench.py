"""nsys kernel-time and gap measurement of one gpumd run (VER-0012).

usage: bench.py BINARY XYZ WORKDIR [--steps 100] [--rc 12] [--cn 6] [--skip 5] [--json OUT]
Runs `potential NEP89; dftd3 pbe RC CN; velocity 300 seed 1; nve; run STEPS` under nsys, exports the sqlite
database, and reports per D3 kernel: median / mean duration [ms] over the steps (first SKIP calls dropped),
and per step: D3 total (sum of kernel durations), span (first start to last end), gap (span - total).
NEP89 path is taken from REPO/potentials (derived from this file's location).
"""
import argparse, json, os, shutil, sqlite3, statistics, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
NSYS = '/usr/local/cuda/bin/nsys'
p = argparse.ArgumentParser()
p.add_argument('binary'); p.add_argument('xyz'); p.add_argument('workdir')
p.add_argument('--steps', type=int, default=100); p.add_argument('--rc', default='12'); p.add_argument('--cn', default='6')
p.add_argument('--skip', type=int, default=5); p.add_argument('--json'); p.add_argument('--keep', action='store_true')
a = p.parse_args()
d = Path(a.workdir) / ('run_' + Path(a.binary).name + '_' + Path(a.xyz).stem)
shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
shutil.copy(a.xyz, d / 'model.xyz')
(d / 'run.in').write_text(f'potential {NEP}\ndftd3 pbe {a.rc} {a.cn}\nvelocity 300 seed 1\nensemble nve\n'
                          f'time_step 0.5\nrun {a.steps}\n')
r = subprocess.run([NSYS, 'profile', '-o', 'prof', '--force-overwrite', 'true', '--export=sqlite', '--stats=false',
                    a.binary], cwd=d, capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'D3COUNT' in line:
        print(line)
db = sqlite3.connect(d / 'prof.sqlite')
rows = db.execute('''select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id = k.demangledName order by k.start''').fetchall()
SMALL = ['find_neighbor_list_small_box', 'find_dftd3_coordination_number_small_box', 'add_dftd3_force_small_box',
         'add_dftd3_force_extra_small_box']
LARGE_KEYS = ['dftd3', 'find_cell_counts', 'find_cell_contents', 'find_cell_count', 'find_cell_']


def short(n):
    return n.split('(')[0].replace('<unnamed>::', '').replace('(anonymous namespace)::', '').split('<')[0]


# the NEP potential also has a kernel named find_neighbor_list_small_box; the D3 one takes DFTD3::ExpandedBox
names = [short(x[0]) if ('DFTD3::ExpandedBox' in x[0] or 'dftd3' in x[0] or x[0].startswith('find_cell_') or 'DeviceScan' in x[0])
         else 'other:' + short(x[0]) for x in rows]
is_small = any(n in SMALL for n in names)
d3 = [(n, s, e) for n, (nm, s, e) in zip(names, rows) if ('dftd3' in n or n in SMALL or
      (not is_small and n.startswith('find_cell') and False))]
# D3 kernels = names containing 'dftd3' (all force/CN/extra kernels) plus the neighbor/cell kernels of D3
# (the small-box list kernel, the large-box cell kernels found between D3 kernels in the same step)
sel = []
for i, (nm, s, e) in enumerate(rows):
    n = names[i]
    if n in SMALL or 'dftd3' in n or (not is_small and (n.startswith('find_cell_') or 'DeviceScan' in rows[i][0])):
        sel.append((n, s, e))
# group into steps: a step starts at each find_neighbor_list_small_box (small box) / first D3 kernel (large)
per = {}
for n, s, e in sel:
    per.setdefault(n, []).append((e - s) / 1e6)
res = {'kernels': {}}
for n, v in per.items():
    v = v[a.skip:] if len(v) > a.skip else v
    res['kernels'][n] = dict(median_ms=statistics.median(v), mean_ms=statistics.mean(v), n=len(v))
if is_small:
    steps = []
    cur = []
    for n, s, e in sel:
        if n == 'find_neighbor_list_small_box' and cur:
            steps.append(cur); cur = []
        cur.append((n, s, e))
    steps.append(cur)
    steps = [st for st in steps if len(st) == 4][a.skip:]
    tot = [sum((e - s) for _, s, e in st) / 1e6 for st in steps]
    span = [(st[-1][2] - st[0][1]) / 1e6 for st in steps]
    res['steps'] = len(steps)
    res['d3_total_ms_median'] = statistics.median(tot)
    res['d3_span_ms_median'] = statistics.median(span)
    res['d3_gap_ms_median'] = statistics.median([sp - t for sp, t in zip(span, tot)])
    # gaps between consecutive kernels of one step (3 per step)
    g = [[(st[i + 1][1] - st[i][2]) / 1e3 for i in range(3)] for st in steps]
    res['gaps_us_median'] = [statistics.median(x[i] for x in g) for i in range(3)]
if not is_small:
    # large box: D3 total = (all D3 force/CN/extra + find_cell_* + cub scans) per step; NEP uses other kernels
    force_n = [v for k, v in res['kernels'].items() if 'force_large_box' in k and 'extra' not in k][0]['n']
    # per-step total from the medians of each kernel times its calls per step
    tot_ms = 0.0
    for k, v in per.items():
        calls_per_step = len(v) / (force_n + a.skip)
        tot_ms += statistics.median(v[a.skip:]) * calls_per_step
    res['d3_total_ms_median'] = tot_ms
    print(f"{Path(a.binary).name:18s} {Path(a.xyz).stem:10s} D3_TOTAL(large box, sum of medians x calls/step)={tot_ms:.4f} ms")
for n, v in res['kernels'].items():
    print(f"{Path(a.binary).name:18s} {Path(a.xyz).stem:10s} {n:42s} median={v['median_ms']:.4f} mean={v['mean_ms']:.4f} ms (n={v['n']})")
if is_small:
    print(f"{Path(a.binary).name:18s} {Path(a.xyz).stem:10s} D3_TOTAL median={res['d3_total_ms_median']:.4f} span={res['d3_span_ms_median']:.4f} gap={res['d3_gap_ms_median']:.4f} ms; gaps_us={['%.1f' % x for x in res['gaps_us_median']]}")
if a.json:
    json.dump(res, open(a.json, 'w'), indent=1)
if not a.keep:
    shutil.rmtree(d)
