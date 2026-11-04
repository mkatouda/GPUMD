"""S2 measurement: D3 total (VER-0004 section 3 definition, see VER-0012 scripts/bench.py: nsys kernel durations,
neighbor list + CN + force + extra kernels, median over 100 steps without the first 5), base and candidate binary
alternated 3 times (base first), median of the 3 medians.
usage: run_nsys_total_s3c.py SCRATCH   (SCRATCH/bin/gpumd_base, SCRATCH/bin/gpumd_new, SCRATCH/xyz/*.xyz from the two make_systems.py)
Writes results/nsys_total.json and prints a table."""
import json, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
BENCH = HERE.parents[1] / 'VER-0012-dftd3-small-box-diagnostic/scripts/bench.py'
S = Path(sys.argv[1])
SYSTEMS = ['w5184', 'w41472']


def bench(binary, x):
    out = S / 'tmp.json'
    subprocess.run([sys.executable, str(BENCH), str(S / 'bin' / f'gpumd_{binary}'), str(S / 'xyz' / f'{x}.xyz'), str(S),
                    '--json', str(out)], capture_output=True, text=True)
    return json.load(open(out))


rows = []
for x in SYSTEMS:
    tot = {'base': [], 'new': []}
    kern = {'base': {}, 'new': {}}
    for rep in range(3):
        for b in ('base', 'new'):
            r = bench(b, x)
            tot[b].append(r['d3_total_ms_median'])
            for k, v in r['kernels'].items():
                kern[b].setdefault(k.split('(')[0].split('<')[0].replace('void ', ''), []).append(v['median_ms'])
    row = dict(system=x, base_ms=statistics.median(tot['base']), new_ms=statistics.median(tot['new']),
               base_runs=tot['base'], new_runs=tot['new'],
               kernels={b: {k: statistics.median(v) for k, v in kern[b].items()} for b in kern})
    row['speedup'] = row['base_ms'] / row['new_ms']
    rows.append(row)
    print(f"{x:10s} base {row['base_ms']:.3f} ms  new {row['new_ms']:.3f} ms  x{row['speedup']:.2f}  "
          f"kernels new {({k: round(v, 3) for k, v in row['kernels']['new'].items()})}", flush=True)
json.dump(rows, open(RES / 'nsys_total_s3c.json', 'w'), indent=1)
