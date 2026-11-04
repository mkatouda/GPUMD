"""Stage 6a driver: kernels of each timing variant on each system, variants alternated REPS times
(median of the per-run kernel medians).  usage: run_6a.py SCRATCH OUT.json [REPS] [--systems a,b] [--variants a,b]
Reads SCRATCH/bin/gpumd_<variant>, SCRATCH/xyz/<system>.xyz; uses bench6a.py."""
import json, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = Path(sys.argv[1]); OUT = sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 3
opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if a.startswith('--')}
systems = opt.get('--systems', 'w41472,w5184,cu4000,clus972').split(',')
variants = opt.get('--variants', 'base,cn_noexp,cn_const,ex_noexp,ex_const,cn_nomic,cn_xf32,ex_xf32,fused').split(',')
steps = opt.get('--steps', '100')
env = opt.get('--env', '').split(',') if opt.get('--env') else []
KEY = {'cn': 'find_dftd3_coordination_number_large_box', 'force': 'find_dftd3_force_large_box',
       'extra': 'find_dftd3_force_extra_large_box'}
res = {}
for x in systems:
    acc = {v: {k: [] for k in KEY} for v in variants}
    tot = {v: [] for v in variants}
    for r in range(reps):
        for v in variants:
            tmp = S / 'tmp6a.json'
            tmp.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, str(HERE / 'bench6a.py'), str(S / 'bin' / f'gpumd_{v}'), str(S / 'xyz' / f'{x}.xyz'),
                                str(S), '--json', str(tmp), '--steps', steps, '--skip', '5', '--env'] + env,
                               capture_output=True, text=True, timeout=240)
                d = json.load(open(tmp))
            except Exception as e:  # hang / crash of a (result-changing) timing variant
                print('FAILED', x, v, repr(e)[:80], flush=True)
                subprocess.run(['pkill', '-9', '-f', f'gpumd_{v}'])
                continue
            for k in d['kernels']:
                for kk, nm in KEY.items():
                    if k['name'].startswith(nm):
                        acc[v][kk].append(k['median_ms'])
            tot[v].append(d['d3_total_ms_median'])
    if not all(acc[v]['cn'] for v in variants):
        print('incomplete', x, flush=True)
    res[x] = {v: dict({k: statistics.median(a) if a else float('nan') for k, a in acc[v].items()}, total=statistics.median(tot[v]) if tot[v] else float('nan'),
                      runs={k: acc[v][k] for k in KEY}) for v in variants}
    b = res[x][variants[0]]
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"{x:8s} " + '  '.join(f"{v}: cn {res[x][v]['cn']:.4f} fo {res[x][v]['force']:.4f} ex {res[x][v]['extra']:.4f} tot {res[x][v]['total']:.4f}"
                               for v in variants), flush=True)

