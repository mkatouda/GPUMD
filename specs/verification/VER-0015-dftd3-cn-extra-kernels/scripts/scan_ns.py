"""Threads-per-atom scan of the large-box CN and extra kernels (VER-0015 6a).  Uses the `nsenv` variant binary
(env D3_NS_CN / D3_NS_EX).  usage: scan_ns.py SCRATCH OUT.json system[,system...]   (2 repetitions, median)"""
import json, statistics, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = Path(sys.argv[1]); OUT = sys.argv[2]
res = {}
for x in sys.argv[3].split(','):
    res[x] = {}
    for ns in (1, 2, 4, 8, 16):
        cn, ex = [], []
        for rep in range(2):
            tmp = S / 'tmp_ns.json'
            tmp.unlink(missing_ok=True)
            subprocess.run([sys.executable, str(HERE / 'bench6a.py'), str(S / 'bin/gpumd_nsenv'), str(S / 'xyz' / f'{x}.xyz'), str(S),
                            '--json', str(tmp), '--steps', '60', '--env', f'D3_NS_CN={ns}', f'D3_NS_EX={ns}'],
                           capture_output=True, text=True, timeout=300)
            d = json.load(open(tmp))
            for k in d['kernels']:
                if k['name'].startswith('find_dftd3_coordination'): cn.append(k['median_ms'])
                if k['name'].startswith('find_dftd3_force_extra'): ex.append(k['median_ms'])
        res[x][ns] = dict(cn=statistics.median(cn), extra=statistics.median(ex))
        print(x, ns, res[x][ns], flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
