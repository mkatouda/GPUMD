"""S1 / tuning scan: force and extra kernel time of the small-box path versus threads per atom (env
GPUMD_DFTD3_SMALL_BOX_NUM_SUB = 1..128; 0 = default rule) for several N (water, VER-0012 systems), nsys, 40 steps,
first 5 calls dropped, median.  usage: scan_num_sub.py SCRATCH  (SCRATCH/bin/gpumd_new, SCRATCH/xyz).
Writes results/num_sub_scan.json."""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / 'VER-0012-dftd3-small-box-diagnostic/scripts/bench.py'
S = Path(sys.argv[1])
rows = []
for x in ['w81', 'scan_162', 'w648', 'scan_2187', 'scan_4374', 'scan_8748', 'scan_17496', 'scan_34992']:
    for ns in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
        env = dict(os.environ)
        env.pop('GPUMD_DFTD3_SMALL_BOX_NUM_SUB', None)
        if ns:
            env['GPUMD_DFTD3_SMALL_BOX_NUM_SUB'] = str(ns)
        subprocess.run([sys.executable, str(BENCH), str(S / 'bin/gpumd_new'), str(S / 'xyz' / f'{x}.xyz'), str(S),
                        '--steps', '40', '--json', str(S / 'scan.json')], env=env, capture_output=True)
        r = json.load(open(S / 'scan.json'))
        k = {n.split('(')[0].split('<')[0].replace('void ', ''): v['median_ms'] for n, v in r['kernels'].items()}
        rows.append(dict(system=x, num_sub=ns, force=k['add_dftd3_force_small_box'], extra=k['add_dftd3_force_extra_small_box'],
                         nl=k['find_neighbor_list_small_box'], cn=k['find_dftd3_coordination_number_small_box'],
                         total=r['d3_total_ms_median']))
        print(rows[-1], flush=True)
json.dump(rows, open(HERE.parent / 'results/num_sub_scan.json', 'w'), indent=1)
