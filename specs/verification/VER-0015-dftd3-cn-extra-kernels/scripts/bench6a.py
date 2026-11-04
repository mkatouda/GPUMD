"""Per-kernel nsys timing of the DFT-D3 LARGE-box step (VER-0015).

usage: bench6a.py BINARY XYZ WORKDIR [--steps 100] [--skip 5] [--rc 12] [--cn 6] [--json OUT] [--env K=V ...]
Runs `potential NEP89; dftd3 pbe RC CN; velocity 300 seed 1; nve; run STEPS` under nsys (sqlite export).
A D3 step = the cell-list kernels (find_cell_*, cub scan) directly preceding the CN kernel, the CN kernel, the
force kernel and the extra kernel.  Reports per kernel (in launch order within the step): median duration [ms]
over the steps (first SKIP dropped), grid, block, registers, local memory per thread (spill/stack indicator),
and the D3 total per step (sum of the kernel durations, median over steps; VER-0004 section 3 definition).
"""
import argparse, json, os, shutil, sqlite3, statistics, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
NSYS = '/usr/local/cuda/bin/nsys'


def short(n):
    n = n.replace('void ', '').replace('<unnamed>::', '').replace('(anonymous namespace)::', '')
    return n.split('(')[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary'); p.add_argument('xyz'); p.add_argument('workdir')
    p.add_argument('--steps', type=int, default=100); p.add_argument('--rc', default='12'); p.add_argument('--cn', default='6')
    p.add_argument('--skip', type=int, default=5); p.add_argument('--json'); p.add_argument('--keep', action='store_true')
    p.add_argument('--env', nargs='*', default=[])
    a = p.parse_args()
    d = Path(a.workdir) / ('run_' + Path(a.binary).name + '_' + Path(a.xyz).stem)
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    shutil.copy(a.xyz, d / 'model.xyz')
    (d / 'run.in').write_text(f'potential {NEP}\ndftd3 pbe {a.rc} {a.cn}\nvelocity 300 seed 1\nensemble nve\n'
                              f'time_step 0.5\nrun {a.steps}\n')
    env = dict(os.environ)
    for kv in a.env:
        k, v = kv.split('=', 1); env[k] = v
    r = subprocess.run([NSYS, 'profile', '-o', 'prof', '--force-overwrite', 'true', '--export=sqlite', '--stats=false',
                        a.binary], cwd=d, capture_output=True, text=True, env=env)
    extra_out = [l for l in r.stdout.splitlines() if l.startswith('D3')]
    db = sqlite3.connect(d / 'prof.sqlite')
    rows = db.execute('''select s.value, k.start, k.end, k.gridX, k.blockX, k.registersPerThread, k.localMemoryPerThread
                         from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName
                         order by k.start''').fetchall()
    names = [short(r_[0]) for r_ in rows]
    is_cell = lambda n: n.startswith('find_cell_') or 'DeviceScan' in n
    cn_idx = [i for i, n in enumerate(names) if n.startswith('find_dftd3_coordination_number_large_box')]
    steps = []
    for i in cn_idx:
        j = i
        while j > 0 and (is_cell(names[j - 1]) or 'DeviceScan' in rows[j - 1][0]):
            j -= 1
        if i + 2 >= len(rows):
            continue
        win = list(range(j, i + 3))
        steps.append(win)
    steps = steps[a.skip:]
    n_k = max(len(w) for w in steps)
    steps = [w for w in steps if len(w) == n_k]
    res = {'steps': len(steps), 'kernels': [], 'extra_out': extra_out}
    for pos in range(n_k):
        dur = [(rows[w[pos]][2] - rows[w[pos]][1]) / 1e6 for w in steps]
        r0 = rows[steps[0][pos]]
        res['kernels'].append(dict(name=names[steps[0][pos]], full=r0[0][:200], median_ms=statistics.median(dur),
                                   min_ms=min(dur), max_ms=max(dur), grid=r0[3], block=r0[4], regs=r0[5], local=r0[6]))
    tot = [sum((rows[i][2] - rows[i][1]) for i in w) / 1e6 for w in steps]
    res['d3_total_ms_median'] = statistics.median(tot)
    res['d3_total_ms_min'] = min(tot); res['d3_total_ms_max'] = max(tot)
    for k in res['kernels']:
        print(f"{Path(a.xyz).stem:8s} {k['name'][:44]:44s} {k['median_ms']:8.4f} ms grid={k['grid']:6d} block={k['block']:4d} regs={k['regs']:3d} local={k['local']}")
    print(f"{Path(a.xyz).stem:8s} D3_TOTAL median={res['d3_total_ms_median']:.4f} ms (min {res['d3_total_ms_min']:.4f}, max {res['d3_total_ms_max']:.4f}); {extra_out}")
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)
    if not a.keep:
        shutil.rmtree(d)


main()
