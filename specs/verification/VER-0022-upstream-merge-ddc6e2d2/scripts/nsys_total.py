"""S6 nsys check: sum of D3 or D2 kernel durations per step (median over 100 steps minus first 5), base (premerge) vs new, alternated.
usage: nsys_total.py SCRATCH OUT.json REPS ; kernels with 'dftd3' / 'dftd2' (case-insens.) in name; D3 additionally neighbor/CN list kernels
are included via bench.py definition for D3 (VER-0012). Here D3 uses bench.py d3_total_ms_median; D2 uses name filter."""
import json, shutil, sqlite3, statistics, subprocess, sys
from pathlib import Path
S = Path(sys.argv[1]); REPS = int(sys.argv[3])
REPO = Path(__file__).resolve().parents[4]
NEP = REPO / 'potentials/nep/nep89_20250409/nep89_20250409.txt'
BENCH = REPO / 'specs/verification/VER-0012-dftd3-small-box-diagnostic/scripts/bench.py'
NSYS = '/usr/local/cuda/bin/nsys'
def d3(b, x):
    out = S / 'tmp.json'
    subprocess.run([sys.executable, str(BENCH), str(S / 'bin' / f'gpumd_{b}'), str(S / 'xyz' / f'{x}.xyz'), str(S), '--json', str(out)], capture_output=True, text=True)
    return json.load(open(out))['d3_total_ms_median']
def d2(b, x):
    d = S / 'd2run'; shutil.rmtree(d, ignore_errors=True); d.mkdir()
    shutil.copy(S / 'xyz' / f'{x}.xyz', d / 'model.xyz')
    (d / 'run.in').write_text(f'potential {NEP}\ndftd2 0.75 15\nvelocity 300 seed 1\nensemble nve\ntime_step 0.5\nrun 100\n')
    subprocess.run([NSYS, 'profile', '-o', 'prof', '--force-overwrite', 'true', '--export=sqlite', '--stats=false', str(S / 'bin' / f'gpumd_{b}')], cwd=d, capture_output=True, text=True)
    c = sqlite3.connect(d / 'prof.sqlite')
    rows = c.execute("select s.value, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
    ks = [(s, e - st) for s, st, e in rows if 'dftd2' in s.lower()]
    c.close()
    # group into steps: kernels per step = len(ks)/101 (101 force evaluations incl. initial)
    n = len(ks) // 101; per = [sum(x[1] for x in ks[i * n:(i + 1) * n]) / 1e6 for i in range(101)]
    return statistics.median(per[5:]), n
out = []
for x in ('w648', 'w5184'):
    for name, f in (('D3', d3), ('D2', d2)):
        r = {'base': [], 'new': []}
        for _ in range(REPS):
            for b in ('base', 'new'):
                v = f(b, x); r[b].append(v[0] if isinstance(v, tuple) else v)
        mb, mn = statistics.median(r['base']), statistics.median(r['new'])
        out.append(dict(system=x, model=name, base_ms=mb, new_ms=mn, ratio=mn / mb, runs=r))
        print(f"{x} {name}: base {mb:.4f} new {mn:.4f} ms ratio {mn/mb:.3f} runs {[round(v,3) for v in r['base']]} {[round(v,3) for v in r['new']]}", flush=True)
        json.dump(out, open(sys.argv[2], 'w'), indent=1)
