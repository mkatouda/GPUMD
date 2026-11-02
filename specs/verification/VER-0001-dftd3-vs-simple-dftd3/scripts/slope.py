import os, re, shutil, subprocess, json
from ase.io import read, write
GPUMD = '/home/katouda/work/dielectric-response_pj/GPUMD/src/gpumd'
NEP = '/home/katouda/work/dielectric-response_pj/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt'
base = read('water27.xyz'); out = []
def tm(a, d3, steps, d):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d); write(f'{d}/model.xyz', a)
    open(f'{d}/run.in','w').write(f'potential {NEP}\n' + (f'dftd3 pbe {d3[0]} {d3[1]}\n' if d3 else '') + f'velocity 300\nensemble nve\ntime_step 0.5\nrun {steps}\n')
    o = subprocess.run([GPUMD], cwd=d, capture_output=True, text=True).stdout
    return float(re.search(r'Time used = ([0-9.]+) s', o).group(1))
for rep, cs in [(1,(12,6)), (4,(12,6)), (8,(12,6)), (4,(25,15))]:
    a = base.repeat((rep,)*3); n = len(a); r = dict(natoms=n, rc=f'{cs[0]}/{cs[1]}')
    for tag, d3 in (('noD3', None), ('D3', cs)):
        t1 = tm(a, d3, 100, f's_{n}_{tag}_1'); t2 = tm(a, d3, 500, f's_{n}_{tag}_2')
        r[f'{tag}_ms_per_step'] = 1e3*(t2-t1)/400; r[f'{tag}_fixed_s'] = t1 - 100*(t2-t1)/400
    r['D3_extra_ms_per_step'] = r['D3_ms_per_step'] - r['noD3_ms_per_step']
    r['D3_extra_init_s'] = r['D3_fixed_s'] - r['noD3_fixed_s']
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open('slope.json','w'), indent=1)
