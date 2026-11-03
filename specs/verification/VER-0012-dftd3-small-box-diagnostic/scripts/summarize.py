"""Build results/summary.md tables from results/*.json (VER-0012).  usage: summarize.py"""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'results'
scan = json.load(open(R / 'scan_base.json')) + json.load(open(R / 'scan_big.json'))
ref = scan[-1]
thr = ref['force'] / ref['N']  # ms per atom at the largest N (N = %d)
L = []
L.append(f"(a) small-box force kernel, water, rc 12/6, NVE; medians of 3 runs. Large-N throughput = {ref['force']:.3f} ms / {ref['N']} atoms = {thr * 1e3:.4f} us/atom")
L.append('')
L.append('| N | force [ms] | expected from throughput [ms] | measured / expected | NL [ms] | CN [ms] | extra [ms] | D3 total [ms] | gap [ms] |')
L.append('|---|---|---|---|---|---|---|---|---|')
for r in scan:
    e = thr * r['N']
    L.append(f"| {r['N']} | {r['force']:.3f} | {e:.4f} | {r['force'] / e:.1f} | {r['nl']:.3f} | {r['cn']:.3f} | {r['extra']:.3f} | {r['total']:.3f} | {r['gap']:.4f} |")
L.append('')
L.append('(d)(e) small-box variants (timing only), medians of 3 alternated runs; speedup = base / variant')
L.append('')
L.append('| system | variant | force base -> var [ms] | force x | NL x | D3 total base -> var [ms] | total x |')
L.append('|---|---|---|---|---|---|---|')
for r in json.load(open(R / 'variants.json')):
    L.append(f"| {r['system']} | {r['variant']} | {r['force_base']:.3f} -> {r['force_var']:.3f} | {r['force_speedup']:.2f} | {r['nl_speedup']:.2f} | {r['total_base']:.3f} -> {r['total_var']:.3f} | {r['total_speedup']:.2f} |")
L.append('')
L.append('(d)(e) large-box force kernel variants (timing only)')
L.append('')
L.append('| system | variant | force base -> var [ms] | force x | D3 total base -> var [ms] | total x |')
L.append('|---|---|---|---|---|---|')
for r in json.load(open(R / 'variants_large.json')):
    L.append(f"| {r['system']} | {r['variant']} | {r['force_base']:.3f} -> {r['force_var']:.3f} | {r['force_speedup']:.2f} | {r['total_base']:.3f} -> {r['total_var']:.3f} | {r['total_speedup']:.2f} |")
L.append('')
L.append('(f) empty-kernel sequence (all four small-box D3 kernels return immediately), nsys, 3 runs')
L.append('')
L.append('| N | sum of 4 empty kernels [us] | gaps (3) [us] | span [us] | base D3 total [ms] | span / base total |')
L.append('|---|---|---|---|---|---|')
base = {r['system']: r for r in scan}
emp = json.load(open(R / 'empty.json'))
for x in ['w81', 'w648']:
    rs = [e for e in emp if e['system'] == x]
    tot = sorted(e['total'] for e in rs)[1] * 1e3
    gap = sorted(e['gap'] for e in rs)[1] * 1e3
    span = sorted(e['span'] for e in rs)[1] * 1e3
    b = base[x]['total']
    L.append(f"| {x[1:]} | {tot:.2f} | {gap:.2f} | {span:.2f} | {b:.3f} | {span / 1e3 / b * 100:.2f} % |")
(R / 'summary.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
