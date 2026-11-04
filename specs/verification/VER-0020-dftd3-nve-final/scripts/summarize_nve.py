"""Markdown table of NVE results: usage summarize_nve.py nve_*.json"""
import json, sys
import numpy as np
rows = [r for f in sys.argv[1:] for r in json.load(open(f))]
print('| system | seed | base std [eV] | cand std [eV] | base drift [meV/atom/ps] | cand drift | base max-min [eV] | cand max-min |')
print('|---|---|---|---|---|---|---|---|')
for s in dict.fromkeys(r['system'] for r in rows):
    sel = [r for r in rows if r['system'] == s]
    for sd in dict.fromkeys(r['seed'] for r in sel):
        b = next(r for r in sel if r['seed'] == sd and r['tag'] == 'base'); c = next(r for r in sel if r['seed'] == sd and r['tag'] == 'cand')
        print(f"| {s} (N={b['N']}, {b['ps']:.1f} ps) | {sd} | {b['std_eV']:.4f} | {c['std_eV']:.4f} | {b['drift_meV_per_atom_ps']:+.4f} | {c['drift_meV_per_atom_ps']:+.4f} | {b['maxmin_eV']:.3f} | {c['maxmin_eV']:.3f} |")
    for tag in ('base', 'cand'):
        t = [r for r in sel if r['tag'] == tag]
        sd = [r['std_eV'] for r in t]; dr = [abs(r['drift_meV_per_atom_ps']) for r in t]
        print(f"| {s} {tag} | median / worst | std {np.median(sd):.4f} / {max(sd):.4f} | | |drift| {np.median(dr):.4f} / {max(dr):.4f} | | |")
