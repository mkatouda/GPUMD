"""Pooled NVE statistics (protocol A + B) per system: usage pooled_stats.py RESULTS_DIR (needs scipy for the p-value)."""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu
d = sys.argv[1]
for name in ('water648', 'water5184'):
    rows = [r for f in (f'{d}/nve_{name}.json', f'{d}/protocolA_same_initial_velocities/nve_{name}.json')
            for r in json.load(open(f)) if r['system'] == name]
    for key in ('std_eV', 'drift_meV_per_atom_ps'):
        v = {t: np.array([abs(r[key]) for r in rows if r['tag'] == t]) for t in ('base', 'cand')}
        for t in v:
            print(name, key, t, len(v[t]), 'median %.4f mean %.4f max %.4f' % (np.median(v[t]), v[t].mean(), v[t].max()))
        print(' Mann-Whitney U two-sided p = %.2f' % mannwhitneyu(v['base'], v['cand']).pvalue)
