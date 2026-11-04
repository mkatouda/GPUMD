"""AC-3 additional runs (water 648): usage analyze_ac3.py RESULTS_DIR NEW_B.json NEW_A.json NEW_L.json OUT_PREFIX
Sets: B10ps = protocol B seeds 1-5 (results/nve_water648.json) + new seeds 6-20; A10ps = protocol A old 3 + new 7;
B20ps = new 20 ps runs.  Quantities: std, |drift|, max-min.  Two-sided Mann-Whitney U (base vs cand), paired Wilcoxon
(B sets only, same seed), bootstrap (10000 resamples, seed 0) 95% interval of median(cand)/median(base).  No thresholds applied."""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
d, nb, na, nl, out = sys.argv[1:6]
ld = lambda f: [r for r in json.load(open(f)) if r['N'] == 648]
sets = {
  'B_10ps_20seeds': ld(f'{d}/nve_water648.json') + ld(nb),
  'A_10ps_10runs': ld(f'{d}/protocolA_same_initial_velocities/nve_water648.json') + ld(na),
  'B_20ps_5seeds': ld(nl),
}
keys = {'std_eV': 'std [eV]', 'abs_drift_meV_per_atom_ps': '|drift| [meV/atom/ps]', 'maxmin_eV': 'max-min [eV]'}
def val(r, k): return abs(r['drift_meV_per_atom_ps']) if k == 'abs_drift_meV_per_atom_ps' else r[k]
rng = np.random.default_rng(0)
res = {}; md = ['| set | quantity | n | base median / worst | cand median / worst | cand/base median [95% bootstrap CI] | MWU p | paired Wilcoxon p |', '|---|---|---|---|---|---|---|---|']
for s, rows in sets.items():
    seeds = sorted({r['seed'] for r in rows})
    for k, lab in keys.items():
        b = np.array([val(next(r for r in rows if r['seed'] == sd and r['tag'] == 'base'), k) for sd in seeds])
        c = np.array([val(next(r for r in rows if r['seed'] == sd and r['tag'] == 'cand'), k) for sd in seeds])
        # A set: seed labels repeat nothing; rows with same seed may exist twice -> handle below
        if s.startswith('A_'):
            b = np.array([val(r, k) for r in rows if r['tag'] == 'base']); c = np.array([val(r, k) for r in rows if r['tag'] == 'cand'])
        ratios = [np.median(rng.choice(c, len(c))) / np.median(rng.choice(b, len(b))) for _ in range(10000)]
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        p = mannwhitneyu(b, c, alternative='two-sided').pvalue
        pw = wilcoxon(b, c).pvalue if s.startswith('B_') else float('nan')
        res[f'{s}/{k}'] = dict(n=len(b), base_median=float(np.median(b)), base_worst=float(b.max()), base_mean=float(b.mean()),
            cand_median=float(np.median(c)), cand_worst=float(c.max()), cand_mean=float(c.mean()),
            ratio_median=float(np.median(c) / np.median(b)), ratio_ci95=[float(lo), float(hi)],
            ratio_mean=float(c.mean() / b.mean()), mwu_p=float(p), wilcoxon_paired_p=None if np.isnan(pw) else float(pw),
            base_values=b.tolist(), cand_values=c.tolist())
        md.append(f'| {s} | {lab} | {len(b)} | {np.median(b):.4f} / {b.max():.4f} | {np.median(c):.4f} / {c.max():.4f} | '
                  f'{np.median(c)/np.median(b):.2f} [{lo:.2f}, {hi:.2f}] | {p:.3f} | {"-" if np.isnan(pw) else f"{pw:.3f}"} |')
json.dump(res, open(out + '.json', 'w'), indent=1)
open(out + '.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
