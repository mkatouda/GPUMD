"""VER-0020 statistics (same statistics as VER-0011 section D, generalized): usage stats.py OUT_PREFIX LABEL=JSON [LABEL=JSON ...]
Each JSON is an nve.py output list (base/cand rows, same seeds).  Per quantity: n, medians, worst, cand/base median ratio with
95% bootstrap interval (10000 resamples, numpy seed 0), two-sided Mann-Whitney U, paired Wilcoxon (same seed).
Also the VER-0011 criterion check: no quantity with p<0.05 worse (cand larger) and cand worst <= 2 x base worst."""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
out = sys.argv[1]
keys = {'std_eV': 'std [eV]', 'abs_drift': '|drift| [meV/atom/ps]', 'maxmin_eV': 'max-min [eV]'}
val = lambda r, k: abs(r['drift_meV_per_atom_ps']) if k == 'abs_drift' else r[k]
rng = np.random.default_rng(0)
res = {}
md = ['| set | quantity | n | base median / worst | cand median / worst | cand/base median [95% CI] | MWU p | paired Wilcoxon p | worst ratio | criterion |', '|---|---|---|---|---|---|---|---|---|---|']
for a in sys.argv[2:]:
    lab, f = a.split('=')
    rows = json.load(open(f))
    seeds = sorted({r['seed'] for r in rows})
    for k, ql in keys.items():
        b = np.array([val(next(r for r in rows if r['seed'] == s and r['tag'] == 'base'), k) for s in seeds])
        c = np.array([val(next(r for r in rows if r['seed'] == s and r['tag'] == 'cand'), k) for s in seeds])
        rat = [np.median(rng.choice(c, len(c))) / np.median(rng.choice(b, len(b))) for _ in range(10000)]
        lo, hi = np.percentile(rat, [2.5, 97.5])
        pm = float(mannwhitneyu(b, c, alternative='two-sided').pvalue)
        try: pw = float(wilcoxon(b, c).pvalue)
        except ValueError: pw = float('nan')
        wr = float(c.max() / b.max())
        worse_sig = (np.median(c) > np.median(b)) and (pm < 0.05 or pw < 0.05)
        ok = (not worse_sig) and wr <= 2.0
        res[f'{lab}/{k}'] = dict(n=len(b), base_median=float(np.median(b)), base_worst=float(b.max()), cand_median=float(np.median(c)),
            cand_worst=float(c.max()), ratio_median=float(np.median(c) / np.median(b)), ratio_ci95=[float(lo), float(hi)],
            mwu_p=pm, wilcoxon_p=pw, worst_ratio=wr, criterion_met=bool(ok), base_values=b.tolist(), cand_values=c.tolist())
        md.append(f'| {lab} | {ql} | {len(b)} | {np.median(b):.4f} / {b.max():.4f} | {np.median(c):.4f} / {c.max():.4f} | '
                  f'{np.median(c)/np.median(b):.2f} [{lo:.2f}, {hi:.2f}] | {pm:.3f} | {pw:.3f} | {wr:.2f} | {"met" if ok else "NOT met"} |')
json.dump(res, open(out + '.json', 'w'), indent=1)
open(out + '.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
