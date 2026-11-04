# VER-0020: DFT-D3 AC-3 NVE energy conservation on the final D3 code (DR-001)

- Date: 2026-11-04. Status: verification (self-comparison of two binaries). NOT validation.
- Candidate: worktree `d3-nve2`, HEAD `0c353281` (final D3 code, includes the CN list), `make CUDA_ARCH="-arch=sm_86" -j4`, RTX A6000, CUDA 12.3.
  Baseline: `master-base` `3b2c8db8` (before the D3 optimization), sm_86. NEP89 (`nep89_20250409`), `dftd3 pbe 12 6`.
  No file under `src/`, `doc/`, tests was changed.
- Protocol: VER-0011 sections A and D, protocol B (`scripts/nve.py` unchanged from VER-0011): start = NEP-only NVT-BDP 300 K, 0.5 fs, 5 ps
  (candidate binary, one equilibration per system, shared by both binaries); velocities removed from the start so that the seed selects the
  initial velocities (same seed for both binaries); NVE 0.5 fs, `dump_thermo 10`, E_tot = KE + PE, drift = linear fit (meV/atom/ps).
  `analyze_ac3.py`, `summarize_nve.py`, `pooled_stats.py` are copied unchanged (VER-0011); `analyze_ac3.py`/`pooled_stats.py` hard-code the
  VER-0011 file sets, so the tables here were produced by the new `scripts/stats.py`, which computes the SAME statistics as `analyze_ac3.py`
  (medians/worst, cand/base median ratio with 95% bootstrap CI of 10000 resamples with numpy seed 0, two-sided Mann-Whitney U, paired Wilcoxon)
  for arbitrary result files, and additionally evaluates the criterion below. `summarize_nve.py` was used for the per-seed Cu table.
- Criterion (unchanged from VER-0011 D): no quantity significantly worse (cand median larger and MWU or paired Wilcoxon p < 0.05) and
  cand worst <= 2 x base worst. S1 gate (task): median std and |drift| of candidate <= 3 x baseline median, no NaN/crash.

Total NVE runs: 5x2 + 20x2 + 5x2 + 5x2 = 70 (S1 10, S2 40, S3 water seeds 6-10 10 + Cu 10); wall time S1 about 9 min, S2 55.8 min, S3 20.7 min.
No NaN, no crash in any run.

## Results (std of E_tot in eV; |drift| in meV/atom/ps; per-run values in `results/*/nve_*.json`, full table `results/*/stats.md/.json`)
| stage / set | quantity | n | base median / worst | cand median / worst | cand/base median [95% CI] | MWU p | paired Wilcoxon p | worst ratio |
|---|---|---|---|---|---|---|---|---|
| S1 water 5184, 5 ps | std | 5 | 0.0596 / 0.1010 | 0.0726 / 0.0931 | 1.22 [0.65, 2.72] | 0.310 | 0.438 | 0.92 |
| S1 | \|drift\| | 5 | 0.0062 / 0.0128 | 0.0089 / 0.0110 | 1.44 [0.61, 7.54] | 0.310 | 0.312 | 0.86 |
| S1 | max-min | 5 | 0.3848 / 0.4850 | 0.4283 / 0.4747 | 1.11 [0.82, 1.27] | 0.548 | 0.812 | 0.98 |
| S2 water 648, 10 ps | std | 20 | 0.0278 / 0.0662 | 0.0305 / 0.0562 | 1.10 [0.62, 1.52] | 0.756 | 0.985 | 0.85 |
| S2 | \|drift\| | 20 | 0.0120 / 0.0342 | 0.0126 / 0.0279 | 1.05 [0.51, 1.80] | 0.968 | 0.956 | 0.81 |
| S2 | max-min | 20 | 0.1425 / 0.2811 | 0.1456 / 0.2387 | 1.02 [0.71, 1.28] | 0.818 | 0.812 | 0.85 |
| S3 water 5184, 5 ps | std | 10 | 0.0600 / 0.1054 | 0.0763 / 0.0975 | 1.27 [0.77, 2.07] | 0.427 | 0.625 | 0.93 |
| S3 | \|drift\| | 10 | 0.0065 / 0.0128 | 0.0095 / 0.0119 | 1.45 [0.75, 4.27] | 0.385 | 0.232 | 0.93 |
| S3 | max-min | 10 | 0.4623 / 0.5334 | 0.4659 / 0.5313 | 1.01 [0.83, 1.28] | 0.791 | 0.770 | 1.00 |
| S3 Cu 4000, 10 ps | std | 5 | 0.0030 / 0.0031 | 0.0030 / 0.0031 | 1.00 [0.95, 1.05] | 0.841 | 0.125 | 1.00 |
| S3 Cu | \|drift\| (x1e-5) | 5 | 2.06 / 3.53 | 1.92 / 3.55 | 0.94 [0.19, 4.57] | 0.841 | 0.125 | 1.00 |
| S3 Cu | max-min | 5 | 0.0206 / 0.0218 | 0.0207 / 0.0218 | 1.00 [0.91, 1.10] | 1.000 | 0.625 | 1.00 |

Notes: the S3 water 5184 set is seeds 1-5 (the S1 runs, same start and protocol) plus new seeds 6-10, i.e. the S1 runs were reused, not repeated.
For Cu the per-seed values agree to the printed digits for both binaries (the Cu drift is at the noise level, about 2e-5 meV/atom/ps).

## Assessment
- S1 gate met (ratios 1.22 and 1.44 <= 3; no NaN/crash).
- The criterion is met in all sets (no p < 0.05 anywhere, smallest p = 0.125 (Cu Wilcoxon, ties); worst-value ratios 0.81-1.00 <= 2).
- Honest reading: the baseline and candidate are statistically indistinguishable in std, |drift| and max-min for all four systems. For water
  5184 the point estimates of the median std and |drift| ratio are above 1 (1.27, 1.45; same sign as VER-0011 for water 648 B), with 95% intervals
  [0.77, 2.07] and [0.75, 4.27] that contain 1, so a candidate degradation up to about 2x (std) / 4x (drift) is not excluded by 10 runs;
  the cand worst values are lower than the base worst values. For water 648 the ratios are 1.10 / 1.05 / 1.02 with intervals up to 1.52 / 1.80 / 1.28.
  Cu: identical. The VER-0011 scatter of repeated identical runs (factor about 5) is larger than the observed differences.
- This is "indistinguishable at the noise level" (no evidence of being worse; no proof of equality). It is not validation.

## Commands (scratch outside the repository; $S = scratchpad/nve2)
```
cd src && make CUDA_ARCH="-arch=sm_86" -j4
OUTTAG=_S1  nve.py CAND BASE NEP water27.xyz $S/work water5184 10000 1 2 3 4 5
OUTTAG=_S2  nve.py ... water648 20000 $(seq 1 20)
OUTTAG=_S3w nve.py ... water5184 10000 6 7 8 9 10
OUTTAG=_S3c nve.py ... cu4000 20000 1 2 3 4 5
stats.py results/S1/stats water5184_S1=results/S1/nve_water5184_S1.json   (S2, S3 likewise; S3 water file = S1 + S3w concatenated)
```
Generated dumps/restarts/time series are in the scratch directory only; the repository holds per-run JSON and tables.

## Not verified
HIP build, other GPUs, longer than 10 ps, more than 5-20 runs per case, float vs double, no-D3 control (not rerun; see VER-0011).
