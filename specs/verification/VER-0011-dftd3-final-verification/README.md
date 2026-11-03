# VER-0011: DFT-D3 final verification (DR-001 AC-3 NVE energy conservation, AC-6 cutoff-cost measurements)

- Date: 2026-11-03. Status: verification (self-comparison of two binaries, timing). NOT validation.
- Candidate: worktree HEAD `26c64b77` (D3 sources as in `84e5fa83`, VER-0004), `make CUDA_ARCH="-arch=sm_86" -j4`.
  Baseline: `master` `3b2c8db8` (before the D3 optimization), built sm_86. RTX A6000, CUDA 12.3, NEP89 (`nep89_20250409`),
  `dftd3 pbe 12 6`. No file under `src/` was changed.
- Scripts: `scripts/nve.py` (NVE, baseline vs candidate), `scripts/summarize_nve.py`, `scripts/pooled_stats.py`,
  `scripts/timing.py` (slope method of VER-0001, 100 vs 500 steps, min of 2 runs; cutoff scan), `scripts/cn_scan.py`.

## A. AC-3 NVE energy conservation (criterion unchanged: drift no worse than the baseline binary)
Protocol (`scripts/nve.py`): start = structure equilibrated with NEP89 only (NVT-BDP 300 K, 0.5 fs, 10000 steps = 5 ps, candidate binary,
common to both binaries); NVE 0.5 fs, `dump_thermo 10`, E_tot = KE + PE; drift = linear fit; same start and same
velocity seed for both binaries. Systems: water 648 (small-box path, 10 ps), fcc Cu 4000 (a = 3.615 A, 10 ps, large-box path),
water 5184 (large-box path, 5 ps = 10000 steps, as allowed by the task).
The drift unit is meV/atom/ps.

**Finding that changes the protocol.** `gpumd` ignores `velocity ... seed` when `model.xyz` already has velocities
(`Velocity::initialize_cpu`, `src/main_gpumd/velocity.cu`: velocities are generated only if `!has_velocity_in_xyz`).
The first set of runs (protocol A, `results/protocolA_same_initial_velocities/`) kept the velocities of `restart.xyz`, so all seeds
had the identical initial condition and the differences between "seeds" are only GPU run-to-run non-determinism (the
NEP-only runs are bit-identical between binaries and between repeats). The same applies to the VER-0006/VER-0010 `nve.py`
(there the seed was also ineffective; not examined further here). Protocol B (`results/`, `nve.py` default) removes the
velocities from the start, so the seed selects different initial velocities (300 K, from the equilibrated positions).

Protocol B (`results/nve_table.md`, `results/nve_*.json`), per-seed values in the table file. Medians / worst values over seeds
(std of E_tot in eV; |drift| in meV/atom/ps):
| system | binary | runs | std median / worst | max-min worst [eV] | \|drift\| median / worst |
|---|---|---|---|---|---|
| water 648, no D3 (control) | base = cand (bit-identical) | 3 | 0.0331 / 0.0371 | 0.193 | 0.0144 / 0.0152 |
| water 648, D3 | baseline | 5 | 0.0451 / 0.0517 | 0.220 | 0.0216 / 0.0265 |
| water 648, D3 | candidate | 5 | 0.0405 / 0.0551 | 0.212 | 0.0201 / 0.0256 |
| Cu 4000, D3 | baseline | 3 | 0.0031 / 0.0031 | 0.021 | 0.00005 / 0.00005 |
| Cu 4000, D3 | candidate | 3 | 0.0031 / 0.0031 | 0.021 | 0.00005 / 0.00005 |
| water 5184, no D3 (control) | base = cand (bit-identical) | 3 | 0.0444 / 0.0607 | 0.449 | 0.0049 / 0.0069 |
| water 5184, D3 | baseline | 8 | 0.0901 / 0.1071 | 0.559 | 0.0113 / 0.0135 |
| water 5184, D3 | candidate | 8 | 0.0860 / 0.1603 | 0.742 | 0.0105 / 0.0211 |

Protocol A (same initial condition, GPU non-determinism only; `results/protocolA_same_initial_velocities/nve_table.md`, 3 runs each):
| system | binary | std median / worst [eV] | \|drift\| median / worst |
|---|---|---|---|
| water 648 | baseline | 0.0254 / 0.0295 | 0.0035 / 0.0103 |
| water 648 | candidate | 0.0396 / 0.1135 | 0.0119 / 0.0589 |
| Cu 4000 | baseline / candidate | 0.0031 / 0.0031 (both) | 0.00005 (both) |
| water 5184 | baseline | 0.2805 / 0.2838 | 0.0371 / 0.0372 |
| water 5184 | candidate | 0.1555 / 0.1744 | 0.0199 / 0.0229 |
Three runs of the candidate with the identical initial condition (water 5184, seed 1, protocol B, rerun twice):
std 0.160 (original), 0.115, 0.094 eV; drift 0.021, 0.015, 0.012. So the run-to-run scatter of one binary with one initial
condition spans the observed cand/base difference in the water 5184 case (the worst value 0.160 is not reproducible).
Pooled A + B (`scripts/pooled_stats.py`, two-sided Mann-Whitney U, each pair of binaries): water 648 (8 runs each) std median
0.0355 (base) vs 0.0400 (cand), p = 0.38, |drift| median 0.0154 vs 0.0183, p = 0.65; water 5184 (11 each) std median 0.0928
vs 0.0881, p = 0.47, |drift| 0.0117 vs 0.0108, p = 0.39. No difference is statistically distinguishable.

Assessment (not a pass by the letter of "no worse"): Cu 4000: identical. Water 5184: candidate not worse by median or mean, one
worse tail value (0.160 vs 0.107) that a rerun does not reproduce. Water 648: the candidate is lower by median in protocol B and
higher in protocol A (all 3 pairs) and in the pooled mean (0.049 vs 0.034 eV); not significant (p = 0.38) with these few runs.
The D3 runs of both binaries have a larger and more scattered std and drift than the NEP-only control, so part of the D3
energy non-conservation (float arithmetic of NEP/D3, different atomic-add order) is common to baseline and candidate. The
differences between the binaries are within the scatter of repeated runs; "clearly worse than baseline" is not observed, but
"no worse" is shown only at the noise level (3-11 runs per case). Left to the human to accept or ask for more runs.

## B. AC-6 cutoff-cost measurements (`results/timing_cutoff.json`, `results/timing_cn_scan.json`, `results/timing_table.md`)
Slope method of VER-0001 (100 vs 500 steps, min of 2 runs each, whole-process "Time used"), NVE 0.5 fs, `dftd3 pbe RC CN`,
extra = with D3 minus without D3 (ms/step). Water: `tests_pytest/fixtures/dftd2/water27.xyz` repeated (81, 648, 5184, 41472 atoms),
Cu fcc a = 3.615 A 10x10x10. Candidate vs baseline binary:
| system | N | rc / cn | path (cand) | baseline extra | candidate extra |
|---|---|---|---|---|---|
| water | 81 | 12 / 6 | small | 2.57 | 1.55 |
| water | 648 | 12 / 6 | small | 4.33 | 1.95 |
| water | 5184 | 12 / 6 | large | 4.53 | 1.10 |
| water | 41472 | 12 / 6 | large | 8.36 | 7.51 |
| Cu | 4000 | 12 / 6 | large | 3.61 | 0.53 |
| water | 648 | 8 / 6 | small | 1.07 | 0.63 |
| water | 648 | 16 / 6 | small | 6.29 | 3.85 |
| water | 648 | 20 / 6 | small | 15.33 | 7.77 |
| water | 5184 | 8 / 6 | large | 2.20 | 0.56 |
| water | 5184 | 16 / 6 | small (37.2 A <= 2.5 x 16) | 8.39 | 7.24 |
| water | 5184 | 20 / 6 | small | 30.69 | 30.61 |
| water | 648 | 12 / 3, 12 | small | 4.18, 5.32 | 1.80, 2.85 |
| water | 5184 | 12 / 3, 12 | large | 4.24, 7.69 | 0.98, 1.90 |
Time without D3 (candidate binary, ms/step): 81: 1.36, 648: 1.60, 5184: 1.76, 41472: 6.76, Cu 4000: 1.38.
Cubic law check (candidate): water 648: 12 -> 20 A gives x4.0 (cubic: 4.6), 16 -> 20 A x2.0 (cubic 1.95); water 5184: 12 -> 16 A x6.6
(path change) and 16 -> 20 A x4.2 (cubic 1.95); that is steeper than cubic and was not investigated (candidate and baseline are
equal there, 30.6 ms). The doc therefore states the cubic law for the number of neighbors and says the time follows it only roughly.
Not repeated: noise of 0.1-0.5 ms in the small values (single measurement per point). The AC-4 targets were not re-judged here;
for reference the values above give N=81 1.55 ms (target 1.0, not met), N=648 1.95 (2.0, met), N=5184 1.10 (2.0, met),
N=41472 7.51 (6.0, not met), Cu 4000 0.53 (2.0, met), consistent with VER-0004.

## C. Other checks
- `pytest -q -m "not slow" test_dftd3_reference.py` (`tests_pytest/`, conda env `py312-gpumd`): 7 passed.
- Docutils/Sphinx are not installed in this environment; the rst of `doc/gpumd/input_parameters/dftd3.rst` was checked by reading only.

## Commands (paths shortened; scratch directory outside the repository)
```
cd src && make CUDA_ARCH="-arch=sm_86" -j4
nve.py CAND BASE NEP water27.xyz WORK water648 20000 1 2 3 4 5     # also: cu4000 20000 1 2 3; water5184 10000 1 2 3 and 4 5 6 7 8 (OUTTAG=_more)
D3=0 nve.py ... water648 20000 1 2 3 ; D3=0 nve.py ... water5184 10000 1 2 3     # no-D3 controls
KEEP_VEL=1 nve.py ... (protocol A: water648 20000 1 2 3, cu4000 20000 1 2 3, water5184 10000 1 2 3)
timing.py base=BASE,cand=CAND NEP water27.xyz WORK timing_cutoff.json
cn_scan.py base=BASE,cand=CAND NEP water27.xyz WORK timing_cn_scan.json
```
Protocol A was run before `KEEP_VEL` existed (the script then always kept the velocities; equivalent to `KEEP_VEL=1`).
Per-step E_tot time series were not copied into the repository (about 50 files).

## Not verified
HIP build, other GPUs, longer than 10 ps, more than 3-11 runs per case, float vs double. Cutoff scaling beyond 20 A and for
other densities; cause of the steeper-than-cubic growth at N = 5184, rc >= 16 A.

## D. AC-3 additional runs (water 648, small-box path; added after Section A)
- Date: 2026-11-04. Candidate = this worktree (`src/` identical to `26c64b77`; `git diff 26c64b77 ebf85206 -- src` empty), built sm_86;
  baseline = `master-base` `3b2c8db8`. Same NEP89, `dftd3 pbe 12 6`, NVE 0.5 fs, same thermo/fit as Section A. Start structure: the
  SAME `restart.xyz` as Section A (md5 `5445c3eb...`, reused, not re-equilibrated). Scripts: `scripts/nve.py` (unchanged),
  `scripts/analyze_ac3.py`. Results: `results/ac3_additional/` (per-run JSON; no time series; `ac3_stats.json/.md`).
- Sets: (1) protocol B, 10 ps, seeds 1-5 (Section A) + new seeds 6-20 = 20 seeds/binary, same initial velocities for both binaries per seed;
  (2) protocol A (identical initial condition, `KEEP_VEL=1`), 10 ps, Section A 3 runs + 7 new = 10 runs/binary;
  (3) protocol B, 20 ps (40000 steps), seeds 1-5 (5 seeds/binary, as requested, no reduction needed; about 2.5 min per run).
- Statistics: medians/worst over runs; two-sided Mann-Whitney U (base vs cand); paired Wilcoxon signed-rank for protocol B (same seed);
  95% bootstrap interval (10000 resamples, numpy seed 0) of median(cand)/median(base). No new threshold was introduced.

| set | quantity | n | base median / worst | cand median / worst | cand/base median [95% bootstrap CI] | MWU p | paired Wilcoxon p |
|---|---|---|---|---|---|---|---|
| B_10ps_20seeds | std [eV] | 20 | 0.0275 / 0.0612 | 0.0357 / 0.0641 | 1.30 [0.79, 1.75] | 0.490 | 0.189 |
| B_10ps_20seeds | |drift| [meV/atom/ps] | 20 | 0.0105 / 0.0315 | 0.0157 / 0.0327 | 1.50 [0.71, 2.23] | 0.508 | 0.330 |
| B_10ps_20seeds | max-min [eV] | 20 | 0.1255 / 0.2545 | 0.1488 / 0.2585 | 1.19 [0.84, 1.55] | 0.756 | 0.674 |
| A_10ps_10runs | std [eV] | 10 | 0.0452 / 0.0687 | 0.0400 / 0.1135 | 0.89 [0.61, 1.64] | 0.970 | - |
| A_10ps_10runs | |drift| [meV/atom/ps] | 10 | 0.0194 / 0.0349 | 0.0161 / 0.0589 | 0.83 [0.21, 2.45] | 0.910 | - |
| A_10ps_10runs | max-min [eV] | 10 | 0.2079 / 0.2896 | 0.1875 / 0.3734 | 0.90 [0.70, 1.38] | 1.000 | - |
| B_20ps_5seeds | std [eV] | 5 | 0.0935 / 0.1180 | 0.0755 / 0.1334 | 0.81 [0.49, 2.88] | 0.841 | 0.812 |
| B_20ps_5seeds | |drift| [meV/atom/ps] | 5 | 0.0244 / 0.0296 | 0.0185 / 0.0340 | 0.76 [0.44, 7.23] | 0.690 | 0.625 |
| B_20ps_5seeds | max-min [eV] | 5 | 0.3211 / 0.3995 | 0.3200 / 0.5005 | 1.00 [0.58, 2.57] | 1.000 | 1.000 |

Baseline self-scatter (protocol A, 10 runs of one binary with one identical initial condition, i.e. GPU non-determinism only;
these baseline runs are the "baseline repeated" measurement): std min / median / max = 0.0132 / 0.0452 / 0.0687 eV (CV 0.40),
|drift| 0.0013 / 0.0194 / 0.0349 (CV 0.62). Candidate: std 0.0155 / 0.0400 / 0.1135 (CV 0.57), |drift| 0.0020 / 0.0161 / 0.0589 (CV 0.91).
So one binary with one fixed initial condition already varies by a factor of about 5 in std. The earlier 3 + new 7 protocol A runs
(the old 3 had base 0.0254 vs cand 0.0396 median) no longer show candidate worse: median ratio 0.89 (std).

Facts: no quantity in any set is statistically distinguishable between the binaries (all Mann-Whitney and paired Wilcoxon p >= 0.19).
Protocol B 10 ps: cand/base median ratio 1.30 (std), 1.50 (|drift|), 1.19 (max-min), with 95% intervals containing 1 ([0.79, 1.75],
[0.71, 2.23], [0.84, 1.55]); the point estimate is above 1 (candidate larger) for all three, worst values are nearly equal
(0.0612 vs 0.0641 eV). Protocol A: ratios 0.89 / 0.83 / 0.90, intervals contain 1; candidate has the single largest outlier
(std 0.1135, |drift| 0.0589, from Section A). 20 ps: ratios 0.81 / 0.76 / 1.00, intervals wide (n = 5) and contain 1.
Verdict per quantity (std, |drift|, max-min), in all three sets: "equivalent" in the sense of not distinguishable (no significantly worse,
no significantly better). The letter of "no worse than baseline" is shown only at the noise level; the intervals of the
B-10ps ratios extend to 1.75 / 2.23 / 1.55, so a candidate/baseline degradation up to that size is not excluded by these data.
Total 54 new NVE runs (about 2 h on one RTX A6000).

Commands: `nve.py CAND BASE NEP water27.xyz WORK water648 20000 $(seq 6 20)` (`OUTTAG=_s6to20`);
`KEEP_VEL=1 OUTTAG=_A4to10 nve.py ... water648 20000 $(seq 4 10)`; `OUTTAG=_20ps nve.py ... water648 40000 1 2 3 4 5`;
`analyze_ac3.py results <B.json> <A.json> <20ps.json> results/ac3_additional/ac3_stats`. Each of the three used its own copy of the shared
equilibrated start.
