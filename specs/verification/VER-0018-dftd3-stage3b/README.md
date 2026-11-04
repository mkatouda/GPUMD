# VER-0018: DFT-D3 Phase 7 Stage 3 (b): AC-4 re-measurement, `--suite full`, quick pytest

- Date: 2026-11-04. Status: verification / regression (NOT validation). Requirement / design: `DR-001` AC-1, AC-2, AC-4; `DES-0001` Phase 7 Stage 3.
- Code: worktree `d3-stage3`, branch `wt-d3-stage3`, HEAD `0ac42758` (final D3 code, no `src/` change in this task). Candidate = `src/gpumd` built from it
  (`make CUDA_ARCH="-arch=sm_86" -j4`, nothing to rebuild). Baseline = `master-base` binaries (`.claude/worktrees/master-base/src/{gpumd,nep}`).
- Environment: RTX A6000 (sm_86), CUDA 12.3, NEP89, one GPU job at a time.
- Stage S1 (extreme CN) is VER-0017; the human decided that the upstream float-underflow deviation at extreme CN is out of scope.
- Scripts (`scripts/`): `slope.py` (VER-0001 slope method, base/new alternated, 3 repetitions, median), `run_nsys_total.py` (copy of VER-0015
  `run_s2.py`: nsys D3 total per step, uses VER-0012 `bench.py`), `compare_ac2.py` (copy of VER-0015). Systems from VER-0012/0013 `make_systems.py`
  (water27 repeats; `scan_2187` = 3x3x3, `scan_8748` = 12x3x3 repeats, small-box path; `w5184`, `w41472`, `cu4000` fcc Cu a = 3.61 A, `clus972` non-periodic).

## S2: timing (AC-4)
Slope method: time per step = (t500 - t100) / 400 from the `Time used` line, D3 extra = slope with `dftd3 pbe 12 6` minus slope without; for each system base and new
alternated, 3 repetitions, median (`results/slope_main.json`, `.log`). nsys D3 total = sum of D3 kernel durations of one step (VER-0004 section 3), median of 100 steps per
run, 3 alternated runs, median (`results/nsys_total.json`). All in ms.

| system (path) | slope: master base | slope: new | AC-4 target | slope status | nsys D3 total: master base | nsys D3 total: new |
|---|---|---|---|---|---|---|
| water 81 (small) | 2.456 | 0.152 | <= 1.0 | met | 2.630 | 0.097 |
| water 648 (small) | 4.417 | 0.459 | <= 2.0 | met | 4.247 | 0.504 |
| water 2,187 (small) | 3.266 | 0.750 | - | - | 3.183 | 0.729 |
| water 5,184 (large) | 4.568 | 0.846 | <= 2.0 | met | 3.791 | 0.738 |
| water 8,748 (small) | 11.782 | 8.274 | - | - | 11.461 | 8.187 |
| water 41,472 (large) | 8.398 | **6.193** | <= 6.0 | **NOT met (slope)** | 8.112 | 5.875 (met on this definition) |
| Cu 4,000 (large) | 3.612 | 0.554 | <= 2.0 | met | 3.577 | 0.507 |
| cluster 972 (large, non-periodic) | 1.717 | 1.216 | - | - | 1.832 | 1.199 |

- Slowdown against master base (new / base, both definitions): no case above 1.0 (largest new/base ratio by slope 0.74 at water 41,472; by nsys 0.72 at water 41,472), i.e. no case is slower
  than the baseline, the 3 % limit is met everywhere.
- AC-4 is defined with the slope method. At water 41,472 the slope result 6.19 ms (runs 6.175, 6.256, 6.193) is above 6.0 ms; the nsys kernel total (5.875 ms) is below it.
  The slope extra contains more than the D3 kernels (the extra step cost of the D3 energy/force/virial additions to the NEP arrays and the host side). On the definition
  written in AC-4 this target is therefore NOT met; the human decides whether to drop it or to redefine it.
- Slope scatter of the three repetitions (new): water 81 0.03-0.21 ms, water 648 0.36-0.53, water 2,187 0.64-0.78, others <= 0.1 ms.
- Small-box versus large-box path costs as measured: small box 0.15 / 0.46 / 0.75 / 8.3 ms at N = 81 / 648 / 2,187 / 8,748 (the explicit neighbor-list build grows roughly as
  N^2, VER-0012), large box 0.85 / 6.2 ms at 5,184 / 41,472.

Cutoff dependence (slope method, base / new, `results/slope_cut.json`; CN cutoff 6 A unless noted):

| system | rc 8 | rc 12 (CN 3 / 6 / 12) | rc 16 | rc 20 |
|---|---|---|---|---|
| water 648 (small) base | 1.050 | 4.138 / 4.417 / 5.345 | 6.292 | 15.312 |
| water 648 (small) new | 0.109 | 0.436 / 0.459 / 0.868 | 0.547 | 1.573 |
| water 5,184 base | 2.191 (large) | 4.150 / 4.568 / 7.660 (large) | 8.432 (small) | 30.804 (small) |
| water 5,184 new | 0.446 (large) | 0.832 / 0.846 / 1.364 (large) | 3.316 (small) | 22.995 (small) |

Documentation: `doc/gpumd/input_parameters/dftd3.rst` (Cost section) tables refreshed with these new-binary slope values; rows for 2,187, 8,748 atoms and the cluster added, Cu
lattice constant corrected to 3.61 A (the one used), method text (median of three repetitions) and the small-box scaling statement updated; the former statement of
"1.5 / 1.9 ms" at N = 81 / 648 was stale (measured on an earlier code state).

## S3: `run_regression.py --suite full`
Command: `run_regression.py --repo-root <worktree> --baseline <master-base gpumd> --candidate <worktree src/gpumd> --baseline-nep <master-base nep> --candidate-nep <worktree src/nep> --suite full`
(conda python; the system python has no numpy). Result: 216 passed, 41 failed, relations 25/25 passed (`results/regression_full.log`). `tests_regression/.work` and `reports` deleted afterwards.

- 37 failures are exactly the 37 known non-deterministic cases of AGENTS.md (41 failures minus the 4 below).
  Failing known cases: nvt_bdp, npt_scr_iso, heat_bdp, pimd_scr, deform_npt_tri, replicate_default_velocity, replicate_velocity_unseeded, graphene_hac, graphene_nemd_shc, graphene_hnemd_shc,
  silicon_liquid_transport, ilp_{nep,tersoff,tmd_sw}_compute and the `train_*` cases. Magnitudes: these are run-to-run differences of non-deterministic cases (stochastic thermostat, unseeded random
  velocities). Not all are small: `replicate_*_velocity` use unseeded random velocities and differ completely (`compare_ac2.py` listing: thermo abs 0.15-0.17, restart.xyz abs 14),
  `nvt_bdp` thermo rel 3e-3, `pimd_scr` rel 3e-2. These cases do not use `dftd3`.
- Failures outside the known list (4):
  - `dftd3_single_late`, `dftd3_kspace_one_each`: strict manifest tolerance only (restart.xyz rel 2.2e-12, text diff), as in VER-0015. Under the AC-2 rule (rtol 1e-5, atol 1e-12,
    `compare_ac2.py`, `results/ac2_dftd3_only.txt`) both PASS: largest thermo.out rel 1.46e-6 (`dftd3_kspace_one_each`), restart.xyz rel 1.6e-7. (`dftd3_duplicate` passed strictly.)
  - `dftd2_basic`, `dftd2_dftd3_exclusive`: the BASELINE fails (master has no `dftd2` keyword: exit 1 / diagnostic missing); the candidate runs and produces output.
    This is a baseline limitation (new feature), not a candidate regression; the case needs a baseline that has `dftd2` to be judged.

## S4: `pytest -m "not slow" -q` in `tests_pytest` (conda python)
`pytest -m "not slow" -q -p no:cacheprovider` (conda python): **859 passed, 480 skipped, 10 deselected, 0 failed** in 1572 s (`results/pytest.log`). Baseline of AGENTS.md was 738 passed / 480 skipped; the +121 passed are tests added after that measurement (D3/D2 work); skip count unchanged.

## Not verified / limits
- Independent numerical review, NVE closure (VER-0011 protocol), HIP build, other GPUs are not part of this record. Timing is one GPU, single machine, medians of 3.
- The nsys and slope definitions differ by about 0.1-0.35 ms (host/launch/array-addition overhead); AC-4 uses the slope definition.
