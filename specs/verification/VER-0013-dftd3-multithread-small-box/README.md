# VER-0013: DFT-D3 small-box force kernels with several threads per atom (DES-0001 Phase 7 Stage 2 item (i))

- Date: 2026-11-04. Status: verification (timing, round-off agreement, determinism; NOT validation). Requirement / design: `DR-001`
  AC-1, AC-2, AC-4, AC-5; `DES-0001` Phase 7 Stage 2 item (i). Only item (i) was done; (ii)-(v) untouched.
- Code under test: worktree `wt-d3-phase7b` (base HEAD `6c553308`), changed `src/force/dftd3.cu`, `tests_pytest/test_dftd3_reference.py`,
  `doc/gpumd/input_parameters/dftd3.rst` (one sentence). Not committed.
- Environment: RTX A6000 (sm_86, 84 SMs), CUDA 12.3, NEP89, `dftd3 pbe 12 6`, `make CUDA_ARCH="-arch=sm_86" -j4`. One GPU job at a time.
  Baseline binary for timing ("base") = the unchanged HEAD built in a scratch copy of `src/`; AC-2 baseline = `master-base` binary.

## Change
`add_dftd3_force_small_box` and `add_dftd3_force_extra_small_box` (the two small-box kernels with a serial loop over the explicit neighbor list)
are templates on `num_sub` (1..128, power of two): `num_sub` consecutive threads of a 128-thread block (64 for `num_sub == 1`) share one atom,
take every `num_sub`-th list entry, and the partial sums are reduced in a fixed tree order in shared memory (`reduce_sub_large_box`, the Phase 3
reduction of the large-box path; every thread of the block calls it, no `__syncthreads` inside a branch; invalid tail threads use atom 0 for loads and
never store).
- The 15 `atomicAdd(double)` per pair are gone: the neighbor list is full (each pair is in both atoms' lists), so each atom accumulates its own
  force (2 x f12, as in the large-box kernel) and virial (-r12 x f12) from its own list; per-atom virial attribution is unchanged (verified, see below).
  For the CN term (extra kernel) the force on atom i from pair (i,j) is k (D_i + D_j) r_ij and the per-atom virial uses D_j, which reproduces the old
  scatter (`D = dc6_sum + dc8_sum`).
- Accumulators are float (as in the large-box kernels); previously double atomics of float terms. The sm<60 `atomicAdd(double)` fallback of this file is
  removed because no atomic is left in `dftd3.cu`.
- Choice of `num_sub` (`get_num_sub_small_box`): largest power of two <= 128 with `N * num_sub <= SM_count * 640` and an expected list length
  (continuum estimate (4/3) pi rc^3 N / V) of at least 4 entries per thread; separate values for the radial (force) and angular (extra) kernels.
  640 threads/SM was tuned on this GPU from `results/num_sub_scan.json` (see below). Test aid: env `GPUMD_DFTD3_SMALL_BOX_NUM_SUB` (1..128) fixes it
  and prints `DFT-D3 small box: threads per atom = X` once (not in the manual, like `GPUMD_DFTD2_TARGET_THREADS`).
- Determinism: before, the n2 scatter used double atomics in scheduling order (run-to-run non-deterministic at the 1e-16 level). Now every output
  element is written by one thread and summed in a fixed order: bit-reproducible run to run (tested for all `num_sub`).
- Round-off: float accumulation and 2 x f(i,j) instead of f(i,j) + (-f(j,i)) change results at the float level (see AC-1/AC-2 numbers).
  A pair that is accepted from one side only by float rounding at r = rc no longer gets the Newton-3 partner force (same as the large-box path);
  the effect is of the order of the force at the cutoff (not measured separately).

## S1 gate (prototype, force kernel, default rule; nsys median; base = old kernel)
| system | base force [ms] | new force [ms] | speedup | gate |
|---|---|---|---|---|
| water 81 | 1.383 | 0.019 | 73x | >= 1.5x MET |
| water 648 | 1.438 | 0.054 | 27x | MET |
| water 2,187 | 1.659 | 0.143 | 11.6x | MET |
| water 8,748 | 2.117 | 0.425 | 5.0x | - |
(S1 numbers from a 40-step run each, `results/num_sub_scan.json` and the log of the first prototype; the extra kernel: 0.067 -> 0.009 ms at N=81.)

## S2 gate (D3 total, VER-0004 section 3 definition, base/new alternated 3x, median; `scripts/run_s2.py`, `results/s2.json`, `results/s2.log`)
| system (path) | base [ms] | new [ms] | speedup | note |
|---|---|---|---|---|
| water 81 (small) | 1.516 | 0.101 | 15.1x | gate >= 1.2x MET |
| water 648 (small) | 1.873 | 0.487 | 3.84x | MET |
| water 2,187 (small) | 2.380 | 0.793 | 3.00x | |
| water 8,748 (small) | 10.126 | 8.285 | 1.22x | NL 7.77 ms |
| cluster 972 (large box path) | 1.830 | 1.835 | 1.00x | unchanged code |
| water 5,184 (large) | 0.925 | 0.938 | 0.99x (+1.4 %) | unchanged code, run scatter |
| water 41,472 (large) | 6.969 | 6.993 | 1.00x | unchanged code |
| Cu 4,000 (large) | 0.582 | 0.586 | 0.99x | unchanged code |
No point is slower than 1.03x (largest +1.4 %, large-box kernels are not modified). AC-4 at N=81 (<= 1.0 ms) is met now (0.101 ms).
After (i) the neighbor-list kernel share of the D3 total is 39 % (N=81), 77 % (648), 72 % (2,187), 94 % (8,748).

## Threads-per-atom scan (`scripts/scan_num_sub.py`, `results/num_sub_scan.json`; force kernel [ms], default rule | 1 2 4 8 16 32 64 128)
| N | default | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| 81 | 0.019 | 1.099 | 0.560 | 0.284 | 0.145 | 0.081 | 0.049 | 0.030 | 0.019 |
| 648 | 0.054 | 1.374 | 0.714 | 0.374 | 0.198 | 0.085 | 0.072 | 0.054 | 0.055 |
| 2,187 | 0.144 | 1.384 | 0.733 | 0.381 | 0.214 | 0.144 | 0.183 | 0.174 | 0.188 |
| 4,374 | 0.236 | 1.387 | 0.737 | 0.407 | 0.235 | 0.357 | 0.477 | 0.355 | 0.346 |
| 8,748 | 0.442 | 1.323 | 0.738 | 0.419 | 0.289 | 0.952 | 1.231 | 0.684 | 0.686 |
| 17,496 | 0.811 | 1.370 | 0.804 | 0.589 | 0.688 | 2.021 | 2.734 | 1.360 | 1.391 |
| 34,992 | 1.504 | 1.503 | 1.057 | 1.463 | 1.440 | 4.766 | 5.342 | 3.138 | 3.206 |
Above a few thousand atoms many threads per atom lose (the list reads of a warp spread over many rows); the default rule is within 1.0-1.5x of the
best fixed value for N >= 4,374 (17,496: 0.81 vs best 0.59; 8,748: 0.44 vs 0.29) and the old kernel took 2.1 / 5.9 ms at 8,748 / 34,992. Not tuned further.
The old kernel's 34,992 value (5.865 ms) is from VER-0012; with `num_sub = 1` the new kernel alone (no atomics) takes 1.50 ms.

## Tests
- `tests_pytest/test_dftd3_reference.py` (existing tests and tolerances unchanged): 7 original tests pass; new: every `num_sub` in
  {1,2,4,8,16,32,64,128} on the 4 small-box reference cases (graphite, water27, water27_rc25, water27_tri): bit-identical on a repeat run,
  meets the AC-1 tolerance against the simple-dftd3 reference, and agrees with `num_sub = 1` within the AC-1 caps; and the three large-box cases ignore the variable.
  `pytest -q test_dftd3_reference.py`: 14 passed (135 s). (The non-periodic benzene dimer uses the large-box path.)
- AC-1 margins, new vs base binary (`scripts/accuracy.py`, `results/accuracy.txt`): all errors within the unchanged tolerance. Largest change:
  graphite dE 3.94e-6 -> 4.63e-6 (tol 7.88e-6), dF 4.57e-7 -> 5.32e-7 (tol 1.0e-6); others change by < 20 % of the tolerance or improve.
  Candidate - base (default num_sub): max |dF| 2.9e-7 (graphite), per-atom virial 7.2e-7 (graphite, of max 0.18), per-atom energy 1.5e-7; large-box cases: bit-identical or <= 2e-8.
- AC-2: `run_regression.py --suite full --case 'dftd3_*'` baseline `master-base`, candidate this build: strict manifest tolerance fails 2 of 3 as expected
  for any round-off change (`results/regression_dftd3.log`); under the DR-001 AC-2 tolerance (rtol 1e-5, atol 1e-12, `scripts/compare_ac2.py`,
  `results/ac2_dftd3.txt`) all pass: largest differences thermo.out rel 1.0e-7, restart.xyz rel 4.0e-8.
- Build: `make CUDA_ARCH="-arch=sm_86" -j4` warning-clean for `dftd3.cu`; `-Xptxas -v`: 0 spills for all instantiations. clang-format: the changed regions are clean.

## Not verified / limits
- HIP and sm < 60 were not built; no atomics remain, the removed `atomicAdd` fallback was used only by the old kernels.
- Only RTX A6000; the 640 threads/SM target is tuned there.
- Energy conservation (AC-3) and `--suite full` were not run (left to the final stage).
- The manual tables of measured D3 times (`doc/.../dftd3.rst`) now overstate the small-box times; they are re-measured in the final stage.
- The CN kernel (`find_dftd3_coordination_number_small_box`, 0.033 ms at N=81) is still one thread per atom (not in item (i)).
