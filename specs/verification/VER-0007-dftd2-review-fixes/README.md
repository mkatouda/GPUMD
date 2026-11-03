# VER-0007: DFT-D2 independent-review fixes (DES-0002 Phase 5d, DR-002)

- Date: 2026-11-03. Status: verification (code behavior, tests, closed-form estimate). NOT validation.
- Binary: worktree `wt-d2-phase5d` (base HEAD 76a01473 plus the changes of this phase), `make CUDA_ARCH="-arch=sm_86"`, RTX A6000, CUDA from /usr/local/cuda.
  Base binary = build of HEAD 76a01473 before the changes (kept in the session scratch directory).
- Human decisions applied: LGPL header unchanged; default rc stays 30 A; no heavy-element warning feature (documentation only); rc > 200 A only prints a warning.

## 1. Code and test changes (stage 1)
| item | change | result |
|---|---|---|
| 1a determinism | `src/force/dftd2.cu`: atoms of each cell are ordered by increasing original index (new `list_atoms` + rank-based `place_atoms`; scratch array `listed_index_`) | see 2.1 |
| 1b degenerate cases | 6 new GPU-vs-NumPy-reference cases (`DEGENERATE_CASES`) | all pass, 2.2 |
| 1c float/double rc | `initialize` rounds rc to float once (`rc = double(float(rc_in))`); shift terms, cutoff test and cell search use that same value; cutoff test now uses +-4 float ulp | passes, 2.3 |
| 1d rc upper limit | warning printed in `DFTD2::initialize` for rc > 200 A | test passes, 2.4 |
| 1e error message | see 2.5 (no code change needed) | |
| 1f FD flakiness | measured, docstrings corrected | 2.6 |

## 2. Results
### 2.1 Determinism (dense water 2187 atoms, rc = 30, s6 = 0.75, NEP89, 5 runs with dftd2 + 5 without; `scripts` inline in test `test_dftd2_dense_water_is_bitwise_reproducible`)
- NEP-only runs: bit-identical (spread 0.0).
- BASE binary (before the change): NOT bit-identical. D2-on spread over 5 runs: energy 0.0 (at printed precision), force max 1.8e-15 eV/A, virial max 5.7e-14 eV (|E_D2| = 78.64 eV, max |F_D2| = 0.0442 eV/A).
- NEW binary: bit-identical in energy, all force components and the virial over 5 runs. The new test fails on the base binary and passes on the new one.
- Cost: 300 NVE steps water 2187 (3 runs each, wall time): rc = 12: 3.2 s base / 3.2 s new; rc = 30: 42.8 s base / 38.7 s new (no slowdown).
### 2.2 Degenerate cases vs NumPy reference (relative errors, s6 = 75; caps E, W: 2e-6, F: 1e-5; `results/gpu_vs_reference_errors.txt`)
Maximum over all 17 GPU cases: E 3.3e-7, W 5.9e-7, F 1.7e-6. New cases:
`two_atoms_periodic` (E 2.6e-8, F 2.4e-7), `two_atoms_open_beyond_cutoff` (exactly 0), `tiny_box` (2.4 A box, rc = 12, k = 6; E 3.7e-8, F 7.5e-7), `sparse_large_box` (N = 1000, 300 A box, rc = 12, 50^3 cells halved to the cell limit; E 2.1e-7, F 3.5e-7), `triclinic_mixed_pbc` (T,F,T; E 5.6e-9, F 1.2e-6), `far_separated_open_direction` (150 A along the open z direction; E 1.8e-8, F 1.1e-6).
- N = 1 cannot be tested: gpumd itself stops with "Number of atoms should >= 2" (not a D2 property). Replaced by N = 2 periodic (self-images + pair) and N = 2 open beyond the cutoff.
- Coincident atoms: r = 0 gives a non-finite pair term (1/r^6, rinv = inf). It cannot be isolated in a GPU test because NEP89 alone already returns NaN energy, forces and virial for coincident atoms (checked). Decision: documented as unsupported (non-finite values, no check, no per-step synchronization added).
### 2.3 Float/double rc consistency
Before: shift terms phi(rc), phi'(rc) in double rc, cutoff test and `r - rc` with float(rc). Now rc is rounded to float once. The old cutoff test (rc = 15.3, distances rc +- 1e-6, about 1 ulp = 9.5e-7 A) was at the float resolution; it now uses rc_f +- 4 ulp (3.8e-6 A). Results change only at float rounding (all reference comparisons above are within the unchanged caps).
### 2.4 rc > 200 A warning
`dftd2 0.75 250` on an open dimer prints "Warning: the dftd2 cutoff (250 A) is larger than 200 A; ..." and the run completes (test `test_dftd2_large_cutoff_prints_a_warning_and_still_runs`).
### 2.5 Error message `dftd2` without potential
On the base binary `dftd2` without any `potential` line already gives the distinct message "There is no 'potential' keyword in run.in." (read_xyz.cu), not the "unsupported potential" message; the review observation did not reproduce in this form, so no code change was made (an attempted change in `run.cu` was unreachable and reverted). Test added (`test_dftd2_without_potential_gives_a_specific_message`). Reverse-order exclusivity (dftd3 then dftd2) was already tested (`test_dftd2_and_dftd3_are_mutually_exclusive`, both orders).
### 2.6 FD test run-to-run variation (`scripts/fd_variation.py`, 5 repeated full evaluations each)
`results/fd_variation_new_binary.txt`, `results/fd_variation_base_binary.txt`: identical in all 5 runs, for both binaries, at the printed precision (4 significant digits): relative force error 2.162e-5 (cubic), 1.862e-5 (open); relative virial error 3.594e-6 (cubic), 2.851e-6 (triclinic); tolerance 2e-4 (margins factor 9 and 55). No flakiness reproduced. The `Observed (s6 = 0.75)` docstring in `test_dftd2_reference.py` referred to the GPU-vs-reference test, which uses s6 = 75; corrected with the measured maxima above.

## 3. Heavy-element check of the closed form (stage 3; `scripts/heavy_element_closed_form.py`, `results/heavy_element_closed_form.txt`)
Continuum estimate (derived in the manual): per atom, shifted-force minus untruncated sum dE = 7 pi rho s6 C6 / (3 rc^3), dP = rho dE.
fcc 4x4x4 conventional cells (256 atoms), Au a = 4.078 A, Pt a = 3.924 A (typical experimental values), NEP89, s6 = 0.75; D2 part = (on) - (off); the untruncated limit from a fit E(rc) = E_inf + A rc^-3 over rc >= 60 A (rc = 30, 40, 60, 80, 120, 160 A).
Note: the table has the same C6 for Au and Pt (841.993 eV A^6; group value).

| element | rc [A] | dE meas [meV/atom] | dE closed [meV/atom] | dP meas [MPa] | dP closed [MPa] |
|---|---|---|---|---|---|
| Au | 30 | 10.109 | 10.112 | 95.54 | 95.56 |
| Au | 40 | 4.266 | 4.266 | 40.31 | 40.31 |
| Au | 60 | 1.264 | 1.264 | 11.94 | 11.95 |
| Au | 80 | 0.533 | 0.533 | 5.04 | 5.04 |
| Pt | 30 | 11.347 | 11.350 | 120.43 | 120.39 |
| Pt | 40 | 4.788 | 4.788 | 50.79 | 50.79 |
| Pt | 60 | 1.419 | 1.419 | 15.05 | 15.05 |
| Pt | 80 | 0.599 | 0.599 | 6.35 | 6.35 |

(rc = 120, 160 in the results file.) The fitted coefficients A_E agree with the closed form to 0.005 % (Au 273020 vs 273033, Pt 306452 vs 306457 meV A^3/atom); the fit uses the same points, so this is a consistency check of the r^-3 law and of the formula's coefficient, not an independent extrapolation (the closed form has no fitted parameter and reproduces dE at each rc to <= 0.04 %).
Reporting thresholds from the cutoff study (not pass/fail): T1 = 0.5 meV/atom is first met at rc about 82 A (Au) and 85 A (Pt) by the r^-3 scaling (0.533 and 0.599 meV at 80 A); T3 = 0.02 GPa is met at rc about 51 A (Au) and 55 A (Pt) by the r^-3 scaling. At the default 30 A both are exceeded by a factor 20 (energy) and 5-6 (pressure) for these metals. The doc states this; no code behavior was changed (default stays 30 A by human decision). Caveat: the D2 sum for metals is also physically questionable (fixed neutral-atom C6), see the manual's Limitations; this check validates only the truncation estimate.
