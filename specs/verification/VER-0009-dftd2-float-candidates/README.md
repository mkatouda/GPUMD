# VER-0009: DFT-D2 candidate loop with single-precision, cell-relative positions (DES-0002 Phase 6 stage 2b; DR-002 AC-1 unchanged, AC-7 informative)

- Date: 2026-11-03. Status: verification (accuracy against the NumPy reference and the previous double-coordinate GPU kernel, with the unchanged AC-1 tolerances; timing). NOT validation.
- Binary: worktree `wt-d2-phase6c` (base HEAD 1476643a), `make CUDA_ARCH="-arch=sm_86" -j4`, RTX A6000 (84 SMs), CUDA 12.3.
- Question (VER-0008 sec. 4 observation): the pair-kernel candidate loop forms `pj + t - pi` in double (1/64 rate on GA102); a float variant was 1.3-1.7x faster (timing only).
  Does it stay within AC-1 (E, W 2e-6 of |E_ref|, max|W_ref|; F 1e-5 of max|F_ref|; test_dftd2_reference.py, unchanged), and is it faster?

## 1. Variants
- "dbl": HEAD kernel (positions double4, `dx = float(pj.x + tx - pi.x)`).
- "fa" (literal float coordinates, scratch only, not kept): sorted positions float4 = wrapped absolute coordinates; per cell `off = float(t - pi)` (double), `dx = pj + off`.
- "fb" (adopted): sorted positions float4 relative to the CENTER of the atom's own cell. Per neighbor cell the displacement between the two cell centers
  (`H * diag(len/n) * (ox, oy, oz)`, built hierarchically in double over the oz/oy/ox loops) is rounded once to float: `dx = (pj.x - pi.x) + off.x`.
  The pair vector is exactly antisymmetric in i <-> j (same float operations, signs flipped), so the net force is zero to rounding.
  Absolute error of each component of dx ~ 6e-8 x (pair distance r + cell width w), independent of box size and of the distance from the origin
  (corrected in stage 3b; the original text said "1e-7 x cell width"): the offset (up to k*w ~ rc) is rounded to float and the sum `(pj - pi) + off` is rounded again.
  For close pairs this is up to (1 + w/r) times the error of rounding the exact double vector once (~6e-8 r), e.g. r = 3 A, w = 12 A: 5x (2e-7 -> 1e-6 A) worse than the pre-2b kernel.
  The periodic shift (`tx`) is no longer used for the geometry: the image cell offset is the unwrapped cell displacement.
- Everything else (float pair math, double accumulators, NUM_SUB reduction, cell choice) is unchanged.

## 2. S1 accuracy diagnostic (gate: every case <= 1/3 of the AC-1 tolerances, i.e. "AC1 frac" <= 0.333)
Scripts: `scripts/acc.py` (cases, NumPy reference `tests_pytest/dftd2_reference.py` with s6 = 0.75, GPU harness `scripts/harness2.cu` dumping per-atom E, F, W), `scripts/build.sh`.
AC1 frac = max(dE/E_ref / 2e-6, dF/Fmax_ref / 1e-5, dW/Wmax_ref / 2e-6), all against the NumPy reference. Raw output in `results/s1_accuracy_*.txt`.
The double GPU kernel itself differs from the reference by dE 1e-8..1e-7, dF 4e-7..1.4e-6, dW 3e-8..2e-7 (float pair math), i.e. AC1 frac about 0.03-0.14 by itself (largest: dF 1.4e-6 on triclinic water).

| case (N; rc 12 and 30 each) | fa AC1 frac | fb AC1 frac | sum F / Fmax fa -> fb |
|---|---|---|---|
| water 81, cubic 9.3 A | 0.29 | 0.12 / 0.27 | 2e-6 -> 3e-16 |
| water 648, cubic 18.6 A | 0.42 | 0.14 | 2e-5 -> 7e-16 |
| water 648 x3 cell (56 A) | 0.50 | 0.06 / 0.11 | 9e-8 -> 6e-16 |
| water 648 x5 cell (93 A) | 0.73 | 0.11 | 4e-8 -> 5e-16 |
| water 648 triclinic | 0.64 | 0.15 / 0.20 | 1e-5 -> 6e-15 |
| water 648 triclinic x4 cell | 0.67 | 0.19 / 0.09 | 4e-8 -> 1e-16 |
| water 648 open | 0.33 | 0.20 | 3e-15 both |
| water 648 open, +100 A from origin | 2.53 (FAIL) | 0.20 | 3e-15 both |
| Cu 4000 rattled 0.15 A (a = 3.61, 36 A) | 0.55 | 0.17 / 0.16 | 1e-5 -> 4e-15 |
| Cu 4000 perfect fcc (rc 12/30) and +(500,-720,333) shift | not run | E 4e-7 (0.2), W 4e-7 (0.24) | F ~ 0 by symmetry: relative force metric meaningless (reference F ~1e-17) |

Gate result: fb max AC1 frac = 0.27 (water 81, rc 30; an un-centered cell-origin trial gave 0.31 at water 648 rc 12, centering on the cell center is what is kept) <= 0.333: PASS.
fa fails (0.42-0.73 on ordinary systems, 2.5 for coordinates ~100 A away from the origin; net force 1e-5 of Fmax because the pair vector is not antisymmetric): NOT adopted.
Not covered by the NumPy runs: water 5184 and 20736 (reference too slow); accuracy at those sizes was not measured (only the 5184/rc 12 and 20 cases of the existing selection tests, marked slow, would cover it).

## 3. S2 timing (gate: >= 1.2x at water 648/5184 and Cu 4000 with rc 30; no case slower than 1.03x)
`scripts/tim.py`, nsys `cuda_gpu_kern_sum`, median of 3 alternating repetitions, ms per `dftd2_pair_kernel` instance, `results/timing_pair_kernel.txt`.

| system | rc 12 | 16 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| water 81 | 1.60x | 1.55x | 1.67x | 1.59x | 1.86x |
| water 324 | 1.49x | 1.59x | 1.66x | 1.58x | 1.71x |
| water 648 | 1.55x | 1.68x | 1.61x | 1.63x | 1.63x (3.09 -> 1.89 ms) |
| water 2025 | 1.57x | 1.62x | 1.58x | 1.59x | 1.66x |
| water 5184 | 1.58x | 1.59x | 1.64x | 1.67x | 1.63x (15.49 -> 9.48 ms) |
| water 20736 | 1.71x | 1.66x | 1.69x | 1.76x | 1.75x |
| Cu 4000 | 1.51x | 1.52x | 1.57x | 1.61x | 1.59x (15.13 -> 9.53 ms) |
Speed-up = HEAD / candidate. Gate PASS: 1.49x-1.86x everywhere, no slowdown. Registers 34 (unchanged), shared memory unchanged.

## 4. Tests
- `pytest -q -m "not slow" test_dftd2_reference.py test_dftd2_src_table.py test_dftd2_consistency.py` (existing tests and tolerances unchanged): 96 passed, 1 deselected (slow), 460 s (`results/pytest_final.txt`). Includes the dense-water 2187 / rc 30 bitwise determinism test and the cutoff +-4 ulp test.
- New tests in `test_dftd2_reference.py` (8): `test_dftd2_matches_reference_float_positions` (large cubic/triclinic boxes, non-periodic far from the origin, periodic translated off the origin; AC-1 caps)
  and `test_dftd2_is_invariant_under_a_rigid_shift_and_has_zero_net_force` (rc 12/30; D2 part of the shifted and unshifted system agree within the AC-1 caps; net D2 force <= the force cap).
- Not done: mutation check of the new tests against variant fa (not rebuilt in the tree; the equivalent S1 numbers show fa exceeds AC-1 only for the far-from-origin case, which `open_far_from_origin_rc12/30` mirrors).

## 5. Limitations / not verified
- `--suite full` regression and finite-difference/NVE checks were not run here (kernel math beyond dx is unchanged).
- Accuracy depends on the cell width and the pair distance: the absolute error of a pair-vector component is ~6e-8 x (r + w) (see sec. 1); cells are about rc / (1..6) wide; for a very sparse system in a huge box (cells up to L/10) w and hence the error grow proportionally.
- Tuned/measured on one GPU (sm_86, CUDA 12.3). The effect on other architectures (full-rate FP64) is expected to be smaller.

## 6. Stage 3b appendix (independent numerical review, MINOR findings)
Binary: worktree `wt-d2-phase6e` (base HEAD 22467397), sm_86, RTX A6000, CUDA 12.3. Source changes do not alter any computed value (the pair kernel is unchanged apart from the reduction helper).

### 6.1 Accuracy statement (M1)
Sec. 1 and sec. 5 corrected: the pair-vector error is ~6e-8 x (r + w), not 1e-7 x w. Dense water 2187 atoms, 27.9 A box, rc = 30 A gives 5 cells per direction and range k = 6 (the largest, cell width 5.6 A);
new test `test_dftd2_matches_reference_with_neighbor_range_6` (NumPy reference, unchanged AC-1 caps): error / cap = E 0.034, F 0.198, W 0.051 (i.e. 5x margin on the force, the tightest quantity).
Margins of the float-position cases (error / cap, E, F, W): big cubic box rc 30: 0.052, 0.112, 0.071; rc 12: 0.058, 0.117, 0.066; big triclinic rc 30: 0.017, 0.089, 0.029; non-periodic at +100 A: rc 12 0.003, 0.197, 0.079, rc 30 0.001, 0.195, 0.075;
at +300 A (new): rc 12 0.003, 0.197, 0.079, rc 30 0.001, 0.195, 0.075 (identical to +100 A to 3 digits); periodic, translated off the origin, rc 20: 0.003, 0.171, 0.035.
(The force fraction of ~0.2 in the open cases is the same at +100 and +300 A, so it comes from the single-precision pair math, not from the coordinates.)

### 6.2 All NUM_SUB instances (M2)
`GPUMD_DFTD2_TARGET_THREADS` (environment variable, read once in `DFTD2::initialize`, undocumented in the manual, commented in `dftd2.cu`) replaces the device-derived target thread count; with `target = N * s` the kernel uses s threads per atom.
When it is set, gpumd prints `DFTD2 debug: threads per atom = s, cells = ..., range = ...`, which the tests check, so each instantiation is really exercised.
Test `test_dftd2_all_threads_per_atom_agree_and_are_reproducible` (water 648, rc 12 and 30): s = 1, 2, 4, 8, 16, 32, each run twice: bit-identical within each s; agreement with s = 1 in error / cap (F, W): rc 12: <= 1.2e-11, 7.1e-11; rc 30: <= 3.4e-10, 1.4e-10; E identical to the printed precision.
(The dump header has limited energy precision; the force/virial differences are the order of double-precision summation-order noise.)

### 6.3 Detection power against the naive float variant "fa" (M3)
- Added: non-periodic cases at +300 A (accuracy vs reference, rc 12/30) and `test_dftd2_net_force_vanishes_far_from_the_origin` (rc 12/30).
- Net-force cap of the rigid-shift test tightened from 1e-5 to 1e-12 x Fmax (`NET_FORCE_CAP`): measured fb <= 4.4e-14 (results/stage3b_net_force.txt; cap = 23x margin); fa gives 1.1e-5..1.4e-5 for the periodic cases.
- Mutation check: fa was rebuilt temporarily (sorted positions = absolute float coordinates, `dx = pj + float(t - double(pi))` with the periodic image translation t; source restored afterwards, binary rebuilt, same size as before the mutation).
  With fa: `test_dftd2_matches_reference_float_positions[open_far_from_origin_rc12 / rc30 / open_far_300_rc12 / rc30]` FAIL (the +100 A cases existed before; they already detected fa), and both rigid-shift tests (rc 12, 30) FAIL on the net force with the new 1e-12 cap (measured 1.1e-5..1.4e-5 would also exceed the old 1e-5 cap, but the old cap was not re-run).
  The other float cases (big boxes, periodic off origin) pass with fa: they do not discriminate. Net force of non-periodic systems is antisymmetric even for fa (2e-14 at 300 A), so the new net-force-at-300-A test does NOT detect fa; the accuracy cases at 300 A do.
  Result: 6 failed, 4 passed in the selected set (`-k "float_positions or rigid_shift"`).

### 6.4 Other review items
- M4: `gpuGetDevice` + `gpuGetDeviceProperties(device_id)` (macro added to `gpu_macro.cuh`; `main_replica` already had it). M6: early return for N = 0.
- M5: the shared array is now inside `SubReduce<NUM_SUB>` (specialization for 1 is empty), no out-of-range reference. ptxas: pair-kernel instances 89 registers / 13312 B smem (NUM_SUB 2..32) and 88 registers, no smem (NUM_SUB 1), identical before and after.
  Timing (`scripts/tim.py`, nsys median of 3 alternating, `results/stage3b_timing_pair_kernel.txt`): water 648 rc 30: 1.894 / 1.894 ms (1.00x); water 5184 rc 30: 9.551 / 9.507 ms (1.00x) (before / after).
- Docs: cost statement updated to the VER-0010 final measurement (648 atoms ~1.1x NEP time, 41,472 atoms ~10x).

### 6.5 Tests after stage 3b
`pytest -q -s -m "not slow" test_dftd2_reference.py test_dftd2_src_table.py test_dftd2_consistency.py` (existing tests and tolerances unchanged; the new tests above added): 103 passed, 1 deselected (slow), 800 s (`results/stage3b_pytest_final.txt`). A first run had 1 failure caused by a bug in the new return value of the test helper (division by a zero tolerance in a case with E_ref = 0); fixed and rerun in full.
`--suite full` not run (computed values unchanged).
