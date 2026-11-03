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
  Absolute error of dx ~ 1e-7 x cell width (independent of box size and of the distance from the origin).
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
- Accuracy depends on the cell width: cells are about rc / (1..6) wide; for a very sparse system in a huge box (cells up to L/10) the absolute error grows proportionally (still ~1e-7 of the cell width).
- Tuned/measured on one GPU (sm_86, CUDA 12.3). The effect on other architectures (full-rate FP64) is expected to be smaller.
