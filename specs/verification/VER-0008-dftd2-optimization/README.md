# VER-0008: DFT-D2 pair-kernel optimization (DES-0002 Phase 6 stage 2; DR-002 AC-7 informative, AC-1 unchanged)

- Date: 2026-11-03. Status: verification (accuracy against the NumPy reference with the unchanged AC-1 tolerances; timing). NOT validation.
- Binary: worktree `wt-d2-phase6b` (base HEAD b7059f30 + items i and ii), `make CUDA_ARCH="-arch=sm_86" -j4`, RTX A6000 (84 SMs), CUDA 12.3.
- Pair-kernel times: `nsys profile -t cuda` + `nsys stats -r cuda_gpu_kern_sum` (median of the kernel instances), harness `scripts/harness.cu`
  (calls `DFTD2::compute` N_iter times on a fixed configuration; `scripts/build2.sh` builds it against the worktree sources, `scripts/lib.py` runs it),
  systems from `scripts/gen.py` (water27 replicated; Cu fcc 10^3 cells). Medians of 3 alternating repetitions (base and candidate interleaved).
  "base" = HEAD source of `src/force/dftd2.cu` built in the same harness.

## 1. Result summary
| item | gate | measured | decision |
|---|---|---|---|
| (i) cell counts from a cost estimate | >= 1.3x at water 648 rc 30 or >= 1.5x at rc 20; no case slower than 1.03x | 1.62x (rc 30), 2.28x (rc 20); worst case 0.97x-1.03x (within run-to-run scatter, see 2) | kept |
| (ii) several threads per atom, fixed-order reduction | >= 1.3x at water 648 rc 30 on top of (i); no slowdown above 1.03x at N >= 5,184 | 9.4x on top of (i) (15.2x total); N >= 5,184: 0.98x-3.4x | kept |
| (iii) cell size / filtering / tiling | only if (i)+(ii) leave >= 1.2x measured potential on N = 5,184 or Cu 4,000 at rc 30 | forced cell counts n = 3..10 (cubic) are all slower than the automatic choice (best 1.11x slower on Cu, 1.47x slower on water 5,184): no potential in cell size | NOT attempted (gate not met) |

## 2. Item (i): cost model for the cell counts
Model: per atom, work = (candidate atoms visited) + C0 * (neighbor cells visited), visited = N * prod_d(2k_d + 1) / prod_d n_d for a uniform density,
k_d = ceil(rc / (width_d / n_d)). Candidate counts per direction: n = 1 and floor(kk * width / rc), kk = 1..6 (cell width >= rc / kk); each count is limited to cbrt(max_cells).
The old rule was n = floor(2 * width / rc) (kk = 2 only).
Calibration (`scripts/run_fit.py`, `scripts/fit.py`, `results/tab_fit.txt`): the old kernel with a forced cubic n = 1..8 on water 648, 5,184 and Cu 4,000 at rc = 12, 20, 30
(40 points). Fit t = A_sys * (visited + C0 * cells + Ca * accepted) (accepted pairs are the same for every n and do not change the minimizer): the rms log error is
0.14 at C0 = 3, Ca = 3 (0.35 without any cell or accepted term). C0 = 3 is used (CELL_VISIT_COST). The estimate was fitted to these runs;
the validation below is on the same systems plus cutoffs not used in the fit (14, 16, 18, 25) and water 20,736.

Pair-kernel time, base vs item (i) only (`results/tab_it1.txt`, ms):
| system | rc 12 | 14 | 16 | 18 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|
| water 648 | 1.00x | 1.24x | 1.15x | 1.15x | 2.28x | 1.70x | 1.62x |
| water 5184 | 0.97x | 1.03x | 1.08x | 1.16x | 1.32x | 1.97x | 1.77x |
| Cu 4000 | 1.03x | 0.99x | 1.13x | 1.00x | 1.51x | 2.12x | 1.45x |
Speedup = base / candidate. The 0.97x (water 5,184, rc 12) uses the same cell counts as the base (n = 6, k = 2): it is scatter (2.911 vs 2.989 ms), as are 0.99x and 1.00x.
The old 16 -> 20 A jump is gone (water 648: 7.7 -> 15.3 ms instead of 7.5 -> 35.0 ms).

## 3. Item (ii): threads per atom
NUM_SUB consecutive threads share one atom (sorted order). In each neighbor cell (including periodic images) thread `sub` takes atoms `start + sub, start + sub + NUM_SUB, ...`;
the 13 per-thread sums (E, F x3, W x9, double) are added by a tree in shared memory (offsets NUM_SUB/2, ..., 1), fixed order, no atomics. All threads of a block
reach the reduction (out-of-range threads contribute zero). NUM_SUB = largest power of two <= 32 with N * NUM_SUB <= (number of SMs) * 128 and an estimated
>= 32 candidates per thread (rule picked from the measured sweep `results/tab_ts.txt`, `tab_ts2.txt`: the optimum N * NUM_SUB is about 1e4 on this 84-SM device
(648 x 16, 324 x 32, 2025 x 4, 5184 x 2); more threads slow the kernel by the reduction and the divergence).
Pair-kernel time, base vs items (i) + (ii) (`results/tab_it2.txt`, ms; `scripts/run_cmp.py it2 3 ...`):
| system | rc 12 base -> new | speedup | rc 20 | speedup | rc 30 | speedup |
|---|---|---|---|---|---|---|
| water 81 | 4.30 -> 0.73 | 5.9x | 12.06 -> 1.67 | 7.2x | 29.03 -> 1.88 | 15.5x |
| water 324 | 3.43 -> 0.32 | 10.8x | 26.51 -> 2.00 | 13.3x | 44.01 -> 2.28 | 19.3x |
| water 648 | 2.79 -> 0.30 | 9.4x | 34.98 -> 2.16 | 16.2x | 47.10 -> 3.09 | 15.2x |
| water 2025 | 4.22 -> 1.00 | 4.2x | 16.51 -> 3.76 | 4.4x | 40.13 -> 10.45 | 3.8x |
| water 5184 | 2.91 -> 1.53 | 1.9x | 16.55 -> 6.50 | 2.6x | 52.98 -> 15.50 | 3.4x |
| water 20736 | 5.55 -> 5.65 | 0.98x | 29.19 -> 22.69 | 1.3x | 99.92 -> 59.10 | 1.7x |
| Cu 4000 | 3.06 -> 1.49 | 2.1x | 14.83 -> 5.38 | 2.8x | 43.91 -> 15.13 | 2.9x |
(water 20,736 at rc 12: NUM_SUB = 1, same kernel as before except the cell choice; 0.98x is scatter.)

## 4. Item (iii): diagnostic (no code kept)
`results/tab_pot.txt` (`scripts/run_pot.py`, cubic n forced on top of items i + ii, rc = 30): auto 15.5 ms (water 5,184) / 15.1 ms (Cu 4,000); forced n = 3 ... 10: 22.7-68.2 ms / 16.8-104.5 ms.
No cell size beats the automatic choice, so the cell-size/CELL_DIVISOR variant of (iii) has no measured potential; gate not met, not implemented. Shared-memory tiling was not tried.
Observation (not in scope, no code kept; `results/f32_potential.txt`): the candidate loop computes `pj.x + tx - pi.x` in double (cheap on a data-center GPU, 1/64 rate on this GA102),
and a timing-only variant that reads float positions in the candidate loop is 1.3-1.7x faster (accuracy not checked). That changes the precision of the pair distance
(about 1 float ulp of the coordinate, ~1e-6 A at 15 A) and was not attempted; a decision is needed.

## 5. Accuracy and determinism (AC-1 unchanged)
- `pytest -q -m "not slow" test_dftd2_reference.py test_dftd2_src_table.py test_dftd2_consistency.py` (existing tests unchanged, plus the new test below): final run (kept code): 88 passed, 1 deselected (the slow case), 399 s.
- Phase 5d determinism test (dense water 2187, rc 30; now NUM_SUB = 4 there) passes: bit-identical energy, forces and virial over 5 runs.
- New test `test_dftd2_matches_reference_cell_and_thread_selection` (9 cases: N = 81 rc 12/30, N = 648 rc 12/20/30, triclinic 648 rc 20, open 648 rc 20, N = 5,184 rc 12 and rc 20 (slow, 380 s))
  against the NumPy reference with the caps E, W 2e-6, F 1e-5: all 9 pass.
- Mutation sanity (rebuilt binary, then restored): reduction starting at offset NUM_SUB/4 (misses half of the partial sums): n648_rc12 and n648_open_rc20 fail (n81_rc12 passes:
  there are only 3 atoms per cell, so only 3 of 32 threads have work; this shows that the 648 cases are needed); stride `NUM_SUB/2 + 1` instead of `NUM_SUB`: the same two cases fail.

## 6. Slope-method timing (`scripts/timing.py`, `results/timing_new.txt`, `results/timing.json`; same method as the previous measurement)
Extra = (run with the term) - (run without), ms per step of the whole MD step (NEP89, water, NVE dt 0.5 fs), D3 = `dftd3 pbe 12 6`.
| N | NEP only | D3 extra | D2 rc12 extra: before -> now | D2 rc30 extra: before -> now | D2(30)/D3 before -> now |
|---|---|---|---|---|---|
| 81 | 1.41 | 1.40 | 4.27 -> 0.63 | 30.6 -> 1.79 | 22.6 -> 1.3 |
| 648 | 1.60 | 1.85 | 3.04 -> 0.17 | 54.3 -> 2.92 | 27.1 -> 1.6 |
| 5184 | 1.75 | 1.04 | 3.15 -> 1.56 | 55.9 -> 15.96 | 54.1 -> 15.4 |
| 41472 | 6.75 | 7.43 | 12.03 -> 12.01 | 211.4 -> 116.9 | 28.1 -> 15.7 |
"Before" is the previous measurement (a different session on the same machine). The 648 rc 12 extra (0.17 ms) is at the noise level of the slope method.
AC-7 (D2 <= 0.5 x D3): met at rc 12 for N = 81 and 648 (0.63 vs 1.40, 0.17 vs 1.85 ms), not at N = 5,184 and 41,472 (rc 12: 1.5x, 1.6x D3); not met at rc 30 for any N (1.3x-15.7x of D3). Informative only.

## 7. Limitations / not verified
- Tuned and measured on one GPU (84 SMs, sm_86). The thread target scales with the SM count; the constants (CELL_VISIT_COST = 3, 32 candidates per thread) are not re-tuned for other devices.
- Cell cost model assumes a uniform density; strongly inhomogeneous systems (slabs, clusters in vacuum) may pick a non-optimal n (still correct).
- With few atoms per cell NUM_SUB can exceed the atoms per cell (idle threads, e.g. N = 81, rc 12); a cell-level split would use them better (not tried).
- `--suite full` and the finite-difference checks at rc 30 (stage 3) were not run here.
