# VER-0004: DFT-D3 GPU optimization (DR-001 Phases 3 and 4) - results

- Date: 2026-11-03. Status: Approved by the human on 2026-11-03; verification record (regression and performance measurement;
  not a validation against DFT or experiment). Code: commit `84e5fa83` on `develop`.
- Subject: `src/force/dftd3.cu` (only file changed, +436/-156).
- Requirement / design: `DR-001`, `DES-0001` (Phases 3 and 4; Phase 2 is on hold, see
  Phase 2'), `ADR-0002`. Baseline measurements: `VER-0001`, `VER-0002`.
- Environment: RTX A6000 (sm_86, 84 SMs), CUDA 12.3 (`/usr/local/cuda`), NEP89
  (`nep89_20250409`), `dftd3 pbe 12 6`, built with `make CUDA_ARCH="-arch=sm_86" -j4`
  (no warnings from `dftd3.cu`).

## 1. What changed
- **Phase 4 (small box, any periodic thickness <= 2.5 rc).** `find_neighbor_list_small_box` is one
  block (128 threads) per atom; candidates (n2, ia, ib, ic) are processed in blocks of 128 in the
  original order, and a shared-memory inclusive scan gives each accepted candidate its slot.
  List order, values and layout are unchanged; the `N x MN` buffers (MN = 10000) are unchanged.
- **Phase 3 (large box).** `find_dftd3_force_large_box`, `find_dftd3_force_extra_large_box` and
  `find_dftd3_coordination_number_large_box` are templated on `num_sub`: `num_sub` consecutive
  threads share one atom, split the 125 neighbor cells, and reduce partial sums with a
  fixed-order shared-memory tree (no atomics). `num_sub` is chosen at run time: 1 for N >= 1e5,
  otherwise the largest power of two with N*num_sub <= 3.4e5 (cap 16 for N <= 4125, else 8), then
  limited by the number of cells (5^(periodic axes)). With `num_sub = 1` the block size is 64 and
  the summation order equals the old code; the kernels use `blockDim.x` and shared memory
  sized for 128. The numbers (4125, 3.4e5, 1e5, caps) are tuned on one RTX A6000.
- Float summation order differs for `num_sub > 1`, so results change at round-off level.

## 2. Accuracy and regression (AC-1, AC-2)
| Check | Result |
|---|---|
| `tests_pytest/test_dftd3_reference.py` (7 cases vs simple-dftd3) | 7 passed |
| same with `num_sub` forced to 1, 2, 4, 8, 16 and cell-count capping (scratch-only env-var build) | 7 passed in each of 6 configurations |
| `tests_pytest/test_io_basic_setup.py -k dftd3` | 7 passed, 2 skipped (reason not examined) |
| `run_regression.py --suite full --case 'dftd3_*'` (baseline = HEAD build before the change) | 3 passed, 0 failed (manifest default comparison; the rtol 1e-5 / atol 1e-12 of DR-001 AC-2 was not applied separately) |
| `run_regression.py --suite full`, baseline = `master` 3b2c8db8 (built sm_86), candidate = this change | 218 passed, 37 failed, relations 25/25 passed; the 37 failures are the known non-deterministic list of `AGENTS.md`, no other case failed (the size of each of the 37 differences was not examined). The `nep` binary of `master` was used on both sides. Summary: `results/suite_full_summary.txt` |
| Phase 4 neighbor lists, old vs new kernel (harness `scripts/harness.cu`, full `N x MN` buffers compared with `memcmp`): 13 cases (candidate counts multiple and non-multiple of 128, multi-chunk lists up to 838 neighbors, N=1, triclinic, partial PBC) | bitwise identical in all 13 (N=1 cases have 0 neighbors, a weak check) |
| Phase 4, 648-atom NVE, double dump, frames 0 and 1, old vs new | dE = 0, max |dF| <= 5.3e-15, dW <= 2.8e-14 |
| Phase 3, force difference vs `num_sub=1` (water, Cu, slab, cluster; N = 1,372-14,580) | <= 1.7e-5; two runs of the same binary with `num_sub=1` already differ by <= 1.3e-5 (cell-list `atomicAdd` order in the NEP/cell code), so the D3 contribution alone is not separated; the reference test above is the accuracy evidence |
| Phase 3, noise-free small systems (7-129 atoms), force difference vs `num_sub=1` | 6e-11 to 7e-7 |
| Phase 3, atom-order permutation (6 systems) | force difference <= 1.9e-5, equal to the noise of the old binary (<= 2.0e-5) |
| Phase 3, tail blocks: N = 2, 3, 7, 33, 127, 129, `compute-sanitizer` racecheck and synccheck, 24 runs (`num_sub` = 1, default, 4, 16) | 0 errors, 0 hazards (N = 1 is rejected by gpumd itself) |

## 3. Performance (AC-4), nsys kernel time, ms/step
Method: `nsys profile`, 100 NVE steps, old binary (HEAD before the change) and new binary
alternated twice, averages; scripts `scripts/bench.py`, `scripts/alt.sh`, `scripts/run.sh`
(paths point to the scratch directory used; adapt before reuse). "D3 total" = neighbor list
(`find_cell_*` or small-box list) + CN + force + extra kernels. This is the kernel time, not
the VER-0001 slope method (100 vs 500 steps). Raw data: `results/perf_merged_vs_head.txt`.

| System | Force kernel old -> new | D3 total old -> new | AC-4 target | Status |
|---|---|---|---|---|
| water 81 (small box) | 1.346 -> 1.361 (0.99x) | 1.445 -> 1.460 | <= 1.0 | not met |
| water 648 (small box) | 1.415 -> 1.401 (1.01x) | 1.564 -> 1.553 | <= 2.0 | met (already before) |
| slab 1,296 (periodic x, y) | 1.102 -> 0.146 (7.5x) | 1.371 -> 0.210 | - | - |
| cluster 972 (non-periodic) | 1.171 -> 1.172 (1.00x) | 1.793 -> 1.808 | - | no slowdown |
| Cu 4,000 | 3.087 -> 0.436 (7.1x) | 3.645 -> 0.573 | <= 2.0 | met |
| water 5,184 | 3.321 -> 0.766 (4.3x) | 3.828 -> 0.920 | <= 2.0 | met |
| water 41,472 | 6.559 -> 5.458 (1.20x) | 8.221 -> 6.962 | <= 6.0 | not met |

- Small-box neighbor-list kernel, water 648: 2.79-2.86 -> 0.38-0.40 ms/step (about 7.2x; gate
  of Phase 4 was 2x). This kernel is part of the D3 total of the 648-atom row above; the total
  barely changes because the force kernel (1.4 ms) dominates in the small-box path.
- N scan (force kernel, water cut to N atoms, old -> new, ms): 2,048: 2.08 -> 0.35;
  8,192: 3.41 -> 0.99; 16,384: 4.55 -> 2.03; 32,767/32,768: 5.28 -> 3.82; 40,000: 6.15 -> 5.14;
  82,944: 11.57 -> 10.43; 124,416: 13.78 -> 13.97 (+1.4%); 165,888: 18.77 -> 18.87. No
  discontinuity. (Agent-reported; not repeated by the parent.)
- No system was more than 3% slower than before (largest: +1.0% for water 81, within run
  scatter).
- Resource usage (`cuobjdump --dump-resource-usage`): force_large_box 72 registers, 0 B shared
  for `num_sub = 1`, 7,680 B for `num_sub > 1`; extra_large_box 70 / 63 registers, 0 / 6,144 B;
  CN_large_box 52 / 48 registers, 0 / 512 B.

## 4. Independent reviews (fresh `numerical-gpu-reviewer` and `api-compat-reviewer` agents)
- Phase 4: 0 BLOCKER, 0 MAJOR. MINOR: launch failed for N = 0 (fixed), `MN` overflow is not
  guarded (pre-existing, out of scope; known limitation).
- Phase 3 (first version and the final version): 0 BLOCKER, 0 MAJOR. MINOR: N = 0 guard,
  `switch` default, `static_assert`, dead code with out-of-range constant index for
  `num_sub = 1` (all fixed), `num_sub` rule not justified outside one GPU (rule re-derived from
  measurements, see section 1).
- The merged file was not reviewed again after the last MINOR fix (shared-array sizes only; the
  resource usage of all 15 instances is identical before and after).

## 5. Not verified / limitations
- HIP build, `sm < 600` fallback, float vs double comparison, long NVE energy conservation,
  memory usage measurement (allocations unchanged by inspection; `N x MN` buffers remain).
- `MN = 10000` overflow still writes out of bounds (pre-existing).
- Performance by the slope method; GPUs other than the RTX A6000; the `num_sub` rule constants.
- Results are not bit-identical to the old code for `num_sub > 1` (summation order).
- AC-4 not met: N = 81 (1.46 ms vs 1.0) and N = 41,472 (6.96 ms vs 6.0). Next: Phase 2'
  (factorized C6 weights, the small-box force kernel was 1.22-1.27x faster in the Phase 2
  prototype) and Phase 5.

## 6. Acceptance-criteria mapping (DR-001)
| AC | Status |
|---|---|
| AC-1 accuracy | met by `test_dftd3_reference.py` (all `num_sub` values) |
| AC-2 no regression | met: `--suite full` equals the known list, `dftd3_*` pass |
| AC-3 NVE energy conservation | not done |
| AC-4 performance | partly met (4 of 6 target points; see section 3) |
| AC-5 portability | CUDA warning-clean only; HIP and sm<600 not built |
| AC-6 docs | not done (cost scaling statement for `dftd3.rst`, Phase 6) |
