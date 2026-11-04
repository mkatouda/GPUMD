# VER-0015: DFT-D3 CN and extra kernels of the large-box path (DES-0001 Phase 7 item (vi))

- Date: 2026-11-04. Status: verification (timing, round-off agreement, bit identity); NOT validation. Requirement / design: `DR-001`
  AC-1, AC-2, AC-4, AC-5; `DES-0001` Phase 7 (vi). Stage 6a (diagnostic) gate: MET. Stage 6b: (1), (2) already in the code (Phase 3), (3) implemented and
  adopted by its gate. Left uncommitted in worktree `wt-d3-phase7d` (base HEAD `ccf7ed31`) for the human decision.
- Environment: RTX A6000 (sm_86, 84 SMs), CUDA 12.3, NEP89, `dftd3 pbe 12 6` (rc_radial 12, rc_angular 6), nsys (no ncu counters). One GPU job at a time.
  Timing base = unchanged HEAD built in a scratch copy of `src/`; AC-2 baseline = `master-base` binary. D3 total = sum of nsys kernel durations of one step
  (cell-list kernels + CN + force + extra for the large-box path; neighbor list + CN + force + extra for the small-box path), VER-0004 section 3.
- Scripts (`scripts/`): `bench6a.py` (per-kernel nsys table with grid/block/registers/local memory), `patch_variants.py` + `build_variants.sh` (timing-only
  variants, built in a scratch copy, the repository is never touched), `run_6a.py` (alternating repetitions), `scan_ns.py` (threads-per-atom scan),
  `ptxas_report.sh`, `run_s2.py` (D3 total base/new alternated 3x, uses VER-0012 `bench.py`), `accuracy.py`, `compare_ac2.py` (VER-0013/0014 copies).
  Systems from VER-0012/0013 `make_systems.py`: w41472, w5184, cu4000 (fcc Cu 10x10x10), clus972 (non-periodic water cluster).

## Stage 6a: diagnostic (no `src/` change)
Kernel times of HEAD (100 steps, median over steps; `results/base_kernels_*.json`), launch shape and registers (nsys `registersPerThread`, ptxas: 0 spills, 0 local memory):

| system | cell-list kernels (8) [ms] | CN [ms] | force [ms] | extra [ms] | D3 total [ms] | CN / extra: blocks x threads, threads per atom | regs CN / extra / force |
|---|---|---|---|---|---|---|---|
| water 41,472 | 0.033 | 0.693 | 4.718 | 0.741 | 6.198 | 2592 x 128, 8 | 50 / 63 / 79 |
| water 5,184 | 0.024 | 0.069 | 0.623 | 0.074 | 0.788 | 324 x 128, 8 | 50 / 63 / 79 |
| Cu 4,000 | 0.021 | 0.060 | 0.395 | 0.064 | 0.540 | 500 x 128, 16 | 50 / 63 / 79 |
| cluster 972 (non-periodic) | 0.019 | 0.276 | 0.824 | 0.305 | 1.425 | 16 x 64, 1 | 54 / 70 / 77 |

- Facts about the current code (found while reading, they change what item (vi) can still do): the large-box CN and extra kernels ALREADY use `num_sub`
  threads per atom (1..16) with the fixed-order shared-memory tree reduction (`reduce_sub_large_box`), added in DR-001 Phase 3; no `atomicAdd` exists in
  `dftd3.cu`. Candidates (1) and (2) of item (vi) therefore exist already. The only atomics of the large-box step are the integer `atomicAdd` of the cell-list
  kernels `find_cell_counts` / `find_cell_contents` (2 per atom per list, 4 per atom per step; 0.005-0.0085 ms per kernel at N = 41,472, no
  serialization problem; the order of `cell_contents` is the known run-to-run non-determinism of the large-box path).
- Candidates visited / accepted (r < rc_angular = 6 A, `count` variant): water 41,472 and 5,184: 382 / 84.6 per atom (0.22; 125 cells of 3 A); Cu 4,000: 276 / 78.0
  (0.28); cluster 972: 972 / 61.6 (0.06; the non-periodic box has 1 cell, so one thread loops over all atoms and the block count is 16 for 84 SMs).
- The CN and extra kernels use the same angular cell list, the same rc and the same pair test: 100 % of the neighbor traversal of the extra kernel repeats that
  of the CN kernel. (They cannot be merged into one kernel: the extra kernel needs `dc6_sum`/`dc8_sum` from the force kernel, which needs the CN of both atoms.)
- Threads-per-atom scan of CN / extra (`results/ns_scan.json`, [ms], 1 2 4 8 16): w41472 CN 0.806 0.813 0.743 0.696 0.704, extra 0.847 0.848 0.785 0.742 0.778;
  w5184 CN 0.245 0.138 0.086 0.067 0.070; Cu 4,000 CN 0.283 0.151 0.097 0.061 0.060. The rule of Phase 3 is at the optimum within about 3 % for both kernels (so
  no 1.2x from re-tuning, and the kernels are not thread-count limited at 41,472: 1 thread per atom is only 1.15x slower).

Timing-only variants (`results/variants_water.json`, `variants_cu_cluster.json`; 60 steps, 3 alternating repetitions, median; the variants change the results, only the
named kernel times are read). CN + extra [ms] and speedup versus base:

| variant (what it removes / does) | water 41,472 | water 5,184 | Cu 4,000 | cluster 972 |
|---|---|---|---|---|
| base | 1.446 | 0.141 | 0.124 | 0.625 |
| `cn_noexp` (exp in CN) | 1.516 (0.95x) | 0.146 (0.96x) | 0.136 (0.91x) | 0.609 (1.03x) |
| `cn_const` (exp, sqrt, division, type load of the CN pair term) | 1.504 (0.96x) | 0.149 (0.95x) | 0.132 (0.94x) | 0.558 (1.12x) |
| `ex_noexp` (exp in extra) | 1.450 (1.00x) | 0.139 (1.01x) | not run (hangs: unstable trajectory) | not run |
| `ex_const` (exp, sqrt, divisions, type load in extra) | 1.439 (1.01x) | 0.137 (1.03x) | not run | not run |
| `cn_nomic` (no minimum image) | 1.510 (0.96x) | 0.146 (0.96x) | 0.134 (0.93x) | 0.607 (1.03x) |
| `cn_xf32` (coordinate differences in float, CN) | 1.125 (1.29x) | 0.124 (1.13x) | 0.102 (1.22x) | 0.587 (1.06x) |
| `ex_xf32` (same, extra) | 1.218 (1.19x) | 0.132 (1.07x) | 0.112 (1.10x) | 0.592 (1.06x) |
| `fused` (CN kernel also does the extra pair work, stale dc sums; upper bound of a merged traversal) | 0.758 (1.91x) | 0.076 (1.85x) | 0.066 (1.89x) | 0.330 (1.89x) |
| `cache` (CN stores the accepted partner index, extra loops over that list; realizable) | 0.965 (1.50x) | 0.098 (1.44x) | 0.094 (1.32x) | 0.328 (1.91x) |
| `cachef` (list also stores float r12: more traffic) | 1.707 (0.85x) | 0.179 (0.78x) | - | - |

Reading: (a) the exponential / CN function evaluation is NOT a cost of these kernels (removing it never gains; the kernels are bound by the traversal: 3 double-precision
coordinate differences and a gather per visited candidate, 4.5 candidates per accepted pair); (b) FP64 coordinate differences are the largest single share
(`cn_xf32`: CN 0.70 -> 0.37 ms, extra 0.75 -> 0.52 ms at 41,472), but a float difference changes the numbers (not tried; see "Not done"); (c) repeating the traversal in the extra
kernel is the largest realizable waste (`cache` upper bound: -0.48 ms at 41,472, extra 0.75 -> 0.26 ms).

Gate 6b (DES-0001 (vi): upper bound >= 0.2 ms saved on CN + extra at water 41,472, or >= 1.15x on CN + extra at water 5,184): **MET** (`cache` -0.48 ms and 1.44x;
`fused` -0.69 ms and 1.85x; `cn_xf32` -0.32 ms). Stage 6b started.

## Stage 6b: items and gates
| item | state | gate | result |
|---|---|---|---|
| (1) CN kernel several threads/atom + fixed-order reduction | already in the code (Phase 3); not changed | kernel >= 1.2x | not reachable: scan above, current rule within 3 % of the best fixed value; nothing implemented |
| (2) extra kernel same | already in the code (Phase 3); not changed | kernel >= 1.2x | same |
| (3) share the traversal between CN and extra | implemented (accepted-partner list, see below) | extra kernel >= 1.2x; D3 total no case slower than 1.03x; AC-1; AC-2; determinism not worse | extra 0.760 -> 0.200 ms (3.8x) at water 41,472; no case slower; AC-1 met; AC-2 met; see below |

### Change (item (3))
`src/force/dftd3.cu`, `src/force/dftd3.cuh` (members `cn_list`, `cn_list_count`):
- `find_dftd3_coordination_number_large_box<num_sub>` writes the partner index of every accepted pair into `g_list[k * N * num_sub + slot]` (slot = `n1 * num_sub + sub`,
  k-th accepted partner of that thread in traversal order, coalesced across threads) while `k < list_capacity`, and stores the number of accepted partners in
  `g_list_count[slot]` (also when larger than the capacity). Only valid threads store (tail threads of the last block never write).
- `find_dftd3_force_extra_large_box<num_sub>`: the pair body is a lambda `add_pair(n2)`; if `g_list_count[slot] <= list_capacity` it loops over the stored partners, else it
  repeats the cell traversal for that thread exactly as before. Both visit the same pairs in the same order, so the result is the same number; the reduction (`reduce_sub_large_box`,
  called by every thread, no `__syncthreads` in a branch) is unchanged.
- `get_cn_list_capacity`: capacity per thread = `1.5 * expected_pairs / num_sub + 8`, expected_pairs from the mean density (N / box volume) and rc_angular; 0 (list
  off, plain traversal) if `N * num_sub * capacity * 4 B` > 256 MB; test aid `GPUMD_DFTD3_LARGE_BOX_LIST_CAPACITY` (0..4096) fixes it (not in the manual, prints once).
  A capacity that is too small costs time (overflowing threads repeat the traversal), never accuracy. Buffers grow only (the volume changes in NPT runs).
- Test aid `GPUMD_DFTD3_LARGE_BOX_NUM_SUB` (1, 2, 4, 8, 16) fixes the threads per atom of the three large-box kernels (not in the manual, prints once).
- Memory: `N * num_sub * capacity * 4 B` + `N * num_sub * 4 B`: water 41,472: 31.6 MB + 1.3 MB (capacity 25 per thread, max observed 25 per slot, 0 overflowing
  slots); water 5,184 4.0 MB; Cu 4,000 3.7 MB; cluster 972 0.5 MB (capacity 144, max 92). Registers: CN 50 -> 55, extra 63 -> 72 (0 spills).
- Overflow check (scratch counter, `D3OV`): 0 of 331,776 / 41,472 / 64,000 / 972 slots overflow at the default capacity for the four systems (first step; no claim for other densities).

### D3 total (`scripts/run_s2.py`, `results/s2.json`; base/new alternated 3x, median of medians, final code)
| system (path) | base [ms] | new [ms] | speedup | CN base -> new | extra base -> new |
|---|---|---|---|---|---|
| water 81 (small) | 0.097 | 0.097 | 1.00x | - | - |
| water 648 (small) | 0.502 | 0.504 | 1.00x | - | - |
| water 2,187 (small) | 0.686 | 0.691 | 0.99x | - | - |
| water 8,748 (small) | 8.125 | 8.126 | 1.00x | - | - |
| cluster 972 (large) | 1.523 | 1.191 | 1.28x | 0.296 -> 0.274 | 0.327 -> 0.068 |
| water 5,184 (large) | 0.769 | 0.739 | 1.04x | 0.067 -> 0.068 | 0.073 -> 0.030 |
| water 41,472 (large) | 6.346 | 5.853 | 1.08x | 0.712 -> 0.717 | 0.760 -> 0.200 |
| Cu 4,000 (large) | 0.541 | 0.507 | 1.07x | 0.060 -> 0.060 | 0.064 -> 0.030 |
No point is slower than 1.03x (small-box path: the code is untouched; differences are run noise, one repetition of w2,187 0.729 vs 0.686 in either build). An earlier
run of the same binary before the grow-only change gave water 41,472 6.321 -> 5.820 ms. **D3 total at water 41,472 is 5.85 ms (<= 6.0 ms: the DR-001 AC-4 target is met again).**

## Checks
- AC-1: `pytest -q test_dftd3_reference.py`: 18 passed (201 s) (14 existing, tolerances unchanged + 4 new). `scripts/accuracy.py` (`results/accuracy.txt`): small-box and
  benzene cases bit-identical to HEAD; large-box cases `cand - base` <= dE 5.4e-7 (cu_large, tolerance 1.2e-3), dF 1.7e-8 (tolerance 1e-6), per-atom virial about 1e-6, i.e.
  the same size as the run-to-run difference of HEAD against itself (4.7e-9 in force for w5184 between two runs of the base binary).
- New tests (`tests_pytest/test_dftd3_reference.py`, 65 s): `test_dftd3_large_box_neighbor_list_is_exact_and_reproducible` (24 atoms in a 40 A periodic box: large-box path, one
  warp, so the cell-list order is fixed; for `num_sub` in {1,2,4,8,16} x capacity {default, 0, 1, 4}: reproducible run to run and bit-identical energy/force/virial for list,
  overflow fallback and plain traversal; agreement of the `num_sub` values within the AC-1 caps); `test_dftd3_large_box_neighbor_list_capacities_agree` (benzene dimer,
  water_large, cu_large with capacity {default, 0, 1, 4}: AC-1 tolerance against the stored reference and mutual agreement within the AC-1 caps). The tests fail on the HEAD binary
  (the `threads per atom` message is missing), pass on the new one.
- Determinism: large-box runs with many warps are not bit-reproducible, for HEAD as well (cell-list order from `atomicAdd`, VER-0014); not worse now: the list preserves the
  traversal order, and the 24-atom system is bit-reproducible for every setting.
- AC-2: `run_regression.py --suite full --case 'dftd3_*'` (baseline `master-base`): the strict manifest tolerance fails 2 of 3 as for any round-off change
  (`results/regression_dftd3.log`); under the AC-2 rule (rtol 1e-5, atol 1e-12, `scripts/compare_ac2.py`, `results/ac2_dftd3.txt`) all pass: largest thermo.out rel 1.5e-6
  (`dftd3_kspace_one_each`), restart.xyz rel 1.6e-7.
- compute-sanitizer memcheck (3 MD steps): 24 atoms (num_sub 1, 16, with and without list, capacity 1 and 0), 20 atoms (tail threads: num_sub 16; num_sub 8 capacity 2), cluster 972:
  0 errors. A 200-step `npt_scr` run on water 5,184 (changing volume) completes with the new binary.
- Build: `make CUDA_ARCH="-arch=sm_86" -j4` warning-clean for `dftd3.cu` (an unrelated `eam_alloy.cu` warning exists). clang-format: the changed regions are clean (three pre-existing
  differences of the file, an include order and one continuation indent, were left alone).

## Not done / limits
- `--suite full`, NVE energy conservation (VER-0011 protocol), independent numerical review, HIP build and other GPUs are not run (Stage 3 of the plan). The extra kernel with `num_sub = 1` and
  cell traversal fallback is covered by the bit-identity test only for the 24-atom system; the many-warp systems are checked to round-off only.
- Larger systems: the list is off above 256 MB; at the default density rule this happens for N above about 4.7 x 10^5 atoms (1 thread per atom for N >= 10^5, about 540 B per atom for water at rc_angular = 6 A); there the old traversal runs (no gain, no loss).
- Capacity is a density estimate; strongly inhomogeneous systems overflow more slots and gain less (never lose accuracy).
- Not tried (outside the listed candidates of (vi)): FP64 coordinate differences (`cn_xf32`, `ex_xf32` upper bounds above) could be replaced by a float pre-filter with the double
  difference recomputed for the accepted 22 % only (same results); the non-periodic cluster (1 cell, 16 blocks of 64 threads, 972 candidates per atom per thread) would gain from splitting the
  atoms of the cell over threads (CN 0.28 ms, extra after the list 0.07 ms, force 0.82 ms are latency bound at N = 972). Both are left for the human to decide.
