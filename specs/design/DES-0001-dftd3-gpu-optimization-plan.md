# DES-0001: Work plan for GPU optimization of the GPUMD DFT-D3 implementation

- Status: Approved by the human on 2026-11-03 (plan only); revised 2026-11-03 after Phase 0 (`VER-0002`), revision approved by the human.
  `DR-001` and `ADR-0002` approved by the human on 2026-11-03 (Phase 1 gate met; `src/` changes may start)
- Date: 2026-11-03
- Basis: `specs/verification/VER-0001-dftd3-vs-simple-dftd3/` (accuracy and timing baseline)
- Target code: `src/force/dftd3.cu`, `src/force/dftd3.cuh`; callers `nep.cu`, `nep_charge.cu`,
  `nep_charge_vdw/`, `nep_multigpu.cu`

## 1. Goal and non-goals
Goal: make `dftd3 <functional> <rc> <rc_cn>` faster on GPU without changing results beyond
the single-precision level already measured in VER-0001.

Non-goals: new damping schemes, ATM three-body term, new functionals, changing the keyword
syntax or defaults, changing the cutoff scaling (cost grows with the cube of `rc`; that is
algorithmic and stays).

## 2. Baseline facts (VER-0001, RTX A6000, NEP89, water box, `dftd3 pbe 12 6`)
| N | D3 extra (ms/step) | per atom (us) |
|---|---|---|
| 81 | 2.5 | 31 |
| 5,184 | 4.4 | 0.86 |
| 41,472 | 8.0 | 0.19 |
| 5,184 at 25/15 | 40.7 | 7.9 |

Reading: 8x more atoms costs only 1.8x more time, so the GPU is under-filled at N <= ~10^4
(one thread per atom, block size 64).

Phase 0 result (`VER-0002`, kernel-level `nsys` profile): D3 is 56-96% of the GPU step time.
Two code paths exist (`dftd3.cu:265-284`); the small-box path is used when any periodic
thickness is <= 2.5 x `rc_potential` (cells up to 30 A at rc=12 A):
- large-box path: the force kernel is 80-87% of D3 time; time per pair is 0.89 ns at
  N=5,184 vs 0.22 ns at N=41,472 (under-filled GPU); cell-list build is negligible;
- small-box path: building the explicit neighbor list is 44-65% of D3 time (as much as or more
  than the force kernel), e.g. 2.83 of 4.37 ms at N=648.

Candidate causes (from reading the code, NOT yet verified by profiling):
- A. Thread-per-atom parallelism too coarse for small/medium N (`dftd3.cu` launches
  `(N-1)/64+1` blocks of 64).
- B. C6 interpolation recomputes up to `num_cn_1 x num_cn_2` exponentials per pair
  (`dftd3.cu:143-175`, large-box twin from line 559); the Gaussian weight factorizes into
  per-atom factors, so it can be computed once per atom per reference CN.
- C. Full neighbor list: each pair is evaluated from both atoms.
- D. Neighbor lists rebuilt every step; about 10 kernel launches per step; small-box path
  stores `N x MN(=10000) x 6` floats of pair vectors. Confirmed first-order for the small-box
  path by Phase 0 (44-65% of D3 time); negligible for the large-box path.
- E. D3 runs after NEP on the same stream (no overlap).

## 3. Acceptance criteria (apply to every phase)
Numerical (against the same simple-dftd3 references as VER-0001, same-cutoff):
- for every reference case of `tests_pytest/test_dftd3_reference.py` (both code paths): |dE| <=
  2e-6 x |E_D3|, max|dVirial| <= 2e-6 x max|Virial_ref|, max|dF| <= 1e-5 eV/A, and each error
  no worse than max(2 x the error measured on the unmodified code, a float-noise floor)
  (`DR-001` AC-1 as revised on 2026-11-03: relative limits for energy and virial);
- the `tests_pytest` suite (738 passed / 480 skipped baseline) and
  `run_regression.py --suite full` (218 passed, the same 37 known non-deterministic failures,
  25/25 relations) show no new failure; the `dftd3_*` bugfix cases keep passing;
- energy conservation in a short NVE run with D3 is not worse than before.

Performance (VER-0001 slope method, 100 vs 500 steps, same machine):
- target D3 extra time: N=81: <= 1.0 ms; N=5,184: <= 2.0 ms; N=41,472: <= 6.0 ms
  (12/6 A); no case slower than baseline by more than 3%.
- added after Phase 0 (to be added to `DR-001` AC-4 on approval): N=648 (small-box path)
  <= 2.0 ms (baseline 4.37 ms D3 total in `VER-0002`); the fcc Cu case (N=4,000, 12/6 A,
  baseline 3.65 ms) <= 2.0 ms.
- required acceptance case set: both paths, namely small box (water N=81, 648) and large box
  (water N=5,184, 41,472), plus fcc Cu, plus graphite (small box, C), 12/6 and 25/15 A.
Targets are goals, not promises; a phase is dropped if its gate fails (Section 5).

## 4. Process rules for this work (from AGENTS.md)
- Requirements and ADR first: new `DR-001` (results unchanged within the criteria above,
  faster) and `ADR-0002` (D3 results may change at float round-off level; defines the
  tolerances above). Both need human approval.
- Test first: add a pytest that compares D3 energy/forces/virial with stored simple-dftd3
  reference values (provenance recorded as in VER-0001). It must pass on the current code
  before any optimization.
- One phase = one `gpumd-implementer` delegation in a worktree; one GPU-heavy agent at a time;
  parent re-runs build/tests/diff itself; then fresh reviewers (`numerical-gpu-reviewer`,
  `api-compat-reviewer`; `physics-dielectric-reviewer` only if formulas change).
- One atomic commit per phase with requirement IDs; no commit without being asked.
- Benchmarks and run directories go to a scratch directory, never the repository.

## 5. Phases and gates (cheapest first; stop and report if a gate fails)
### Phase 0: profile (DONE; gate PASSED; record `VER-0002`)
- `nsys` kernel-level profile of 7 cases on both paths; `ncu` counters are unavailable
  (`ERR_NVGPUCTRPERM`) and, by the owner's security policy, must not be enabled with admin
  rights; occupancy statements are estimates from register and thread counts, and every
  phase gate is judged with `nsys` times and the accuracy tests only.
- Gate (D3 >= 20% of a step and force kernel dominant): passed; see Section 2 for the findings
  that changed this plan (small-box neighbor list promoted to its own phase).

### Phase 1: specs and reference test (no `src/` change)
- `DR-001`, `ADR-0002` (drafted), `VER-0002` (done), pytest with stored simple-dftd3
  reference values covering the required case set (both paths), `traceability.yaml` updates.
- Gate: human approval of `DR-001` and `ADR-0002`; the new test passes on unmodified code.

### Phase 2: factorized C6 weights (cause B), both paths
- Per-atom precompute of `L_i(n) = exp(-4 (cn_n - cn_ref_i)^2)` and its derivative terms;
  the pair loop uses products only. Applies to `add_dftd3_force_small_box` and
  `find_dftd3_force_large_box` (and the extra-force kernels).
- Gate: accuracy criteria met; speed-up >= 1.2x of the force kernel on the Cu case. Else drop.
- Status (2026-11-03): ON HOLD by human decision. Prototype (patch kept outside the repo,
  worktree `agent-aed059ade4cc294d7`) passes AC-1 but gave 1.04x on Cu 4,000 (gate 1.2x),
  1.14x water 5,184, 1.27x water 648, 1.22x water 81. Probable cause: the large-box kernel is
  bound by thread count and cell-list traversal, not by exp. Re-evaluate on top of Phase 3.

### Phase 3: parallelism of the force kernels (cause A), large-box path first
- Warp-per-atom (or pair-parallel) force and CN kernels; keep thread-per-atom where faster
  (choose by N at run time). Target: bring the per-pair time at N=5,184 toward the 0.22 ns
  measured at N=41,472.
- Gate: >= 2x on the force kernel at N=5,184 (12/6) and no slowdown at N=41,472. Else drop.

### Phase 4: small-box neighbor list (cause D) - promoted by Phase 0
- Re-design the small-box neighbor-list build (`find_neighbor_list_small_box` in `dftd3.cu`,
  explicit `N x MN` pair vectors, thread per atom): parallelize over atoms x images or pairs,
  reduce the stored data (e.g. recompute pair vectors), and consider reuse across steps
  (Verlet list with skin, rebuild on displacement).
- Gate: neighbor-list kernel time at N=648 (12/6) reduced by >= 2x; accuracy criteria met;
  memory use not larger than baseline. Else drop.

### Phase 2': re-evaluation of the factorized C6 weights on top of Phases 3 and 4 (draft 2026-11-03)
- Precondition: Phases 3 and 4 merged and verified (accuracy tests, `dftd3_*` regression at
  rtol 1e-5 / atol 1e-12, `--suite full` against a `master` baseline, nsys re-measurement).
- Why: the Phase 2 prototype gave 1.04x on Cu 4,000 only because the large-box kernel was bound by
  thread count and cell-list traversal. After Phase 3 the force kernel is 0.44 ms (Cu 4,000) and
  0.79 ms (water 5,184), so the per-pair exp work may be a larger share. The prototype patch
  (`phase2.patch`) cannot be applied mechanically: Phase 3 rewrote the large-box kernels.
  The small-box part (1.22-1.27x) is expected to apply.
- Stage 1 (cheap diagnostic, no code): with nsys, take the share of the force kernel time spent in
  the C6 double loop for Cu 4,000, water 5,184 and water 648 (e.g. by timing a build with the
  C6 weights replaced by a constant; scratch only, results not used). Gate: if the C6 loop is
  < 15% of the force kernel in all three, stop and keep Phase 2 dropped.
- Stage 2: re-implement the per-atom weights (`cn_weight`, `cn_dweight`, computed in the CN
  kernels) on the merged code for both paths. Gate (as before): accuracy criteria met and
  force kernel >= 1.2x on Cu 4,000 versus the merged baseline; otherwise drop.
- Stage 3: accuracy (reference test, `dftd3_*` regression, num_sub forced 1..16 comparison) and the
  memory cost of two N x max_cn float arrays (report the increase).

- Outcome (2026-11-03, stage 1 measured; stage 2 not implemented): RTX A6000 sm_86, base `cc5129d2`,
  nsys, 100-step NVE, median of 7 alternating runs, C6 replaced by a constant (upper bound).
  Share of the force kernel in the C6 loop: Cu 4,000 13.8% (0.4398 -> 0.3792 ms, upper bound 1.16x),
  water 5,184 43.9% (0.781 -> 0.438 ms, 1.78x), water 648 34.1% (1.418 -> 0.934 ms, 1.52x).
  Other kernels unchanged (within 0.002 ms). The stage 2 gate (>= 1.2x on Cu 4,000) is unreachable
  (needs a C6 share >= 16.7%), so Phase 2' is DROPPED under the gate as written (human decision
  2026-11-03: keep the Cu 4,000 gate). Water systems would benefit (upper bounds above); revisit only
  if the gate is changed by a human decision. Results are diagnostic only (scratch variant, not merged).
  Re-opened 2026-11-04 by a human decision (gate change) as item (v) of Phase 7, see there.

### Phase 5: conditional extras (causes C, E; only if still justified after Phases 2-4)
- C: half neighbor list (needs atomics; keep only if measured faster).
- E: run D3 on a second stream concurrently with NEP (only if D3 is latency-bound at small N).
- Gate: each item must show a measured gain >= 10% of the D3 total on the case set.

### Phase 6: integration and release
- Check all callers (`NEP`, `NEP charge`, `NEP charge vdw`, `NEP multigpu` documented as
  unsupported), `--suite full` against a `master` baseline built with `sm_86`, update
  `doc/gpumd/input_parameters/dftd3.rst` (performance tips, cutoff cost, small vs large box).
- Re-measure `VER-0001` tables and the `VER-0002` profile as `VER-0003` (before/after).
- Upstream: cherry-pick only code, tests and `doc/` changes (no `specs/`, `.claude/`, process
  rules); open the PR from a branch based on `upstream/master`.

### Phase 7: small-box path and largest-system performance (draft 2026-11-03; needs human approval)
- Why: after Phases 3 and 4 (VER-0004 section 3) AC-4 is unmet at N = 81 (D3 total 1.46 ms vs 1.0) and
  N = 41,472 (6.96 ms vs 6.0). On the small-box path the force kernel is about 1.4 ms at N = 81, 648 and
  972 (non-periodic cluster 1.0x), i.e. independent of N: the GPU is under-filled or latency-bound. The
  DFT-D2 work (DES-0002 Phase 6) gained 15x at N = 648 from a cost-based cell count, several threads per
  atom with a fixed-order reduction, and a cell-centre-relative float candidate loop; the D3 small-box
  path has not had the equivalent treatment (Phase 3 `num_sub` covers the large-box path).
- Constraints: AC-1 tolerances (DR-001) unchanged and met; AC-2 `dftd3_*` tolerance rule unchanged; results
  stay deterministic where they are now; CUDA/HIP via `gpu_macro.cuh`; default numerics unchanged.
- Stage 1 (cheap diagnostic, no code in `src/`; scratch variants and nsys only; ncu counters are not
  available): on water 81, 648, cluster 972 and water 41,472 at `dftd3 pbe 12 6`,
  (a) kernel time versus N at fixed rc for the small-box force kernel (fill/latency test: time expected from
  the large-N throughput versus measured), (b) registers, spills and occupancy from the compiler report,
  (c) candidate pairs visited versus accepted pairs (r < rc) per atom and the explicit-list length versus the
  `MN = 10000` limit, (d) share of the force kernel spent in the C6 interpolation exponentials (variant with
  the interpolation replaced by a constant, timing only), (e) share of FP64 work in the pair loop (variant
  with float coordinate differences, timing only; accuracy not checked), (f) launch/latency floor: time of the
  empty-kernel sequence of one D3 step (neighbor, CN, force, extras).
  Gate for continuing to Stage 2: at least one of (a) measured time >= 2x the time expected from the large-N
  throughput at N <= 648, (d) share >= 25%, (e) float variant >= 1.2x, (f) launch/latency floor >= 40% of the
  D3 total at N = 81. If none holds, stop and record AC-4 at N = 81 and 41,472 as unmet.
- Stage 2 (candidates, each separately, in this order, each with its own gate): (i) several threads per atom with
  a fixed-order reduction on the small-box force kernel (as in DR-001 Phase 3 and DES-0002 Phase 6), (ii) a
  cost-based choice of the cell/list parameters for the small-box path, (iii) cell-centre-relative float
  candidate loop (only with an accuracy study against the AC-1 tolerances: gate 1/3 of the tolerance on all
  cases), (iv) fusing kernels of one D3 step to cut launches (only if (f) shows a latency floor).
  Gate per item: >= 1.2x on the D3 total at N = 81 or 648 (or >= 1.1x at N = 41,472 for items affecting the
  large-box path), no case slower than 1.03x, AC-1 tolerances met, `dftd3_*` cases within the AC-2 tolerance,
  warning-clean build, determinism not worse than now.
- Stage 1 outcome (2026-11-04, VER-0012, nsys, RTX A6000 sm_86): gate passed via (a) and (d); (e) and (f) not met.
  (a) the small-box force kernel takes 1.34-1.44 ms for N = 81..972, independent of N (101x / 12.6x the time
  expected from the large-N throughput at N = 81 / 648): one thread per atom, 2 / 11 blocks of 64, about 740
  accepted pairs per atom processed serially. (b) 48 registers, no spills. (c) accepted/visited 0.14-0.33; the
  `MN = 10000` limit is reached only at rc about 24-30 A (margin 1.7x for diamond C at rc = 20 A). (d) C6
  interpolation share of the force kernel: 47% (N = 81), 38% (648), 34% (2,187), 12% (8,748), 59% (cluster 972,
  large-box path), 23.5% (water 41,472). (e) float variants of the pair loop: <= 1.14x at N <= 648. (f) launch gaps
  about 2 us per step (0.4%). The explicit-list build (NL) grows as N^2 and exceeds the force kernel above
  N about 2,900. The 972-atom non-periodic cluster uses the large-box path, not the small-box path.
- Stage 2 scope (human decision 2026-11-04, replaces the list above): (i) several threads per atom with a
  fixed-order reduction on the small-box force kernel is done first; (ii) cell/list parameters of the small-box
  path is decided after (i) from its measured result; (iii) float candidate loop and (iv) kernel fusion are
  dropped from Stage 2 (Stage 1: <= 1.14x at N <= 648, launch floor 0.4%). (v) NEW, the factorized C6 weights of
  Phase 2' are re-evaluated, delegated after (i) (same file, GPU): the C6 interpolation weight separates as
  L_ij = exp(-4 diff_i^2) exp(-4 diff_j^2); per-atom tables remove most exponentials in the pair loop.
  Gate change (human decision 2026-11-04): the Phase 2' gate (>= 1.2x force kernel on Cu 4,000, which was
  unreachable: upper bound 1.16x) is replaced for (v) by: D3 total at N = 41,472 <= 6.0 ms (DR-001 AC-4 target;
  the measured variant upper bound is 5.59 ms) AND >= 1.2x on the D3 total at water 648 or 5,184, AND no case
  (including Cu 4,000) slower than 1.03x, AC-1 tolerances unchanged and met, `dftd3_*` cases within the AC-2
  tolerance, determinism not worse than now, warning-clean build, memory increase of the per-atom tables reported.
  The results change at round-off level; the AC-1 and AC-2 rules above apply unchanged.
- Stage 2 outcome (2026-11-04). (i) adopted (VER-0013): small-box D3 total N = 81 1.516 -> 0.101 ms (15.1x),
  N = 648 1.873 -> 0.487 ms (3.84x), N = 2,187 2.380 -> 0.793 ms (3.0x); large-box unchanged; AC-1 tolerances
  met; `dftd3_*` within the AC-2 tolerance; the double `atomicAdd` of the small-box kernels was removed (no
  atomics left in `dftd3.cu`, so the `sm < 60` fallback was deleted). (ii) put on hold by human decision
  (not needed for AC-4). (v) measured (VER-0014): D3 total 6.931 -> 6.316 ms at water 41,472 (1.10x; the gate
  <= 6.0 ms is NOT met; the C6-constant upper bound 5.59 ms is not reachable because the C6 reference-table
  double sum stays in the pair loop and registers rise 72 -> about 80), 1.21x at water 5,184, 1.23x at
  cluster 972, 1.02-1.08x elsewhere, no case slower; AC-1/AC-2 met; memory +20 B/atom. Human decision
  2026-11-04: adopt (v) although the N = 41,472 gate is not met; the AC-4 target at N = 41,472 is recorded as
  unmet and dropped (DR-001 AC-4 allows this). Stage 3 below applies to the merged result (the AC-3 NVE
  closure of 2026-11-03 predates (i) and (v) and must be repeated).
- Item (vi) NEW (human instruction 2026-11-04, independent of the adoption decision for (v)): speed up the
  coordination-number (CN) kernel and the extra (angular/three-index-free CN-derivative) kernel of the large-box
  path. Motivation: at water 41,472 the CN kernel takes 0.71 ms and the extra kernel 0.75 ms of the 6.32 ms D3
  total (VER-0014); the force kernel is 4.81 ms. Stage 6a (diagnostic, no `src/` change; nsys and the
  compiler report only): for water 41,472, water 5,184, Cu 4,000 and cluster 972 at `dftd3 pbe 12 6`, per kernel
  (CN, extra, and the cell-list kernels): time, registers/spills, threads per atom and blocks, accepted/visited
  candidates, share of the time in exp/CN function evaluation (constant-replacement timing variant), the
  atomic operations (counts and whether they serialize), and how much of the neighbor traversal is repeated
  between the CN and extra kernels (a fused or cached traversal is a candidate). Gate for 6b: a measured
  upper bound (timing variant) of at least 0.2 ms saved on CN + extra at water 41,472, or >= 1.15x on CN + extra
  at water 5,184. Stage 6b (implementation, only after the gate): candidates in this order, each with its own gate
  (>= 1.2x on the kernel and no case slower than 1.03x on the D3 total; AC-1 tolerances and the AC-2 `dftd3_*`
  tolerance met; determinism not worse than now): (1) several threads per atom with a fixed-order reduction for
  the CN kernel, (2) the same for the extra kernel, (3) sharing the neighbor traversal between the CN and extra
  kernels (only if the 6a data support it). If the resulting D3 total at water 41,472 is <= 6.0 ms, the AC-4
  target there is met again (the human decision of 2026-11-04 to drop it remains valid but is then moot).
  Stage 3 below is run after (vi), once, on the final code.
- Item (vi) outcome (2026-11-04, VER-0015): 6a gate passed (index-cache upper bound -0.48 ms at water 41,472, 1.44x at
  water 5,184). Items (1) and (2) already exist (Phase 3: num_sub 1..16, fixed-order reduction; a num_sub scan found the
  current rule within 3% of the best fixed value), so nothing was changed there. Item (3) adopted: the CN kernel stores the
  accepted partners, the extra kernel reuses the list (fallback to cell traversal on overflow; identical pair order).
  Extra kernel 0.76 -> 0.20 ms; D3 total water 41,472 6.346 -> 5.853 ms (nsys kernel sum; the AC-4 slope method gives 6.19 ms in VER-0018, i.e. NOT met, target stays dropped), water 5,184
  1.04x, Cu 4,000 1.07x, cluster 972 1.28x, small box unchanged; AC-1 and AC-2 met. Candidate not yet decided (human):
  float pre-filter of the CN/extra candidate loops (upper bound CN 0.70 -> 0.37 ms, extra 0.75 -> 0.52 ms).
- Item (vii) NEW (human instruction 2026-11-04): float pre-filter in the large-box CN kernel (and, if it helps, the
  extra kernel's fallback traversal): candidate differences in float relative to the cell centre (as D2 Stage 2b), the
  r < rc test in float with a guard band, and the exact double distance recomputed only for pairs that pass or lie within
  the band, so that the accepted pair set and the double arithmetic of accepted pairs are unchanged. Gate: accepted set
  identical to HEAD on all test systems (count check), AC-1/AC-2 unchanged, CN kernel >= 1.2x at water 41,472 and
  5,184 and no case slower than 1.03x on D3 total, determinism not worse; otherwise revert and report.
- Item (vii) outcome (2026-11-04, VER-0016): S1 gate NOT met, rejected, no src change kept. A float pre-filter with exact double
  fallback made the CN kernel slower (water 41,472 0.93x, water 5,184 0.86-0.92x): about 22% of candidates are accepted, so
  almost every warp (1-0.78^32 > 99.9%) still executes the double path. The timing-only upper bound (CN 0.71 -> 0.37 ms)
  needs warp-level compaction of accepted candidates or a float-only decision (changes results); not pursued.
- Stage 3 outcome so far (2026-11-04): independent numerical review: no BLOCKER/MAJOR (MINOR fixed in VER-0019: capacity clamp, 25% list
  headroom, tests for N~130 with num_sub 64/128 and NPT list growth). S1 extreme CN (VER-0017): pre-existing upstream float weight
  underflow found, out of scope (UPSTREAM-0001). S2-S4 (VER-0018): AC-4 by slope method met at N=81/648/5,184/Cu 4,000, not met at
  N=41,472 (6.19 ms, target dropped); `--suite full` only known non-deterministic cases plus strict-tolerance dftd3_* (AC-2 met) and
  dftd2_* cases that master cannot run; pytest -m "not slow" 859 passed, 0 failed. NVE repeat (VER-0020, final code 0c353281, VER-0011 protocol B, 70 runs, no NaN/crash): water 648 (20 seeds, 10 ps), water 5,184 (10 seeds, 5 ps),
  Cu 4,000 (5 seeds, 10 ps): std, |drift| and max-min indistinguishable from master (all p >= 0.125, worst ratio 0.81-1.00); water 5,184 median
  ratios 1.27 (std) and 1.45 (|drift|) with CIs up to 2.07 and 4.27, so degradation of that size is not excluded. Phase 7 and Stage 3 are complete.
- Stage 3: re-measure VER-0004 section 3 and the VER-0001 slope method, `--suite full` against the `master`
  baseline, independent numerical review of the changed kernels, NVE with the VER-0011 protocol (single
  trajectory is not enough; see VER-0011), then record AC-4 per target point (met/unmet) and let the human
  decide whether remaining targets are dropped (DR-001 AC-4 allows this).

## 6. Risks
- Results change at round-off level (float summation order); `--update-golden` must not be
  used to hide a real change.
- Double-precision `atomicAdd` on `sm < 60` has a hand-written fallback (`dftd3.cu:84`);
  keep it working.
- The small-box path (`MN = 10000`) and large-box path are separate code; every change needs
  both covered by tests (VER-0001/VER-0002 systems include both: N=81 and 648 small box,
  N=5,184 and 41,472 large box).
- Run-to-run non-determinism already exists in the suite (37 known cases); a new
  non-deterministic D3 result must be distinguished from those.
- Results measured only on one GPU (sm_86); other architectures (and HIP) are unverified.
- CUDA/HIP: any new CUDA-only construct must go through `gpu_macro.cuh`.

## 7. Estimated effort and order
Phase 0 is done; Phase 1 next (about one working session). Phases 2, 3 and 4 are independent
and are delegated one at a time (Phase 4 matters most for cells <= 30 A, Phases 2-3 for larger
cells); Phase 5 only if still justified; Phase 6 once. Each phase ends with a completion report
(commands, results, tolerances, limitations).
