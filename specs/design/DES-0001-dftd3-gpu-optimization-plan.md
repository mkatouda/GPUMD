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
