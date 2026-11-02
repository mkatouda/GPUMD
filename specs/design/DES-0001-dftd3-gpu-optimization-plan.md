# DES-0001: Work plan for GPU optimization of the GPUMD DFT-D3 implementation

- Status: Approved by the human on 2026-11-03 (plan only); `DR-001` and `ADR-0002` still need approval before any `src/` change
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
(one thread per atom, block size 64). Not yet measured: kernel-level breakdown.

Candidate causes (from reading the code, NOT yet verified by profiling):
- A. Thread-per-atom parallelism too coarse for small/medium N (`dftd3.cu` launches
  `(N-1)/64+1` blocks of 64).
- B. C6 interpolation recomputes up to `num_cn_1 x num_cn_2` exponentials per pair
  (`dftd3.cu:143-175`, large-box twin from line 559); the Gaussian weight factorizes into
  per-atom factors, so it can be computed once per atom per reference CN.
- C. Full neighbor list: each pair is evaluated from both atoms.
- D. Neighbor/cell lists rebuilt every step; about 10 kernel launches per step; small-box path
  stores `N x MN(=10000) x 6` floats of pair vectors.
- E. D3 runs after NEP on the same stream (no overlap).

## 3. Acceptance criteria (apply to every phase)
Numerical (against the same simple-dftd3 references as VER-0001, same-cutoff):
- for every VER-0001 system and cutoff set, |dE|, max|dF|, max|dVirial| are each no worse than
  2x the VER-0001 value and never above 1e-5 eV, 1e-5 eV/A, 1e-4 eV respectively;
- the `tests_pytest` suite (738 passed / 480 skipped baseline) and
  `run_regression.py --suite full` (218 passed, the same 37 known non-deterministic failures,
  25/25 relations) show no new failure; the `dftd3_*` bugfix cases keep passing;
- energy conservation in a short NVE run with D3 is not worse than before.

Performance (VER-0001 slope method, 100 vs 500 steps, same machine):
- target D3 extra time: N=81: <= 1.0 ms; N=5,184: <= 2.0 ms; N=41,472: <= 6.0 ms
  (12/6 A); no case slower than baseline by more than 3%.
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
### Phase 0: profile (no source change)
- `nsys` (and `ncu` for the top kernels) from `/usr/local/cuda/bin`.
- Cases: water (small-box path at N=81, 648; large-box path at N=5,184, 41,472), graphite
  (C), an element-rich/metal case such as fcc Cu (up to 5x5 reference CNs); cutoffs 12/6 and 25/15.
- Record per kernel: time share, achieved occupancy, registers, memory vs compute bound;
  split D3 time into neighbor/cell build, CN, force.
- Gate: continue only if D3 is >= 20% of a NEP+D3 step in at least one typical case and the
  force kernel is the dominant D3 cost. Output: `VER-0002` record. Decides which of B-E matter.

### Phase 1: specs and reference test (no `src/` change)
- `DR-001`, `ADR-0002`, `VER-0002` benchmark protocol, pytest with stored references,
  `traceability.yaml` entries. Gate: human approval; test passes on unmodified code.

### Phase 2: factorized C6 weights (cause B)
- Per-atom precompute of `L_i(n) = exp(-4 (cn_n - cn_ref_i)^2)` and its derivative terms;
  pair loop uses products only. Applies to both small-box and large-box kernels.
- Gate: accuracy criteria met; speed-up >= 1.2x on the element-rich case. Otherwise drop.

### Phase 3: parallelism for small and medium N (cause A)
- Warp-per-atom (or pair-parallel) force and CN kernels; keep the thread-per-atom path for
  large N if it is faster; choose by N at run time.
- Gate: >= 2x at N=5,184 (12/6) and no slowdown at N=41,472. Otherwise drop.

### Phase 4: conditional extras (causes C, D, E; only those Phase 0 justified)
- C: half neighbor list (needs atomics; keep only if measured faster).
- D: Verlet list with skin and rebuild on displacement; fewer launches; reduce the
  `N x MN` small-box storage.
- E: run D3 on a second stream concurrently with NEP.
- Gate for each: the targeted part is >= 30% of D3 time (D, C) or D3 time at N=81 is latency-bound (E).

### Phase 5: integration and release
- Check all callers (`NEP`, `NEP charge`, `NEP charge vdw`, `NEP multigpu` documented as
  unsupported), `--suite full` against a `master` baseline built with `sm_86`, update
  `doc/gpumd/input_parameters/dftd3.rst` (performance tips, cutoff cost).
- Re-measure VER-0001 tables as `VER-0003` (before/after).
- Upstream: cherry-pick only code, tests and `doc/` changes (no `specs/`, `.claude/`,
  process rules); open the PR from a branch based on `upstream/master`.

## 6. Risks
- Results change at round-off level (float summation order); `--update-golden` must not be
  used to hide a real change.
- Double-precision `atomicAdd` on `sm < 60` has a hand-written fallback (`dftd3.cu:84`);
  keep it working.
- The small-box path (`MN = 10000`) and large-box path are separate code; every change needs
  both covered by tests (VER-0001 systems include both: N=81 small box, N=5,184 large box).
- Run-to-run non-determinism already exists in the suite (37 known cases); a new
  non-deterministic D3 result must be distinguished from those.
- Results measured only on one GPU (sm_86); other architectures (and HIP) are unverified.
- CUDA/HIP: any new CUDA-only construct must go through `gpu_macro.cuh`.

## 7. Estimated effort and order
Phase 0 and 1 first (about one working session), then Phases 2 and 3 as separate delegations;
Phase 4 only if Phase 0 supports it; Phase 5 once. Each phase ends with a completion report
(commands, results, tolerances, limitations).
