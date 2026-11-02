# DR-001: Faster GPU DFT-D3 with unchanged results

- Status: Proposed (drafted 2026-11-03; needs human approval)
- Design: `specs/design/DES-0001-dftd3-gpu-optimization-plan.md`
- Decision: `specs/decisions/ADR-0002-dftd3-optimization-numerics.md`
- Baseline: `specs/verification/VER-0001-dftd3-vs-simple-dftd3/`

## Requirement
The `dftd3 <functional> <potential_cutoff> <coordination_number_cutoff>` keyword shall run
faster on GPU than the baseline recorded in VER-0001, while its energy, forces, and virial
stay equal to an independent two-body D3(BJ) implementation within the tolerances below.

## Scope
In: `src/force/dftd3.cu`, `src/force/dftd3.cuh`, and their callers (`NEP`, `NEP charge`,
`NEP charge vdw`; `NEP multigpu` is documented as unsupported and must not regress).
Out: new damping schemes, the ATM three-body term, new functionals, keyword syntax or
defaults, and the cube-law dependence of the cost on the potential cutoff.

## Acceptance criteria
AC-1 (accuracy). For every VER-0001 system (benzene dimer, graphite, water27 cubic and
triclinic) and both cutoff sets (12/6 and 25/15 A), compared with simple-dftd3 using the same
cutoffs, |dE|, max|dF|, and max|dVirial| are each no worse than 2x the corresponding VER-0001
value, and never above 1e-5 eV, 1e-5 eV/A, and 1e-4 eV.

AC-2 (no regression). `pytest -m "not slow"` in `tests_pytest/` shows no new failure
(baseline: 738 passed, 480 skipped); `run_regression.py --suite full` shows no failure other
than the 37 known non-deterministic cases listed in `AGENTS.md`, and all 25 relations pass;
the `dftd3_*` bugfix cases pass.

AC-3 (energy conservation). A short NVE run with D3 shows an energy drift no worse than the
baseline binary on the same system.

AC-4 (performance). With the VER-0001 slope method (100 vs 500 steps), `dftd3 pbe 12 6`,
NEP89, water box, RTX A6000: D3 extra time per step is at most 1.0 ms at N=81, 2.0 ms at
N=5,184, and 6.0 ms at N=41,472, and no case is slower than baseline by more than 3%.
Targets for a phase may be dropped by the human if a phase gate in DES-0001 fails.

AC-5 (portability). The code builds warning-clean with CUDA; any new construct goes through
`gpu_macro.cuh` so the HIP build is not broken; the `sm_60`-and-older `atomicAdd` fallback
keeps working.

AC-6 (docs). `doc/gpumd/input_parameters/dftd3.rst` states the cost scaling with the cutoff.

## Verification
- Accuracy: new pytest with stored simple-dftd3 reference values (provenance as in VER-0001),
  plus a re-run of the VER-0001 scripts.
- Performance: VER-0001 slope method, recorded as VER-0002 (profile) and VER-0003 (before/after).
- Regression: `--suite full` against a `master` baseline built for the same GPU architecture.
