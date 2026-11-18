# VER-0039: element sweep of dftd3 (zero damping, BJ) and dftd2 against independent references, and HIP check, before a pull request

- Status: measured 2026-11-18, RTX A6000 (sm_86), CUDA 12.3, binary = develop 150a3189 (ADR-0008 applied). Type: verification (accuracy against independent references) + static HIP check. Requirements DR-001, DR-002, DR-003, DR-004. Gate chosen by the human: the DR-001 AC-1 caps (energy and virial relative 2e-6, force 1e-5 eV/A).
- Status of the conclusion: **gate NOT met for D3 on some elements; the cause is not a fork change (see below). The handling as a recorded limitation (not a defect to fix in this fork) was APPROVED by the human on 2026-11-18.**

## Method
- All-zero NEP with the 86 elements H..Rn (`scripts/common.py::zero_nep`, neighbour capacity 300), so that energy, force and virial of a run are the dispersion term alone.
- 21 structures, each with one atom of every element Z = 1..86 at random positions: 10 isolated clusters (radius 6 A, min. distance 1.3 A), 5 compressed clusters (radius 4.5 A, 1.0 A), 3 periodic small-box (cell 25 A, `dftd3 ... 12 6` small-box path), 3 periodic large-box (cell 40 A, cell-list path). Seeds in `common.py::structures`.
- References: D3(0) and BJ: simple-dftd3 1.7.0 (`py312-simpledftd3`) via `tests_pytest/fixtures/dftd3_zero/generate_reference.py::dispersion` (zero damping; PBE on all 21 structures plus revpbe, blyp, b3lyp, tpss on three structures = 33 cases) and `tests_pytest/fixtures/dftd3/generate_reference.py::reference` (BJ, PBE, 21 cases), cutoffs 12/6, two-body. D2: `tests_pytest/dftd2_reference.py` (independent double precision implementation; shares the C6/R0 table with GPUMD, so this tests the arithmetic, cutoff shift and periodicity, not the table; the table is traced to its sources in VER-0003).
- Scripts: `scripts/` (run `ref_*.py` with the simple-dftd3 environment and `cmp_*.py` with the pytest environment; `GPUMD_REPO`, `RUNS_DIR` select the repository and a scratch run directory; the zero NEP and the reference pickles are written next to the scripts, not committed). Logs: `results/`.

## Results
| check | result |
|---|---|
| DFTD-2 (23 runs: 21 structures at s6 = 0.75, rc = 12, plus rc = 30 and s6 = 1.0 on one) | all within the caps; worst: energy 5.3e-8, force 3.0e-6 eV/A, virial 4.3e-7 (relative) |
| D3(0) zero damping vs simple-dftd3 (33 cases) | energy within the cap in all cases (3e-7 .. 5e-7 relative); force and/or virial exceed the cap in 5 or 6 of 33 cases depending on the run (worst: force 9.9e-5 eV/A, virial 2.0e-5 relative); the rest agree to 1e-7 .. 1e-6 eV/A |
| D3 BJ vs simple-dftd3 (21 cases) | energy within the cap everywhere (5e-7 .. 6e-7 relative); 3 or 4 of 21 exceed in force/virial depending on the run (worst: force 5.0e-5 eV/A, virial 7.0e-6) |
| upstream master 5c67f365, BJ, same 21 cases | the same worst cases with the same values (Mo 5.0e-5, Ru 1.0e-5, Tc 1.2e-5, Se 1.1e-5 eV/A in iso_r6_2, iso_r6_6, iso_r6_9, pbc25_2) |
| pre-ADR-0008 binary (develop bccda6b6), zero damping | force errors 7e-2 .. 1.2 eV/A and virial relative 2.5e-2 .. 2.8e-1 on five of these structures: the underflow defect (UPSTREAM-0001) affects ordinary mixed-element clusters, not only H70/CuH70; ADR-0008 removes it |
| repeat noise on the exceeding cases (6 repeats) | iso_r6_2: spread 7.5e-8 eV/A, error 9.9e-5 in every run (systematic); iso_r6_9: error 6.4e-5 in 5 runs and 2.8e-5 in one (spread 3.6e-5: two discrete values) |

## Reading
- The exceeding forces are concentrated on a few 4d metals (Mo, Ru, Tc) and Se; median force error over all atoms is 1e-7 eV/A. The same structures and elements exceed with the BJ damping and with upstream master's own float code, with identical numbers, so the excess is not specific to zero damping, not caused by ADR-0008, and present in upstream. It is not a pair at a cutoff boundary (nearest pair to 6 A or 12 A is >= 2e-4 A away in the exceeding cases, while a passing case is closer).
- The mechanism was not identified. Hypothesis (untested): float coordination numbers or weights for elements whose C6 depends steeply on CN (4d metals), amplified in dC6/dCN. The two-valued error of iso_r6_9 points to a discrete branch whose outcome depends on float summation order (atomicAdd); not investigated.
- Consequently the DR-001 AC-1 caps, defined for the reference set, are not met by this stress set for D3 (BJ and zero damping); D2 meets them.
- Handling (approved by the human 2026-11-18): record as a limitation of the float D3 implementation shared with upstream; in the PR description state that the caps hold for the reference set and that on 86-element mixed clusters a few forces deviate by up to 1e-4 eV/A (Mo, Ru, Tc, Se), identically in upstream's BJ code. An alternative (double-precision coordination numbers) was not tried: it costs run time and departs from upstream.

## HIP
- No HIP toolchain on this machine (`hipcc`, `rocminfo`, `/opt/rocm` absent): no HIP build or run was done. **HIP remains unverified.**
- Static check of `src/force/dftd3/dftd3.cu`, `dftd3.cuh`, `src/force/dftd2.cu`, `dftd2.cuh`: no CUDA-only tokens (`cudaMemcpy`, `cudaMalloc`, `cudaMemset`, `cudaStream`, `cub::`, `__shfl*`, `cooperative_groups`, `cuda_runtime`); they include `utilities/gpu_macro.cuh`. `makefile.hip` and CMake collect sources by wildcard/glob (`force/*.cu`, `force/*/*.cu`), so both files are built. The ADR-0008 change (develop vs 5de76eb5) touches only `dftd3.cu` and uses no new API.
- Final confirmation needs an AMD GPU machine; say so in the PR description.

## Not covered
- Other functionals or cutoffs for BJ beyond PBE/12/6; the ATM three-body term (out of scope, DR-003); three elements Z > 86 (D3(0) refuses them by design).
- `qNEP`/`charge_transfer` with dftd2/dftd3 (separate requirements).
