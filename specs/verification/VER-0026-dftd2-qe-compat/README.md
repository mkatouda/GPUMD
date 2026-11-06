# VER-0026: dftd2 `table qe` / `trunc plain` verification (DR-004 R1, R2, R4) and R3 stage 1

- Date: 2026-11-06. Status: verification (against the numpy reference and the requester's reference data), NOT validation. QE 7.2 itself was never run here (VER-0025).
- Code: branch `wt-d2-qe` c3257df2 (develop af269060 + implementation + tests + review fixes). Hardware: RTX A6000 (sm_86), CUDA 12.3, GPU unshared at the measurements.
- Oracles: `tests_pytest/fixtures/dftd2_qe/` (requester's `d2_reference.py` and stored references for 4 PTFE structures, rc = 12; provenance in its README), `tests_pytest/dftd2_qe_oracle.py` (numpy reference built from the QE 7.2 tables, CODATA2018 constants), `tests_pytest/dftd2_reference.py` (DR-002).

## 1. Parameters (VER-0025, DES-0004 phase 1)
QE 7.2 `c6ab(Zi,Zj,1,1,1)` and `r0ab` for Z <= 86 equal the GPUMD/dftd3-lib tables entry by entry (0 mismatches, 3741 pairs); the generated double tables (`src/force/dftd2qe_table.cuh`, generator `tools/Miscellaneous/dftd2/generate_dftd2qe_table.py`) are byte-identical to a fresh generation from the qe-7.2 sources and to the VER-0025 CSVs. Constants: CODATA2018 as in QE 7.2 `constants.f90` (AUTOEV relative difference to the requester's value 1.3e-15). QE 7.2 stores r0ab and some s6 as single-precision literals: relative differences up to 5.9e-8 (1.2e-6 in the damping exponent); the autoang/CODATA mismatch (4.4e-9) is neglected on purpose.

## 2. Accuracy (DR-004 AC-1, AC-3, AC-4, AC-5, AC-6)
- Four PTFE structures, rc = 12, s6 = 0.75, relative errors against the requester's references (caps: energy 2e-6, virial 2e-6, force 1e-5): qe/plain energy 2.5e-9..6.4e-8, force 9.6e-7..1.25e-6, virial 3.3e-7..3.8e-7; qe/sf energy 4.1e-9..6.7e-8, force 9.7e-7..1.25e-6, virial 3.5e-7..4.0e-7.
- Worst-case pairs (new extended tests): Rn-Rn plain energy 0.4%, virial 15%, force 13% of the caps; H-H plain 5.7%, 18%, 13%.
- Mixed elements (H, C, N, O, Si, Cl, Rn, pair Z = 1/86), non-periodic and periodic, plain and sf: within the caps. Table: all 3741 C6 and r0ab entries equal the CSVs and an independent re-parse of the QE 7.2 sources.
- Boundary: a pair exactly at rc is excluded (E = 0.0), rc = d + 1e-4 includes it (QE excludes only r^2 > rthr; negligible, documented).
- Replicated 4x4x2 supercell gives the same energy per atom and forces; finite-difference force and virial consistency pass for qe/plain and qe/sf (sparse test systems with rc in a >= 0.3 A gap for plain).
- Parser/errors (ADR-0004 items 1, 2, 4, 6): valid forms, option order, explicit cutoff required with `table qe` / `trunc plain`, unknown or repeated tokens, case sensitivity, Z >= 87 with `table qe` (element named), legacy forms and the dftd3 exclusion: all pass.
- Default compatibility (AC-2): outputs of `dftd2 0.75` and `dftd2 0.75 12` are bit-identical to the pre-change binary (develop af269060; frozen outputs reproduced 3x; D2 has no atomics) for 5 structures; no `table`/`trunc` header lines for defaults. Frozen outputs are valid for the R1/R2 change only (ADR-0005 sec. 5).
- Tests: `test_dftd2_qe.py` 129 passed, `test_dftd2_qe_extended.py` 14 passed, `test_dftd2_consistency.py` 4, `test_dftd2_src_table.py` 3; `test_dftd2_reference.py` 97 (before the R1/R2 change, kernel untouched). Full: `pytest -m "not slow"` 1075 passed, 480 skipped, 0 failed (2718 s).
- Regression `run_regression.py --suite full` (baseline master 3b2c8db8, candidate wt-d2-qe c3257df2): 216 passed, 41 failed, relations 25/25. The 41 failing cases are identical to the final run of DR-003 (VER-0024 sec. 6): the 37 known non-deterministic cases and 4 unrelated ones (`dftd2_basic`, `dftd2_dftd3_exclusive`: the master baseline has no dftd2; `dftd3_single_late`, `dftd3_kspace_one_each`: develop-vs-master difference, also with the pre-change binary).

## 3. NVE with `trunc plain` (AC-5, informative)
PTFE 2x2x1 (180 atoms), NEP89 + `dftd2 0.75 12 table qe`, NVT-BDP 5 ps then NVE 20 ps, 0.5 fs, 3 seeds per mode. Total-energy slope: plain |slope| <= 2e-4 meV/atom/ps, sf <= 1e-4 (indistinguishable from zero); E_tot standard deviation plain 5.1-5.7e-4 eV, sf 4.9-5.1e-4 eV (plain slightly larger, not conclusive with 3 seeds). The per-crossing step of 7.4e-6 eV (0.75 x 49.113 x 0.597527/12^6) is buried in the per-step noise (median 9e-5 eV): no systematic drift seen; not a proof. Force step at rc about 6|phi|/rc = 3.7e-6 eV/A.

## 4. R3 speed, stage 1 (timing-only diagnostic, accuracy NOT evaluated)
Slope method (t500 - t100)/400, NEP89, NVE 0.5 fs, 5 interleaved repeats, medians, ms/step; extra = with correction minus NEP only.

| system | D3 extra | D2 extra | D2-qe plain extra | D2/D3 | D2 extra, float-accumulator variant |
|---|---|---|---|---|---|
| water 5184 | 0.91 | 1.10 | 1.01 | 1.22 | 0.64 |
| water 41472 | 6.73 | 7.09 | 7.06 | 1.05 | 1.16 |
| PTFE 40500 (10x10x9) | 5.88 | 6.38 | 6.37 | 1.09 | 1.15 |

nsys (N = 41472, per step): current `dftd2_pair_kernel` 6.44 ms, float-accumulator variant 1.09 ms (-83%), double accumulators with 6 virial components 5.04 ms (-22%); D3 `force_large_box<8>` 5.38 ms, D3 coordination number 0.72 ms. The variant (scratch only) turns the 13 per-thread accumulators (energy, 3 force, 9 virial) into float, converted to double once before the sub-thread reduction. Gate (>= 20% faster) met: the hypothesis of the requester (FP64 accumulators dominate on GA102) is supported. Limits: A100 (FP64 = 1/2 FP32) not measured (the requester measures it); accuracy of the variant unknown; the D3 third small kernel (about 0.21 ms) not attributed. Stage 2 (conforming variant, ADR-0005) follows.

## 5. Open
R3 stage 2 (accurate float accumulation, ADR-0005); manifest cases for the regression suite (runner design conflict, SPEC_CONFLICT as in VER-0024); A100 timing by the requester; QE 7.2 binary run not available.
