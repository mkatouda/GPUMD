# VER-0024: DFT-D3(0) zero damping, Phase 4 verification (DR-003, DES-0003)

- Date: 2026-11-06. Status: verification (against simple-dftd3 and the pre-change binary), NOT validation.
- Code: branch `wt-d3-zero` (develop 86899ba2 + zero damping 337a73c9 + small-box CN in double c200f20c).
- Hardware: RTX A6000 (sm_86), driver 595.71.05, CUDA 12.3.107. GPU unshared at the Stage 1-3 measurements (other processes 15 MiB).
- Reference data: `tests_pytest/fixtures/dftd3_zero/reference.json` (simple-dftd3 ZeroDampingParam, two-body), provenance `generate_reference.py`; parameters VER-0023.

## 1. Accuracy against simple-dftd3 (AC-1)
- Provisional BJ-derived tolerances failed 2 of 20 cases on the virial only (pbe:graphite 9.79e-6 eV vs 8.96e-6; pbe:heavy_cluster 2.05e-6 vs 1.06e-6, ADR-0002 caps unchanged).
- Zero-damping errors equal the BJ errors on identical systems (ratio dV 0.35-1.84, gate-1 factor 3); results deterministic over 3 runs.
- Diagnosis (double-precision numpy, GPUMD conventions): the CN-gradient and pair virial terms cancel (graphite xx: -3.427 + 3.413 = -0.0142, sum|terms|/|total| up to 1331). Float accumulation, pair-term precision and C6 interpolation were not the dominant cause; the single-precision small-box CN kernel was. Of the graphite 9.79e-6, about 7.9e-6 remains even in double precision relative to the oracle (length-constant convention: GPUMD `Bohr` vs CODATA2018 in simple-dftd3; the CN term has R_cov sensitivity about 100 eV), the float part is about 1.9e-6.
- Adopted: small-box CN kernel in double for zero damping only (BJ path unchanged, bit-identical). Effect: graphite dV 6.98e-6 (cap 8.96e-6, margin 22%), heavy_cluster 6.03e-7 (cap 1.06e-6, margin 43%); max dF 8.83e-7 (revpbe graphite, floor 1e-6, margin 12%); 20/20 reference cases within tolerance.
- Limitation: the cap margin of graphite depends on the length-constant convention (not removable by the implementation); the large-box CN kernel is unchanged (large-box cases pass).

## 2. Tests (Stage 1)
`pytest -q test_dftd3_zero.py` alone: 85 passed (649 s), including BJ bit identity against the frozen outputs, parser/error cases (ADR-0003, DR-003 D5), finite differences, small/large-box agreement, tiny r. Transient failures seen once under a shared GPU did not reproduce on the unshared GPU (cause not identified).
Also `test_dftd3_reference.py` 24, `test_dftd2_reference.py` 97, `test_dftd2_consistency.py` 4, `test_dftd2_src_table.py` 3 passed (cnA port).

## 3. Performance (Stage 2, AC-6 informative; BJ non-regression)
Slope method of VER-0021 ((t500-t100)/400, NEP89, NVE 0.5 fs), median of 5 interleaved repeats, ms/step. Comparison binary: develop e4f396e9 `dftd3.cu` built in a scratch copy (nvcc 12.3).

| system | BJ new | zero | zero/BJ (step) | zero/BJ (D3 extra) | BJ new vs develop |
|---|---|---|---|---|---|
| water 81 | 1.458 | 1.569 | 1.076 | 2.0 (noise) | +1.3% |
| water 648 | 2.096 | 2.248 | 1.073 | 1.29 | -1.0% |
| water 5184 | 2.640 | 2.797 | 1.059 | 1.15 | +1.8% |
| water 41472 | 13.478 | 14.464 | 1.073 | 1.14 | +2.8% |
| graphite 3x3x3 (972) | 2.627 | 2.725 | 1.037 | 1.10 | -2.2% |

- Decision (human, 2026-11-06): AC-6 is judged on the whole step (1.04-1.08, met); the D3-extra ratio (1.10-1.29) is reported as informative.
- BJ non-regression: borderline at the 2% criterion. Repeat spread is 1-4%; water 41472 gave +2.8% first and +1.4% (ranges overlap) in a reversed-order re-measurement (5184: +1.5%). Judged noise-level by the human (2026-11-06); not proven <= 2%; cause not investigated.

## 4. NVE energy conservation (Stage 3, AC-2)
Protocol of VER-0020 (scripts reused), NEP89, `dftd3 pbe 12 6` (+ `zero`), NVT-BDP 5 ps then NVE 0.5 fs. S1: water5184, 5 ps, seeds 1-5; S2: water648, 10 ps, seeds 1-10 (VER-0020 used 20; reduced). 45 runs, no NaN or crash. Criterion as VER-0020: candidate not significantly worse (MWU/Wilcoxon p < 0.05) and worst case <= 2x baseline.
- zero vs BJ(new), |drift| median ratio: S1 1.26 [0.68, 3.14], S2 1.19 [0.75, 1.71]; std 1.19 / 1.10; max-min 1.05 / 1.08; all p >= 0.31; criterion met.
- |drift| (meV/atom/ps, median): S1 zero 0.0102, BJ 0.0081, develop BJ 0.0077; S2 zero 0.0170, BJ 0.0143, develop BJ 0.0143.
- BJ(new) vs develop BJ: S1 ratio 1.05-1.07 (p >= 0.31), S2 ratio 1.00 for all quantities (likely bit identical).
- Reading: zero is indistinguishable from BJ; this is not proof of equivalence (point estimates 1.05-1.26x larger).
- Not done: Cu 4000 NVE; the 20-seed S2. The earlier thrust `exclusive_scan` cudaErrorIllegalAddress (water_large, zero NEP, run >= 200, during GPU contention) did not reproduce; its cause is unknown and was not probed.

## 5. Review follow-up (2026-11-06)
Independent reviews (physics, numerical/GPU, API compatibility): no BLOCKER. Fixes made: manual statements (sr8 exception, zero-only functional names, R0 in Angstrom), example `run_zero.in`, parse message, r = 0 guard.
- Large-box vs small-box (numerical review MAJOR 1). Same atoms replicated into the large-box branch, virial difference in units of the ADR-0002 cap: graphite 0.45, heavy_cluster 1.14, water27 0.02 (adopted code). BJ on heavy_cluster: 1.72 (develop code). Making the large-box CN kernel double for zero damping did not help (heavy_cluster 1.14 -> 1.40 vs small-box; versus the reference 0.78 -> 1.60, over the cap) and was REVERTED. CN-in-float is not the dominant error there: the difference was unchanged for 1-16 threads per atom, box size 50-90 A and list capacity 0; with CN nearly off (rc_cn = 0.5) it was still 0.70. Recorded as the AC-5 limitation of DR-003. The `g_cn` storage in float is kept: weights from the float CN made heavy_cluster (small box) worse, 0.57 -> 0.94 x cap.
- Length-constant floor (physics review MINOR 2). No additional reference was made. A reference without rescaling sr6/sr8 by CODATA2018/Bohr would shift r/R0 by about 8.8e-8 (about 1e-6 in the alpha = 14 damping), i.e. be worse than the current one. The agreement floor of about 1e-6 relative is set by the length-constant convention (`Bohr` of dftd3-lib/GPUMD vs CODATA2018 of simple-dftd3) and cannot be removed by the implementation; the graphite virial margin (22% of the cap) depends on it.
- Tried and not adopted: weights from the float-rounded CN (m5), above.
- Not done: `Bohr` as a float literal in the double CN (m4; effect below the margins, unmeasured); manifest cases for the regression suite (the runner applies the same expectation to baseline and candidate; needs a human decision, SPEC_CONFLICT).

## 6. Final checks (2026-11-06, branch wt-d3-zero 5efef9cb + header change)
Hardware as above (RTX A6000 sm_86, CUDA 12.3). Remaining Stage 3 checks, all passed:
- BJ bit identity: frozen-output tests pass (85/85 zero tests). Pre-change binary (86899ba2) vs new, 1000 steps NEP89 `dftd3 pbe 12 6`: small box (81 atoms) byte-identical (thermo and dump, 2 runs each); large box (5184 atoms) not byte-identical, but the old binary is itself not reproducible (atomic add order); new/old differences are at the old run-to-run level. Registers (cuobjdump, old -> new): cn_small 32 -> 34, force_small 72 -> 77, force_large 79 -> 82, other kernels unchanged; all increases <= 8; no zero-only template, 33 kernels as before.
- Performance (slope, 5 interleaved repeats, ms/step): zero/BJ whole step 1.045-1.064; BJ new vs old 0.99 (water 5184), 1.003 (graphite), 1.042 (water 41472; re-measured 1.025; borderline, ranges overlap, cause not investigated, consistent with the register increase).
- compute-sanitizer: memcheck 0 errors (2187 atoms small box, 5184 and 41472 atoms large box, 300 steps each); racecheck and initcheck 0 errors (648 atoms small box, 5184 atoms large box, 50 steps). The earlier thrust exclusive_scan cudaErrorIllegalAddress did not reproduce unshared (water 41472, zero, 1000 steps, twice); its cause is unknown.
- pytest: `-m "not slow"` 945 passed, 480 skipped, 0 failed (2462 s). D3/D2 files separately: test_dftd3_zero.py 85, test_dftd3_reference.py 24, test_dftd2_reference.py 97, test_dftd2_consistency.py 4, test_dftd2_src_table.py 3; tests_regression runner unit tests 36.
- Regression `run_regression.py --suite full` (baseline: master 3b2c8db8 binaries, candidate: this branch, NEP binaries from the same builds): final run 216 passed, 41 failed, relations 25/25. The 41 failures = the 37 known non-deterministic cases of AGENTS.md plus 4 that do not come from this change: `dftd2_basic` and `dftd2_dftd3_exclusive` (the master baseline has no dftd2), `dftd3_single_late` and `dftd3_kspace_one_each` (restart.xyz differs at 2e-12 / 1e-11 relative: these two fail identically with the PRE-change binary 86899ba2 as candidate, 3 runs, same values, i.e. a develop-vs-master difference, not caused by DR-003; origin not investigated; VER-0022 listed only the 37). A first run also failed `dftd3_duplicate` because the header printed `damping = bj`; ADR-0003 item 4 was revised (BJ output unchanged) and the case then passed.
- Not done: manifest cases for the new keyword form (the runner applies one expectation to baseline and candidate; SPEC_CONFLICT pending a human decision); Cu 4000 NVE; 20-seed S2.

## 7. Limitations carried (see DR-003)
AC-5 large-box vs small-box difference for heavy_cluster (pre-existing, also in BJ); length-constant floor ~1e-6; Z >= 87 not supported; two-body only (no three-body ABC term).

Regression suite `--suite full`, pytest `-m "not slow"` and independent reviews are not part of this record (see DES-0003 Phases 5-6). Scripts and raw outputs of the measurements are in the agent scratchpad (not kept in the repository).
