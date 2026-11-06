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

## 5. Open
Regression suite `--suite full`, pytest `-m "not slow"` and independent reviews are not part of this record (see DES-0003 Phases 5-6). Scripts and raw outputs of the measurements are in the agent scratchpad (not kept in the repository).
