# ADR-0002: Numerical tolerance for D3 optimization

- Status: Accepted (approved by the human on 2026-11-03)
- Requirement: `DR-001`

## Context
The GPUMD D3 kernels use single precision (`float`) with double-precision accumulation of
forces. Reordering sums, factorizing the C6 weights, changing parallelization, or using a
half neighbor list changes the result at round-off level. Existing regression cases for D3
check parsing and behavior, not numerical agreement, and 37 other regression cases already
differ between identical binaries on this GPU (non-determinism).

## Decision
1. D3 results may change at the single-precision round-off level; they must not change
   beyond the tolerances of `DR-001` AC-1, which are tied to the measured baseline
   (VER-0001, `tests_pytest/test_dftd3_reference.py`) and to an independent reference
   (simple-dftd3, two-body, PBE-BJ). Energy and virial limits are relative (2e-6 of the
   D3 energy / virial scale), because their float error scales with their magnitude.
2. Golden or reference values are never regenerated to make a test pass; reference values
   come only from simple-dftd3 with recorded provenance.
3. Precision is not lowered (no half precision, no fast-math flags). Raising precision in a
   specific accumulation is allowed if it does not cost the performance targets.
4. The ATM three-body term, other damping functions, and keyword changes are out of scope
   and need their own requirement and ADR.
5. A phase that fails its gate in `DES-0001` is dropped, not relaxed by loosening
   tolerances.

## Consequences
Optimizations stay checkable against a fixed numerical oracle. A change that improves speed
but exceeds the tolerances is rejected. Bitwise reproducibility with the old binary is not
required, and the non-deterministic regression cases are interpreted against the known list.
