# UPSTREAM-0001: DFT-D3 C6 interpolation: float weight underflow selects a wrong C6 at large coordination numbers

- Status: recorded 2026-11-04; NOT yet reported. The human plans a bug report or a fix PR later (decision 2026-11-04).
- Fork-only note (do not include in an upstream issue/PR): found during DR-001 Phase 7 Stage 3, S1 (VER-0017). Evidence: `specs/verification/VER-0017-dftd3-stage3a/`.
- Affected upstream code: `brucefan1983/GPUMD` `master`, `src/force/dftd3.cu`, the C6 interpolation in the pair kernels
  (in `master`: lines ~142-179 of the small-box force kernel and the same block in the large-box kernel, ~line 649). Identical in the fork's
  baseline binary `master-base` (3b2c8db8). In the fork the code was refactored into `get_c6_factorized` (same semantics, Phase 7 (v)); that refactor is NOT the origin.

## Problem
```
float L_ij = exp(-4.0f * (diff_i*diff_i + diff_j*diff_j));   // float
...
if (W < 1.0e-30f) { i = num_cn_1 - 1; j = num_cn_2 - 1; c6 = c6_ref[last i, last j]; }   // dc6 = 0
```
- With float, `exp(-4 diff^2)` underflows once the CN is more than about 4.2 away from every reference CN of an element. W then falls below 1e-30 and
  the code takes the C6 of the LAST reference index. The derivative dc6 is set to 0 (forces then omit the CN dependence).
- The reference implementation (dftd3 / simple-dftd3) keeps the normalized weights in double down to |diff| ~ 13 and, beyond that, uses the reference with the
  HIGHEST CN (`max CN` reference), not the last index.
- For elements whose last reference index is not the largest CN (H: references CN {0.9118, 0.0}, last = free atom) the fallback picks the wrong (smallest-CN) C6.
  Even before the fallback, float weights lose normalization accuracy as W gets small.

## Observed effect (RTX A6000, sm_86, CUDA 12.3, reference simple-dftd3 1.7.0, PBE/BJ; fork binary and master-base identical)
| system | dE (eV) | dF (eV/A) | dV (eV) |
|---|---|---|---|
| H70 dense cluster | 8.1 | 4.4e-2 | 1.1 |
| Cu35H35 dense cluster | 7.3 | 7.2e-2 | 2.2 |
- cn_scan (70-atom clusters of decreasing radius): H clusters from approximate CN ~5 upward dE 0.2-6.7 eV; Cu clusters CN 11-19 dF 5e-3 eV/A.
  (The H CN there is an estimate with the D3 counting function and r_cov(H) = 0.32 A, not the value printed by the code.)
- Within tolerance: Cu fcc compressed (a = 2.2 A), H fcc compressed small box, dilated crystals (CN ~ 0), short pairs.
- Typical NEP89 MD systems (water, crystals, organics) are not affected at the tolerances used in DR-001 AC-1.

## Reproduction
- `specs/verification/VER-0017-dftd3-stage3a/scripts/extreme_cn.py` (structures; `--make-reference` needs `dftd3` python package) and `scripts/cn_scan.py`;
  logs in `results/`. Reference values come from simple-dftd3 (same code path as `tests_pytest/fixtures/dftd3/generate_reference.py`).
- A minimal upstream-style reproduction: a 70-atom compact H cluster (see the script) with `dftd3 pbe 12 6` vs `simple-dftd3`.
- Unrelated crash seen in the same study (not part of this issue): H fcc a = 1.9 A, 5324 atoms, illegal memory access in `src/force/nep.cu:908` (also without D3).

## Proposed fix directions (not implemented; changing results needs an ADR and a regression result in this fork)
1. Compute the weights in double (or log-sum-exp: subtract the minimum exponent, then normalize); keep the `W` threshold only as a guard against exact zero.
2. Fall back to the reference with the highest CN (as dftd3 does), not `num_cn - 1`; and keep dc6 consistent (0 only in the true plateau).
3. Add a test with a dense H cluster against simple-dftd3 (tolerance as AC-1).
In the factorized form (`cn_weight` per atom) the same repair applies per atom: normalize each atom's weights with its own minimum exponent so
W = sum_w1 * sum_w2 cannot underflow before the products are formed.

## Before filing
- Re-check against current upstream `master` (code may have changed).
- Re-run `extreme_cn.py` with a double-weight variant to confirm the cause (the cause was read from the code; no unconstrained experiment has been done).
- Decide: bug report only, or fix PR (separate from the D3 speed-up and D2 PRs; code, tests, doc only; no `specs/`, `.claude/`).
