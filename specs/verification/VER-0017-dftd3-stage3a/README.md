# VER-0017: DFT-D3 Stage 3 final verification, part a (DES-0001 Phase 7 Stage 3)

- Date: 2026-11-04. Status: verification (S1 only); NOT validation. Worktree `wt-d3-stage3`, HEAD `0ac42758` (items (i), (v), (vi) merged).
- Environment: RTX A6000 (sm_86), CUDA 12.3, NEP89, reference simple-dftd3 1.7.0 (PBE, BJ, two-body, same cutoffs; same code path as
  `tests_pytest/fixtures/dftd3/generate_reference.py`), baseline binary = `master-base` (upstream master).
- **Stage 1 (extreme coordination numbers): gate NOT met. Stages 2-4 were not started (gate rule).**

## S1 scripts and results
- `scripts/extreme_cn.py` (structures + reference with `--make-reference` in the env that has `dftd3`; GPUMD runs with the ase env):
  10 systems (compressed H/Cu fcc, dilated Cu/Au/C, short pairs, dense H and Cu/H clusters). `results/s1_extreme_cn.log`.
- `scripts/cn_scan.py`: 70-atom H and Cu clusters of decreasing radius (error versus coordination number). `results/s1_cn_scan.log`.
- AC-1 caps used (no per-case baselines exist): energy / virial 2e-6 relative, force 1e-5 eV/A. NEP noise (two runs without D3) was 0 in all cases.

Findings:
1. No NaN/inf in any case, both binaries. Candidate and master-base agree on every D3 contribution (max |dE| 3e-5 eV at |E| = 9807 eV, |dF| 1e-6 eV/A, |dV| 2e-3 eV
   on compressed Cu; <= 6e-8 eV / 8e-8 eV/A on the clusters): items (v) and (vi) change nothing at extreme CN.
2. Within the AC-1 caps: Cu fcc compressed (a = 2.2 A, 4000 atoms), H fcc compressed small box, the four dilated crystals (CN ~ 0), short pairs.
3. NOT within the AC-1 caps, identical for master-base and candidate (so pre-existing, not caused by Phase 7): `H70_dense_cluster` (dE 8.1 eV, dF 4.4e-2 eV/A, dV 1.1 eV),
   `CuH70_dense_cluster` (7.3 eV, 7.2e-2, 2.2), and in `cn_scan.py` H clusters from approximate CN ~ 5 upward (dE 0.2 .. 6.7 eV) and Cu clusters with CN 11-19 (dF 5e-3 eV/A);
   Cu with CN > 25 and H with CN <= 3 are within tolerance.
4. Cause (read from `src/force/dftd3.cu`, `get_c6_factorized`): the weights are computed in float, `exp(-4 diff^2)` underflows for diff > about 4.2 from every reference, and
   `W < 1e-30f` then selects the C6 of the LAST reference index `(num_cn - 1)` (a fixed value, derivative 0), while simple-dftd3 keeps the double-precision normalised weights up to
   |diff| ~ 13 and only then falls back to the reference of highest CN. For H (references CN 0.9118, 0.0) the last index is the free atom, not the highest CN. The same code is in
   master-base (upstream behaviour). Item (v) did not introduce it.
5. H fcc compressed (a = 1.9 A, 5324 atoms) crashes inside NEP (`force/nep.cu:908`, illegal memory access, also in master-base, also without D3): not a D3 matter (NEP neighbor limit), system dropped.
6. No pytest added: a test with these structures would fail against the stored-reference tolerances and tolerances must not be loosened.

## Follow-up
Human decision 2026-11-04: handled as (a) (pre-existing upstream behaviour, noted in DR-001 AC-1); recorded for a later bug report or fix PR in `specs/upstream-issues/UPSTREAM-0001-dftd3-weight-underflow.md`. Stages S2-S4 continue.
