# ADR-0008: Per-atom normalised, shift-stabilised C6 reference weights in the fork's `dftd3`

- Status: Accepted (approved by the human on 2026-11-18; text as drafted, with the decisions Q1-Q4 recorded below). Implementation not started; it follows this ADR under the acceptance criteria.
- Numbering: ADR-0007 is reserved for the DR-007 revision (`/home/katouda/work/dielectric-response_pj/tmp/reply_to_requester_20261118/02_ADR-0007_DRAFT_rev2026-11-18.md`, not placed yet), hence this ADR is 0008.
- Related: DR-001 (dftd3 speed, the cached `cn_weight` table), DR-003 (zero damping), ADR-0002 (dftd3 optimisation numerics, tolerances), `specs/upstream-issues/UPSTREAM-0001-dftd3-weight-underflow.md` (the recorded defect), VER-0017 (evidence), VER-0037 (merge record; upstream commit `5c104c42` was not adopted there, "to be reconsidered as a separate commit with its own VER").
- Supersedes in part: DR-003 AC-3 ("`bj` reproduces the previous binary bit for bit") for the C6 interpolation only. See Consequences.

## Context
The fork computes the C6 reference weights once per atom (`store_cn_weights`, `src/force/dftd3/dftd3.cu` ~l.64) as `w_k = exp(-4 (CN - CNref_k)^2)` and stores them as float (BJ: the exponential is evaluated in float; zero damping: in double, then stored as float). The pair kernels combine them in `get_c6_factorized` (~l.109):

    c6 = Z / W,  dc6 = dZ / W - c6 dW / W,   W = (sum_k w1_k)(sum_l w2_l)
    if (W < 1e-30f) c6 = c6_ref[last i][last j], dc6 = 0

This is the same defect as upstream's pre-`5c104c42` code and as `UPSTREAM-0001` (recorded by the fork 2026-11-04, never reported). A float `exp(-4 d^2)` underflows once an atom's CN is about 4.2 or more away from every reference CN of its element. Then W falls below 1e-30 and the code uses the last reference pair (not necessarily the highest-CN reference; for H the last reference is the free atom) and sets dC6/dCN to 0, so the force omits the CN dependence. The D3 energy is discontinuous where the branch switches. The reference implementations (dftd3, simple-dftd3) keep normalised weights and, far outside the reference range, use the reference nearest in CN.

### Measurement (2026-11-18, this session; RTX A6000, sm_86, CUDA 12.3, NEP89 + `dftd3 pbe 12 6`, one static frame, D3 contribution = energy with D3 minus energy without)
Fork `develop` (fc67a8cc) versus upstream `5c67f365` (contains the fix `5c104c42`); structures are the VER-0017 cases (`specs/verification/VER-0017-dftd3-stage3a/scripts/extreme_cn.py`).

| system | dE (develop - upstream) | max dF | max dV |
|---|---|---|---|
| H70 dense cluster (non-periodic) | -8.11 eV | 4.3e-2 eV/A | 1.14 eV |
| Cu35H35 dense cluster | -7.25 eV | 7.2e-2 eV/A | 2.20 eV |
| H fcc a=1.9 A, 256 atoms (small box) | 1.3e-5 eV | 4.5e-8 eV/A | 1.4e-6 eV |
| Cu fcc a=2.2 A, 4000 atoms | 4.7e-5 eV | 1.2e-6 eV/A | 1.7e-3 eV |
| H2 / Cu2 short pairs | 6e-9 eV | 3.6e-12 eV/A | 2.9e-11 eV |

The first two rows equal the errors of the fork against simple-dftd3 recorded in UPSTREAM-0001 (8.1 eV / 4.4e-2 eV/A / 1.1 eV and 7.3 eV / 7.2e-2 eV/A / 2.2 eV), to two digits. This suggests that upstream's fixed code agrees with the reference implementation there; it is NOT a direct comparison with simple-dftd3 (the python `dftd3` package is not installed here). Smooth crystals are not affected: lattice scans of Cu (fcc, a 2.8-4.2 A, +C interstitial) and Na (bcc, a 3.4-5.6 A, +C interstitial) differ from upstream by at most 1e-6 eV/atom, with no discontinuity in either binary. Scripts and raw numbers: session scratchpad `c6diag/` (`scan.py`, `scanC.py`, `ext.py`).

## Decision
1. **Normalise the weights per atom and shift the exponent, at the place where the fork already builds the weight table.** In `store_cn_weights` (per atom, once per CN evaluation; same kernels, same table size `N * max_cn` floats):
   - `d_k = CN - CNref_k` (type `T` as now: float for BJ, double for zero damping), `m = min_k d_k^2`;
   - `e_k = exp(-4 (d_k^2 - m))` (the nearest reference gives exactly 1), `S = sum_k e_k >= 1`;
   - store the normalised weight `w~_k = float(e_k / S)`; `sum_k w~_k = 1`, no underflow for any CN.
2. **Pair kernels.** In `load_cn_weights`: `d_mean = sum_k w~_k d_k` and `dw_k = -8 w~_k (d_k - d_mean)` (derivative of the normalised weight with respect to CN of the central atom); `sum_w = sum_dw = 1, 0` need not be accumulated. In `get_c6_factorized`: `c6 = Z`, `dc6 = dZ` (no division by W, no recomputation of `sum_w2`), and the `W < 1e-30f` fallback is removed. The `num_cn_1 == 1 && num_cn_2 == 1` early return is kept. Same formulas as upstream's `find_c6_stable` (`5c104c42`), but evaluated per atom and cached instead of per pair.
3. Scope: `src/force/dftd3/dftd3.cu` only, both small-box and large-box kernels (they share the helpers). No keyword, file-format, or option change. `dftd2`, `charge_transfer`, replica, and the CPU/HIP macros are not touched (D3 is not available in replica).
4. Not adopted from upstream (unchanged): `d5bf3f7f` (own weight cache exists, this ADR modifies it), `1c2c85e5` (the fork still uses `cn`), `c79ed9a4` (DR-006 covers it).
5. Result policy. In the regime where the old code did not underflow, results change only by float rounding (different operation order); in the underflow regime they change to the physically correct value (the point of the change). The change is a numerical bug fix, flagged as such in the doc and release notes.

## Alternatives considered
- **A. Port upstream's `find_c6_stable` verbatim (per-pair evaluation).** Same maths, easiest to compare with upstream bit-for-bit. Rejected as the default: it evaluates `n1 + n2` exponentials per pair, undoing DR-001's per-atom weight cache (an estimate; not measured), and keeps two implementations of the weights in the tree (the cache is also used by the zero-damping path, which has no upstream counterpart).
- **B. Store the weights in double.** Moves the underflow to |d| of about 13 (`UPSTREAM-0001`) but does not remove it; doubles the table and loads; fp64 is 1/64 rate on this GPU class. Rejected.
- **C. Keep the sum-based form and replace only the fallback by the highest-CN reference (UPSTREAM-0001 fix direction 2).** Still discontinuous where W crosses 1e-30 and still sets dC6 = 0 inside the plateau. Rejected.
- **D. Leave the defect and document it.** Rejected: errors of 7-8 eV and 7e-2 eV/A on dense H and Cu-H clusters; it affects any user who compresses H or metals.

## Acceptance criteria (proposed; none may be loosened to pass)
- AC-1 (defect fixed). VER-0017 structures `H70_dense_cluster`, `CuH70_dense_cluster`: the D3 energy, forces, virial agree with upstream `5c67f365` (the same binary as in the table above) within the DR-001 AC-1 caps (energy and virial 2e-6 relative, force 1e-5 eV/A), and the 8.1 / 7.2 eV differences are gone. The agreement with simple-dftd3 (same caps, reference computed with the same settings as `extreme_cn.py --make-reference`, PBE/BJ) is required (human decision 2026-11-18, Q2); agreement with upstream is reported in addition.
- AC-2 (force consistency). Central finite difference of the D3 energy along displacements of a dense-cluster atom (where the old dC6 was 0) matches the analytic force within `max(1e-4 eV/A, 1e-3 x max|F_D3|)` (fixed by the human 2026-11-18, before implementation; not to be loosened); the same test fails on the pre-change binary. Method: central difference of (E with D3 - E without D3), h = 0.02 and 0.04 A, atoms 0, 17, 35, 52, 69, x direction, structures `H70_dense_cluster` and `CuH70_dense_cluster`; each displaced run in its own run directory (gpumd appends to an existing dump.xyz). Calibration measured 2026-11-18 (max error over the 5 atoms, h = 0.02 / 0.04): `H70_dense_cluster` develop 1.6e-1 / 7.9e-2 eV/A (fails; error scales as 1/h, i.e. the analytic force is inconsistent with the energy), upstream 5c67f365 9.9e-7 / 5.3e-6 eV/A (passes, 20x margin); `CuH70_dense_cluster` develop 5.7e-6 / 5.8e-6, upstream 6.3e-6 / 8.1e-6 (both pass; this system does not expose the defect, so the H70 case is the discriminating one).
- AC-3 (continuity). A lattice-scale or cluster-radius scan through the old switch point shows no jump of the D3 energy larger than 10x the median step-to-step change (diagnostic of this session as a template).
- AC-4 (no regression in the normal regime). DR-001 AC-1 reference set (`test_dftd3_reference.py`, `test_dftd3_zero.py`, VER-0001 systems, water and Cu large box): errors against the stored references no worse than max(2 x the old binary's error, same-binary repeat noise) as in DR-001. Existing tests pass unchanged.
- AC-5 (speed). nsys D3 kernel total, water 5184 and 41472 atoms and Cu large box, 3 alternations, new vs `develop` binary: not slower than 1.03x (the Stage 3c gate of VER-0019). A speed-up is reported but not required.
- AC-6 (build and suite). Build warning-clean (sm_86); focused pytest (`dftd3*`, `dftd2*` suites); `run_regression.py --suite full` against DEVELOP: only `dftd3_*` cases may change, by float rounding, and every changed case is listed with its max difference (the runner compares with atol 1e-17, so some dftd3 cases will report numeric differences; this is expected and is recorded, not loosened); against `master` (`5c67f365`) baseline the two cases known to differ by C6 rounding (`dftd3_single_late`, `dftd3_kspace_one_each`) are re-measured and reported either way.
- AC-7 (reproducibility). Repeat-run noise of the new binary (float atomicAdd) is measured as in the VER-0037 G3 procedure and is not worse than the old one.
- AC-8 (review). Independent `numerical-gpu-reviewer` read of the diff; no open BLOCKER/MAJOR.
- AC-9 (records). New VER (next free number, expected VER-0038); `specs/traceability.yaml` (DR-001, DR-003); `doc/gpumd/input_parameters/dftd3.rst` only if a user-visible statement changes; `UPSTREAM-0001` marked withdrawn/fixed (human decision, same treatment as UPSTREAM-0003); AGENTS.md "Known test results" re-measured if the dftd3 case list changes; `docs_ja/dftd/` notes updated.

## Consequences
- Fixes the underflow defect in the fork; the fork then agrees with upstream's fixed behaviour in the previously wrong regime, so a later merge of upstream dftd3 changes has one conflict less in this area.
- DR-003 AC-3 (BJ bit identity with the previous binary) no longer holds for the C6 part: rounding-level changes (the 2026-11-18 measurement of the upstream rewrite on `dftd3_single_late`: relative 1.6e-9 in PE). This ADR is the decision record for that exception; DR-003 itself is amended by one dated line (human approval).
- Possible bit-level differences from upstream remain (different evaluation order: per atom vs per pair); they are reported, not eliminated.
- Cost: the CN kernel gets one extra pass over `num_cn <= max_cn` references (min and sum); the pair kernels lose a division, the recomputation of `sum_w2`, and the fallback branch. Expected neutral to slightly faster; to be measured (AC-5), not assumed.
- Memory: unchanged (`N * max_cn` floats).
- Risks: (1) the shared helpers are used by the large-box (cell list, `cn_list`) and small-box (multi-thread) kernels; a missed call site would mix old and new semantics: AC-6 and the reviewer check all `store_cn_weights`/`load_cn_weights`/`get_c6_factorized` uses (call sites at ~l.302/307, 669/700, 1109, 1155/1218 of the pre-change file). (2) Zero-damping CN is double: the min-shift and sum are done in `T`, only the final weight is stored as float. (3) The table value changes meaning (normalised); any other reader of `cn_weight` must be checked (grep found only `dftd3.cu`).

## Decisions by the human, 2026-11-18 (answers to Q1-Q4 below)
- Q1: ADR alone, no DR-008.
- Q2: the comparison with simple-dftd3 is REQUIRED for AC-1 (the two dense clusters), in addition to the comparison with upstream. Environment: `/home/katouda/miniforge3/envs/py312-simpledftd3/`. The reference values come from there; none are invented.
- Q3: UPSTREAM-0001 is withdrawn, like UPSTREAM-0003 (file kept with the reason; done together with the implementation, AC-9).
- Q4: AC-2 tolerance `max(1e-4 eV/A, 1e-3 x max|F_D3|)`, as written in AC-2.
- ADR text approved as a whole by the human on 2026-11-18.

## Open questions for the human (answered above; kept for the record)
- Q1. ADR alone, or also a short requirement DR-008 (R: "D3 C6 interpolation stays finite, smooth and consistent with the reference at any CN")? Recommended: ADR alone, with `traceability.yaml` entries under DR-001/DR-003, because the requirement is a defect fix of an existing one (DR-001 "unchanged results" must be read as "unchanged where the old result was correct").
- Q2. Is the comparison with upstream's fixed binary an acceptable oracle for AC-1, or is a simple-dftd3 reference required (needs installing the python `dftd3` package in a scratch environment)? Recommended: required for at least the two dense clusters, because UPSTREAM-0001's errors were measured against it; upstream agreement alone only shows the two codes are consistent.
- Q3. UPSTREAM-0001: withdraw (as UPSTREAM-0003), or keep it as a record that the fork found the defect first? It would be fixed upstream either way (`5c104c42`, 2026-11-06). Recommended: withdraw, keep the file with the reason.
- Q4. Tolerance of AC-2 (finite-difference force check), to be fixed before implementation.
