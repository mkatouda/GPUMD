# DES-0004: dftd2 QE-compatible table, plain cutoff and speed-up plan

- Status: Approved by the human on 2026-11-06 (together with DR-004). Implements DR-004; process per `AGENTS.md`.
- Precondition: DR-003 is merged into `develop` (provides `dftd3zero_table.cuh`, `r0ab`, generator of VER-0023). Work waits for this (human decision 2026-11-06).
- Order: Phase 0 -> 5; each phase has a gate; if it fails, stop and report. R3 is Phase 4 and does not block Phases 1-3.

## Facts (2026-11-06)
- `dftd2` host builds `pair_param_` (float4: s6 C6_ij, R0_ij, phi(rc), phi'(rc)); damping d = 20 fixed (`D_DAMP`); pair kernel `dftd2_pair_kernel` accumulates e_i, fx, fy, fz and 9 virial components in double; candidate positions are float (VER-0009).
- The C6 table `c6_ref_cpu` (dftd3para.cuh) already holds the D3 values; `r0ab` is added by DR-003 (VER-0023, dftd3-lib commit 3c99ee9).
- QE sources: `/home/katouda/build/q-e` (7.6-515 after tag qe-7.6); tag qe-7.2 (commit b4c5e8de) is present: read it with `git show qe-7.2:dft-d3/...`. Comparison 2026-11-06 (qe-7.2 vs qe-7.6 vs HEAD): D2 numerics identical (see DR-004 D3); release notes 7.3-7.6 have no DFT-D3 entry; INPUT_PW: dftd3_version default 3 in both, three-body always disabled for version 2. Default QE cutoff `sqrt(9000)` bohr = 50.2 A (`dft-d3/api.f90`).
- Requester data: `tmp/request_to_fork/reference/` (`d2_reference.py`, 4 PTFE structures, reference energies/forces/virials for `{qe,grimme2006} x {plain,sf}` at rc = 12, `summary_rc12.csv`).

## Phase 0: approval of DR-004/DES-0004; ADR-0004 for the `table`/`trunc` keyword extension (defaults unchanged).
## Phase 1: parameter provenance (no kernel code)
- Generate by script: `r0ab` (Z <= 86 subset of VER-0023) and C6(Zi,Zj; first reference point) from the D3 table; compare entry by entry with the QE 7.2 `dft-d3/pars.f90` (`c6ab`) and `core.f90` (`setr0ab`) and with `setfuncpar` (version 2: rs6, alp6, s6 for PBE). QE 7.2 vs dftd3-lib vs QE latest differences are listed (version 2 behavior, cutoff).
- Cross-check the pair values C-C, C-F, F-F against the request table and `d2_reference.py`; compare with QE-printed D2 energy/forces/stress if the requester supplies a QE 7.2 output (otherwise run the local QE build on a small PTFE cell; label which QE version it is).
- Gate: 0 mismatches against the QE 7.2 source for Z <= 86, or each mismatch documented and decided by the human. Result: VER record (next free number).
## Phase 2: tests first (must fail on the current binary for the right reason)
- Reference tests with the requester's data and `d2_reference.py` (provenance recorded, stored reference arrays, tolerances DR-002 AC-1); default-path bit identity; parser/error tests (AC-6); replication test (AC-4); finite-difference tests (AC-5).
## Phase 3: implementation (R1, R2, R4)
- Parser for `table`/`trunc` in the existing `dftd2` parse function (`DFTD2::parse_tokens`), used by `NEP` and `NEP_Charge`; state in the dftd2 object.
- Host-side `pair_param_` construction: `table qe` uses C6 from the D3 table and `rs6 * r0ab` with `D_DAMP = alp6 = 20`; `plain` zeroes phi(rc), phi'(rc). Kernel unchanged. `grimme2006/sf` path untouched (bit identity).
- Z >= 87 check with `table qe` (error naming elements; at the first force evaluation as for DR-002 D5).
- Docs: `doc/gpumd/input_parameters/dftd2.rst`, `docs_ja/dftd/`; examples; traceability.
- Gate: warning-clean build; AC-1,2,3,4,5,6 tests pass; `run_regression.py --suite full` (known-failure list per AGENTS.md); clang-format only on changed code.
## Phase 4: R3 speed (gated, cheapest first)
- Stage 1: timing-only variant (float accumulators in `dftd2_pair_kernel`, scratch copy) at N = 5,184 and >= 40,000 (water or PTFE-like), rc = 12, `dftd3 pbe 12 6` as comparison, 3+ repeats, slope method of VER-0021. Gate: variant at least 20% faster at large N; else stop and report (hypothesis rejected).
- Stage 2 (only after the human agrees): conforming implementation (float per-thread partial sums, one double conversion, optional compensation; 6 virial components), accuracy within DR-002 AC-1 and ADR-0002 style tolerances, determinism kept (no atomics), timing, NVE.
- Report kernel time on sm_86 only; A100 is measured by the requester.
## Phase 5: independent reviews (numerical-gpu, physics-dielectric, api-compat), final report.

## Later QE versions
No separate step is needed now: qe-7.2 .. HEAD are identical for D2 (DR-004 D3). Phase 1 re-runs the comparison against the then-current QE; a difference would be proposed as a new `table` value, not as a change of `qe`.

## Upstream PR note
DR-004 is fork-driven; any upstream PR contains code, tests, examples and `doc/` only.
