# DES-0003: DFT-D3 zero damping implementation plan

- Status: Approved by the human on 2026-11-06 (together with DR-003). Implements DR-003; process per `AGENTS.md`.
- Order: Phase 0 -> 6. Each phase has a gate; if it fails, stop and report.

## Facts (2026-11-06)
- BJ damping formula is duplicated in `src/force/dftd3.cu` (small-box ~l.593, large-box ~l.1129);
  parameters in `DFTD3_Para {s6,s8,a1,a2}` (`dftd3.cuh`), table via `set_para()` (~60 lines).
- `dftd3para.cuh` has `r2r4` and the C6 table but NO `r0ab` (pair cutoff radii), needed by D(0).
- Parsing: `NEP::initialize_dftd3` (`nep.cu:48`) and `NEP_Charge::initialize_dftd3`
  (`nep_charge.cu:146`) require exactly 4 tokens.
- simple-dftd3 supports zero damping: usable as oracle (as in VER-0001).

## Phase 0: approval of DR-003/DES-0003 and an ADR for the keyword extension (default bj).
## Phase 1: parameter provenance (no kernel code)
- Generate `r0ab` (94x94 symmetric) and D3(0) parameters (s6, sr6, s8; alpha fixed) from
  dftd3-lib (`pars.f`/`param.f90`) by script; cross-check with simple-dftd3. Stage A: PBE, revPBE,
  BLYP, B3LYP, TPSS. Stage B: all D3(0) functionals.
- Gate: generated tables equal the source and simple-dftd3 entry by entry (VER record).
Phase 1 result (VER-0023, 2026-11-06): gate met for Z <= 86 and all tabulated functionals;
Z = 87-94 disagree between sources, resolved by DR-003 D5 (limit to Z <= 86, error otherwise).
## Phase 2: tests first (must fail): oracle comparison, finite difference, parser, BJ bit-identity.
## Phase 3: implementation
- Add damping kind + sr6 (+alpha constants) to `DFTD3_Para`; move the damped pair terms into one
  `__device__` inline helper used by both small- and large-box kernels (BJ numerics unchanged).
- `r0ab` as `__constant__` (convert Angstrom/Bohr once, as for BJ a2). Guard (r/R0)^-alpha for
  small r (compute in a form that cannot overflow in float).
- Parse optional 5th token in both `initialize_dftd3` functions; `set_para`-style D(0) table.
- Gate: warning-clean build; BJ outputs bit-identical to the previous binary (AC-3).
## Phase 4: verification per DR-003 AC-1,2,5,6 (VER record: tolerances, GPU arch, CUDA).
## Phase 5: regression and docs: `run_regression.py --suite full` vs master baseline (37 known
  non-deterministic cases excluded per AGENTS.md); pytest `-m "not slow"`; manual, traceability.
## Phase 6: independent reviews (numerical-gpu, physics-dielectric, api-compat) on the diff.

## Upstream PR note
Stage B (all D3(0) functionals) precedes the PR. The PR branch is built from `upstream/master`
with code, tests, examples and `doc/` only (no `specs/`, `.claude/`, `docs_ja/`).
