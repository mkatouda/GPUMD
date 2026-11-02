# DES-0002: DFT-D2 implementation plan

- Status: Approved by the human on 2026-11-03 (together with DR-002). Implements DR-002; process per `AGENTS.md`.
- Order: Phase 0 -> 1 -> 2 -> 3 -> 4 -> 5. Each phase has a gate; if it fails, stop and report.

## 1. Facts established (2026-11-03)
- D3 hook: `NEP::initialize_dftd3` and `NEP_Charge::initialize_dftd3` parse `dftd3` tokens from
  `run_input`; `compute()` calls `dftd3.compute(...)` (`nep.cu` ~1250 and ~1716, `nep_charge.cu`
  ~1995). `run.cu:371` allows `dftd3` once. `NEP_Charge_VDW` and `NEP_MULTIGPU` reject it.
- `simple-dftd3` has no D2 option (only BJ, zero, modified, optimized-power, CSO), so it cannot
  be the independent reference as it was for D3 (VER-0001).
- NEP89 elements: H..Pu (89 elements, includes La-Lu, Ac, Th, Pa, U, Np, Pu).
- D2 values: `dftd3-lib/prg/extras.f90` (`loadoldpar`) has Z = 1-86; H-Xe verified against the
  published table; Z = 55-86 single source; Ac-Pu (6 NEP89 elements) have none (VER-0003).
- `dftd3-lib` D2 mode (`dftd3 -old -grad [-pbc]`, built with gfortran) is a usable oracle for
  energy and forces (agreement at printed precision, VER-0003). Its cell gradient differs from a
  finite difference by 8e-4 relative (unexplained): not usable as the virial reference yet.

## Phase 0: parameter provenance (no code; human-visible result)
- Obtain C6/R0 from the primary source (Grimme, J. Comput. Chem. 27, 1787 (2006)) and the
  extension source(s); cross-check against >= 2 independent tables (Turbomole, VASP wiki,
  Grimme's dftd3 program D2 section, QE/other source). Record source, version, units.
- Fix the unit conversion (J nm^6 mol^-1 -> eV A^6) with cited constants.
- Output: `specs/verification/VER-0003-dftd2-parameters/` with a table of Z, C6, R0, source,
  cross-check status, and a list of NEP89 elements WITHOUT a verified value.
- Decision (2026-11-03): built-in table = Z <= 86; any element with Z > 86 in the model is an
  error naming the element; no parameter file. The table is generated from the dftd3-lib
  source by a script (not typed by hand) and checked entry by entry (86 values).
- Gate: the generated table equals the dftd3-lib values and the H-Xe cross-check of VER-0003.

## Phase 1: independent reference and failing tests
- Extend `VER-0003/scripts/d2ref.py` to a reference with shifted-force truncation (energy,
  forces, virial by numerical strain derivative) and a finite-difference self-check.
  Validate it against `dftd3 -old` in the unshifted limit; the shift terms are then added
  analytically over the pair list within rc (oracle run with `-cutoff` = rc). Investigate the
  oracle cell-gradient discrepancy before using it for the virial.
- Cases: dimers, benzene dimer, graphite (hexagonal), water box (cubic), a triclinic cell,
  a 2000+ atom cell. Add `tests_pytest/test_dftd2_reference.py` (fails before the feature).
- Gate: reference agrees with finite differences and, if available, with the external program.

## Phase 2: cutoff study (resolves OQ-4)
- With the Phase 1 reference (and later the GPU code), tabulate energy/atom, force and
  pressure vs rc = 12, 16, 20, 25, 30, 40 A on water, graphite and a metal/oxide cell.
  Set thresholds in the VER record BEFORE measuring; choose the smallest rc meeting them.
- Gate: a default rc and the table are recorded and accepted by the human.

## Phase 3: GPU implementation
- New `dftd2.cu/.cuh`; interface mirrors `DFTD3` (`initialize`, `compute`) so the NEP hooks are
  small. One pair kernel, no coordination numbers; cell-list neighbor search (the DR-001
  small-box path with `N x MN` explicit pair lists is NOT reused); force accumulated in
  registers per atom, virial by block reduction, shifted-force terms applied per pair (energy and force both shifted).
- Do not duplicate kernels for small and large boxes; handle images by the shared neighbor
  routine. Keep `gpu_macro.cuh` for HIP; keep a fallback for double `atomicAdd` only if used.
- Gate: Phase 1 tests pass with AC-1 tolerances; warning-clean build (`sm_86`).

## Phase 4: integration and exclusion
- Hook into `NEP`, `NEP_Charge` and `NEP_Charge_VDW` (init + every `compute` path); `run.cu`
  `has_seen_dftd2` and the mutual-exclusion error; `NEP_MULTIGPU` keeps rejecting `dftd2`. 
- Element check at init: scan the model's element list, error naming every element with Z > 86.
- Tests: AC-5 input errors (pytest, `test_io_basic_setup.py` style); a run-level case in
  `tests_regression/manifest.json`.
- Gate: AC-5 and AC-6.

## Phase 5: consistency, performance, docs, review
- Finite-difference force and strain-derivative virial on the GPU code; NVE drift (AC-4);
  slope-method timing (AC-7); `doc/gpumd/input_parameters/dftd2.rst` (AC-8); `--suite full`
  against a `master` baseline; fresh `numerical-gpu-reviewer`, `physics-dielectric-reviewer`
  (units, external-correction caveat) and `api-compat-reviewer`.
- Upstream: cherry-pick code/tests/doc only (no `specs/`, `.claude/`).

## Risks
- NEP89 elements Ac, Th, Pa, U, Np, Pu have no D2 value: `dftd2` is unusable for models that
  contain them (decided: error, D5). Z = 55-86 values are single-source group fills.
- Shifted-force changes absolute energies slightly relative to the plain sum; document it.
- Physical double counting if the NEP training data already contains a dispersion term.
- Shares `NEP`/`NEP_Charge` files with DR-001 only at the hook lines; keep GPU-heavy
  subagents sequential (AGENTS.md).
