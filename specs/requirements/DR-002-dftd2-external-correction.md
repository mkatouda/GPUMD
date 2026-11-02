# DR-002: DFT-D2 (Grimme 2006) dispersion correction added externally to NEP

- Status: Approved by the human on 2026-11-03. OQ-4 (default cutoff) is an output of DES-0002 Phase 2, not an open requirement question
- Design: `specs/design/DES-0002-dftd2-implementation-plan.md`
- Related: DR-001 (D3 speed-up; independent, no shared files until Phase 3 of DES-0002)

## Requirement
GPUMD shall provide a `dftd2` keyword that adds the DFT-D2 pairwise dispersion energy, forces,
and virial on top of a NEP model, for the NEP variants listed in Scope. The correction is
applied externally: it is NOT assumed to be part of the NEP training data, and the user is
responsible for the physical appropriateness (no double counting with a learned or built-in
dispersion term).

## Model (to be confirmed against the primary source in DES-0002 Phase 0)
E_D2 = - s6 * sum_{i<j} C6_ij / r_ij^6 * f_damp(r_ij),  f_damp = 1 / (1 + exp(-d (r/R0_ij - 1)))
C6_ij = sqrt(C6_i C6_j), R0_ij = R0_i + R0_j, d = 20 (fixed, as in Grimme 2006),
sum over all pairs and periodic images with r_ij < rc.

## Decisions taken by the human (2026-11-03)
D1. Cutoff: default chosen from tests in the 12-40 A range (DES-0002 Phase 2). A
    shifted-force truncation is applied (decided 2026-11-03, resolves OQ-3):
    phi_sf(r) = phi(r) - phi(rc) - phi'(rc) (r - rc) for r < rc, 0 beyond; so energy and
    force both vanish continuously at rc. phi(r) = -s6 C6 f_damp(r) / r^6 per pair.
D2. s6 is given directly by the user; there is no functional table.
D3. `dftd2` and `dftd3` are mutually exclusive; specifying both is an input error.
D4. Scope: NEP, the charge variants, and qNEP including the vdW models `nep_vdw` and
    `qnep3_vdw` (decided 2026-11-03, resolves OQ-1): `dftd2` is accepted there. The existing
    rejection of `dftd3` on those models is NOT changed. Multi-GPU NEP keeps rejecting
    `dftd2` (not requested; confirm if needed).
D5. Parameters (decided 2026-11-03, resolves OQ-2): the built-in C6/R0 table covers Z <= 86
    (H-Rn) only, taken from dftd3-lib (`prg/extras.f90`, `loadoldpar`, commit `3c99ee9`) after
    the H-Xe cross-check of VER-0003. A model that contains an element with Z > 86 (for NEP89:
    Ac, Th, Pa, U, Np, Pu) makes `dftd2` stop with an error that names the element(s); there is
    no silent zero, extrapolation or fallback. A user-supplied parameter file is NOT part of
    this requirement (possible later extension). Z = 55-86 values come from a single source
    and are group-filled (e.g. Ce-Lu share one value); the manual must say so.
D6. Intended use: D2 added externally to NEP.

## Open items (SPEC_CONFLICT candidates; do not guess)
- OQ-1: RESOLVED (D4). Remaining check: `NEP_Charge_VDW` already has its own vdW term; the
  manual must warn about double counting when `dftd2` is used with it.
- OQ-2: RESOLVED (D5). Evidence is in VER-0003. Licensing: dftd3-lib is GPL v1+, GPUMD is
  GPL v3; cite Grimme, J. Comput. Chem. 27, 1787 (2006) and dftd3-lib commit `3c99ee9` where
  the table is stored.
- OQ-3: RESOLVED (D1: shifted-force).
- OQ-4 (default rc): the value is an output of Phase 2 (tolerance in AC-3), not fixed here.

## Scope
In: new files `src/force/dftd2.cu/.cuh`, the hook in `NEP` and `NEP_Charge` (init + compute),
`run.cu` keyword bookkeeping and exclusion check, `doc/gpumd/input_parameters/dftd2.rst`,
tests, a VER record. Out: any change to D3 numerics; `nep` (training) executable; HIP-specific
tuning beyond going through `gpu_macro.cuh`.

## Keyword (proposed)
`dftd2 <s6> [<cutoff>]`   s6 dimensionless (> 0), cutoff in angstrom (default per D1).
No parameter-file option (D5).

## Acceptance criteria
AC-1 (accuracy). Energy, forces and virial equal an independent D2 implementation (see
  DES-0002 Phase 1) within relative 2e-6 (energy, virial) and 1e-5 (force, relative to the
  largest force component) for non-periodic, cubic, hexagonal and triclinic cases, with the
  same shifted-force definition on both sides. The independent implementation is validated
  against the dftd3-lib D2 mode in the unshifted limit (VER-0003).
AC-2 (parameters). Every built-in C6/R0 value (Z <= 86) is traceable to the cited source;
  H-Xe agree with the published table (VER-0003), Z = 55-86 are labelled single-source in the
  docs; units and conversion constants are documented. A test checks the built-in table against
  the dftd3-lib values (all 86 entries). A model element with Z > 86 (or not in the table) stops
  the run with an error naming the element(s), tested with NEP89 and a model containing Ac..Pu
  elements; it is never a silent zero.
AC-3 (cutoff). The default cutoff is the smallest value in 12-40 A for which, on the test
  systems, energy per atom, force and pressure changes against rc = 40 A are below stated
  thresholds (set in the VER record before measuring). The measured table is recorded.
AC-4 (consistency). Force equals the negative finite-difference energy gradient; virial equals
  the numerical strain derivative; NVE drift is no worse than the same run without D2.
AC-5 (input). `dftd2` + `dftd3` together, a repeated `dftd2`, s6 <= 0, cutoff <= 0, or use on an
  unsupported potential each stop with a specific error message.
AC-6 (no regression). Without `dftd2`, results of `pytest -m "not slow"` and
  `run_regression.py --suite full` are unchanged (known non-deterministic cases excepted).
AC-7 (performance). D2 extra time per step is reported (slope method) for N = 81 .. 41,472;
  target: no worse than 0.5x of the D3 time on the same system (informative, not a gate).
AC-8 (docs). The manual page states formula, units, the shift, default cutoff and its basis,
  cost scaling, the "external correction" caveat, and element coverage.
