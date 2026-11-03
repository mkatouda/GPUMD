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

## Phase 6: performance optimization (draft 2026-11-03)
- Why: at the default rc = 30 A the D2 term costs 22-54x the D3 term and about 35x the NEP evaluation
  (VER-0006: N = 81..41,472, slope method); DR-002 AC-7 (D2 <= 0.5x D3) is unmet. The VER-0005/0006
  data also show a cost jump between rc = 16 and 20 A (x4.8 time for x1.95 neighbors) and equal
  cost at N = 648 and 5,184 (rc = 30), i.e. the GPU is not filled at small N. Current kernel: one
  thread per atom, full neighbor traversal, no atomics, deterministic (Phase 5d).
- Constraints: results stay within the AC-1 tolerances against the NumPy reference; the
  deterministic (bit-reproducible) behaviour is kept unless the human approves otherwise
  (a half list with force atomics would give it up: decision required before implementing it).
  Default numerics (rc = 30 A, shifted-force) are not changed.
- Stage 1 (cheap diagnostic, no code in src/; scratch variants only; nsys, ncu counters are not
  available): on water 648, water 5,184 and Cu/Au 4,000 at rc = 12 and 30 A,
  (a) share of the pair kernel spent in the transcendental part (variant with the damping/exp
  replaced by a constant), (b) candidate pairs visited versus accepted pairs (r < rc) per atom
  (host-side count in a harness), (c) kernel time versus the cell divisor and the 16-20 A jump
  (find the cause: registers/occupancy via the compiler report, local memory spills, cell-visit
  count), (d) time versus N at fixed rc to quantify the fill problem at N <= 5,000.
  Gate for continuing to Stage 2: at least one of (a) >= 25% share, (b) accepted/visited <= 0.3,
  (c) a removable cause of the 16-20 A jump, (d) a small-N time that is >= 2x the time expected
  from the large-N throughput. If none holds, stop and keep AC-7 as an informative miss.
- Stage 1 outcome (2026-11-03, scratch harness, nsys, pair-kernel median, RTX A6000 sm_86; raw tables
  kept outside the repository): gate passed via (b), (c), (d); (a) not met.
  (a) exp/damping replaced by a constant saves only 0.5-1.7% of the pair kernel (water 648/5,184,
  Cu 4,000, rc = 12 and 30 A): candidate scanning dominates, not the transcendental part.
  (b) accepted/visited = 0.08-0.19 at rc = 30 A (0.04-0.34 over all cases); CELL_DIVISOR = 3 is about 11%
  faster at rc = 30 A for large N and slower for water 648 at rc = 12 A.
  (c) the 16 -> 20 A jump on water 648 (18.6 A box) comes from the cell-count quantization
  n = floor(2*width/rc): n drops from 2 to 1 at rc >= 18.6 A, every atom then scans the 125 images
  (k = 2), visited candidates x8, time x4-4.8; 82 registers, no spills, no local memory. A scratch
  `ceil` variant (timing only, accuracy not checked) was 2.18x (rc 20), 1.93x (rc 25), 1.20x (rc 30)
  faster, but 0.63x (slower) at rc = 12: the cell count must be chosen by a cost estimate.
  (d) at rc = 30 one thread per atom fills only ~6 of 84 SMs at N = 648: measured time is 15.7x
  (N = 648), 4.2x (N = 2,025), 2.4x (N = 5,184) the time expected from the N = 20,736 throughput.
  Consequence: item (iv) (half list) is expected to help little (transcendental part < 2%) and costs
  determinism; it is not pursued unless the human decides otherwise after (i)-(iii).
- Stage 2 (candidates, each separately, in this order, each with its own gate): (i) the cause found
  in (c) if any; (ii) several threads per atom with a fixed-order reduction for small N (as in
  DR-001 Phase 3; deterministic); (iii) tighter candidate filtering / cell size / shared-memory tiling of
  the neighbor cells; (iv) half list with atomics (only with human approval of non-determinism).
  Gate per item: >= 1.3x D2 time on water 648 at rc = 30 A (and no slowdown above 1.0x at
  the other measured sizes), AC-1 tolerances unchanged and met, determinism test of Phase 5d
  still passing (except (iv)), warning-clean build.
- Stage 2 outcome (2026-11-03, VER-0008, pair-kernel median, nsys, RTX A6000 sm_86): (i) cost-based cell count
  (candidates + 3 x visited cells) adopted: water 648 rc = 30 A 1.62x, rc = 20 A 2.28x, no case slower than 1.03x.
  (ii) several threads per atom (power of two, fixed-order shared-memory tree reduction, no atomics, deterministic)
  adopted: a further 9.4x at water 648 rc = 30 A (15.2x in total). D2 extra at rc = 30 A: 54.3 -> 2.92 ms (N = 648),
  211 -> 117 ms (N = 41,472); D2(30)/D3 = 1.3-15.7 (was 22-54); AC-7 still unmet at rc = 30 A (informative).
  (iii) not started (forced cell counts are always slower than the automatic choice). (iv) not pursued.
  Observation: the candidate loop computes coordinate differences in double (GA102 FP64 = 1/64 rate); a float
  scratch variant was 1.3-1.7x faster, accuracy not yet checked (Stage 2b, human-approved 2026-11-03).
- Stage 2b outcome (2026-11-03, VER-0009): float candidate loop with cell-centre-relative float coordinates adopted
  (naive float coordinates failed the accuracy gate: 2.53 x AC-1 tolerance far from the origin). Max error 0.27 x AC-1
  tolerance (gate 1/3); sum of forces ~1e-15; pair kernel 1.49-1.86x faster on 35 points (water 648/5,184, Cu 4,000 at
  rc = 30 A: 1.6x), no slowdown; determinism and all existing tests unchanged. Not yet done: finite difference at
  rc = 30 A, --suite full, independent review (Stage 3).
- Stage 3: re-measure VER-0006 timing tables (AC-7), re-run the finite-difference checks at
  rc = 30 A (closing the rc = 12 A-only gap of AC-4), `--suite full` against the master baseline,
  independent numerical review of the changed kernel; then set the DR-002 status by the rule
  in AGENTS.md (implemented only if all acceptance criteria incl. AC-4 are met; AC-7 is
  informative and is reported as met/unmet).

## Risks
- NEP89 elements Ac, Th, Pa, U, Np, Pu have no D2 value: `dftd2` is unusable for models that
  contain them (decided: error, D5). Z = 55-86 values are single-source group fills.
- Shifted-force changes absolute energies slightly relative to the plain sum; document it.
- Physical double counting if the NEP training data already contains a dispersion term.
- Shares `NEP`/`NEP_Charge` files with DR-001 only at the hook lines; keep GPU-heavy
  subagents sequential (AGENTS.md).
