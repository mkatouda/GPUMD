# AGENTS.md

This file provides guidance to coding agents (Claude Code, etc.) when working with code in this repository.

GPUMD is a CUDA/HIP molecular dynamics package. Executables: `gpumd` (MD/static calculations, input `run.in` + `model.xyz`), `nep` (NEP training/prediction, input `nep.in`), plus `gpumd_replica`, `tb` and optional `gnep`. Docs: `doc/` (Sphinx, user manual); `developer/README.md` is the developer guide — read it before adding keywords, potentials, ensembles, or actions.

## Scope of these rules

This fork (mkatouda/GPUMD) develops dielectric-response features on top of upstream
brucefan1983/GPUMD. The process rules below, `specs/`, and `.claude/` are fork-only:
never include them in upstream PRs (build upstream PR branches from `upstream/master`
and cherry-pick only code, tests, examples, and `doc/` changes). Upstream code style
and conventions (Build / Tests / Architecture sections) always apply.

## Authority and specifications

Approved files under `specs/` are the source of truth. Priority: requirements
(`specs/requirements/`), design (`specs/design/`), ADRs (`specs/decisions/`), this
file, tests, implementation, comments. Each requirement has an ID (e.g. `DR-012`).

- Ambiguous or contradictory requirement → stop and report `SPEC_CONFLICT`; ask the
  human. Never implement a guessed interpretation.
- Spec changes require human approval. Implementation and tests may proceed
  autonomously only within an approved spec.
- Upstream behavior is a contract: changes to existing keywords, file formats, or
  default numerics need an ADR and an explicit regression-suite result.

## Scientific standard

Correctness, traceability, reproducibility, and verification outrank speed.
Never invent reference values, loosen tolerances merely to pass, hide instability
(e.g. by thermostat damping or clamping), regenerate golden files to make a test
pass (`--update-golden` only with human approval and a stated reason), or call a
regression comparison "validation". Keep verification (analytic/limiting cases),
regression (baseline vs candidate binary), and validation (DFT/experiment) separate,
and document the source, settings, and uncertainty of every reference value.

## Required workflow

1. Read the task and all linked requirements, design notes, ADRs, and existing tests;
   read `developer/README.md` before adding keywords, potentials, ensembles, or actions.
2. State affected requirement IDs and planned files; get the plan approved.
3. Add a failing test first when practical (`tests_pytest/`, or a new case in
   `tests_regression/manifest.json` for run-level behavior).
4. Make the smallest conforming change; keep CUDA and HIP paths in sync via
   `gpu_macro.cuh`, and mirror changes into `main_replica/` copies where applicable.
5. Build warning-clean (`make CUDA_ARCH="-arch=sm_86" -j4`), run clang-format, focused
   pytest, and `run_regression.py --suite full` against the `master` baseline binary.
6. Check, as applicable: units, energy/force/virial consistency (finite difference),
   energy conservation in NVE, translational/rotational invariance, atom-order and
   periodic-image invariance, charge neutrality, Ewald/PPPM convergence, float vs
   double agreement, and determinism across runs.
7. Update `specs/traceability.yaml`, the `doc/` manual for user-facing keywords, and examples.
8. Obtain independent reviews (see Delegation) for any force/physics/kernel change.
9. Report per "Completion report".

## Delegation to subagents

After the plan is approved, delegate the actual implementation (edits, builds, test
runs) so the main thread is not filled with build/test noise. The main thread owns
task understanding, spec decisions, plan agreement with the human, and the final
report. This does not relax any other rule in this file.

- **Agent types.** Implementation → `subagent_type: "gpumd-implementer"`
  (`.claude/agents/gpumd-implementer.md`: no `Agent`, `WebFetch`, `WebSearch`) with
  `isolation: "worktree"` by default, so the parent reviews the real diff before
  merging it. Do not skip the worktree merely for convenience. Investigation/survey
  only → `subagent_type: "Explore"` (cannot edit), not `fork`/`general-purpose`.
- **Verify, don't trust.** Re-run/inspect `git status`, `git diff`, test and
  regression output yourself before accepting a subagent's "tests pass".
- **Independent review on a separate lineage.** Reviews run as fresh (non-fork)
  agents from `.claude/agents/`: `numerical-gpu-reviewer`,
  `physics-dielectric-reviewer`, `api-compat-reviewer`. Give them the diff and
  specs, not the implementer's conclusions.
- **Prompts are scope contracts.** State files/specs in scope, files out of scope,
  that commits and destructive git operations are forbidden unless the human asked,
  and that ambiguity means stop with `SPEC_CONFLICT`.
- **No overlapping concurrent delegation.** Confirm parallel subagents touch
  disjoint files.
- **GPU contention.** Only one subagent at a time runs GPU-heavy builds/tests unless
  the human approves otherwise.
- **Technical backstops** (not a substitute for the above): `.claude/settings.json`
  denies `git reset --hard`, `git commit --amend`, and force-push, and a `PreToolUse`
  hook warns on edits under `specs/` or `fixtures/golden/`.

### Gated, staged measurement

Before delegating an expensive study (long MD, PPPM/Ewald convergence scans, NEP
training, many GPU-hours), first design one cheap diagnostic that could predict
NO-GO (a small system, a few steps, a single number; minutes, not days). The
delegation prompt must (1) list the stages in order, cheapest first, (2) give each
stage a numeric gate criterion, and (3) say "if the criterion is not met, stop there
and report". Later stages start only after the parent reads the earlier result and
the human agrees.

### Keep command output short

Long logs stay in context for every later turn. Pipe builds through
`2>&1 | tail -30`, use `pytest -q` and `-m "not slow"`/`-k` for focused runs, and add
`--output-on-failure`-style verbosity only on the failing subset. Run only the tests
of the area you changed; run `run_regression.py --suite full` once, before the final
report. Large files (> ~40 KB) are read in parts (`grep -n`, then `Read` with
`offset`/`limit`), never whole.

## Git policy

Do not commit unless asked. Never force-push, rewrite history, amend a human commit,
or commit generated outputs (`*.out`, dumps, trained `nep.txt`, regression run dirs).
One atomic commit per task, with requirement IDs in the message.

## Generated outputs

Keep generated files out of the tracked source tree. `run_regression.py` writes
`tests_regression/.work/` and `tests_regression/reports/` (both git-ignored; `--keep-all`
keeps them, so delete them when done). Run manual `gpumd`/`nep` jobs in a scratch
directory, never in the repository root or `src/`. Binaries (`src/gpumd*`, `src/nep*`)
are git-ignored. Running `gpumd` directly inside `examples/<name>/` overwrites or appends
to tracked outputs there (e.g. `examples/gpumd_static/dump.xyz` and `neighbor.out`):
copy the example to a scratch directory and run it there; if tracked files were
modified, restore them with `git checkout -- <files>` and never commit them. Before reporting a task done, `git status`
must show no `*.out`, dumps, `nep.txt`, `*.h5`/`*.npy`, or run directories; the
`PostToolUse` subagent audit hook flags them.

## Known test results (not regressions; do not investigate unless your change touches them)

Measured 2026-11-02 on RTX A6000 (sm_86), CUDA from `/usr/local/cuda`, built with
`make CUDA_ARCH="-arch=sm_86" -j4` (this machine is sm_86, not sm_89):

- `pytest -m "not slow"` in `tests_pytest/`: 738 passed, 480 skipped, 0 failed (about 10 min).
  Python deps are not installed system-wide; use the conda env
  `/home/katouda/miniforge3/envs/py312-gpumd` (`ase calorine numpy pytest hypothesis`), e.g.
  `/home/katouda/miniforge3/envs/py312-gpumd/bin/python -m pytest -m "not slow" -q`.
- `run_regression.py --suite full`, once with the SAME binary as baseline and candidate and
  once with a `master` build (`3b2c8db8`, identical `src/` to `develop`) as baseline:
  both gave 218 passed, 37 failed, relations 25/25 passed, and exactly the same 37
  failing cases. Identical sources failing identically means these are run-to-run
  differences (non-determinism on this GPU), not regressions: `nvt_bdp`, `npt_scr_iso`,
  `heat_bdp`, `deform_npt_tri`, `pimd_scr`, `replicate_default_velocity`,
  `replicate_velocity_unseeded`, `graphene_hac`, `graphene_hnemd_shc`,
  `graphene_nemd_shc`, `silicon_liquid_transport`, `ilp_{nep,tersoff,tmd_sw}_compute`, and
  the `train_*` NEP/qNEP/TNEP/vdW training cases. A change is suspect only if a case
  OUTSIDE this list fails, or if a listed case fails in a way other than small numeric
  differences. Re-measure the list when the GPU, CUDA version, or flags change.

## Test execution policy

Tests longer than a few minutes are marked `slow` (pytest) and excluded from the
default quick loop; they still must pass before completion. Timeouts detect hangs
and must never paper over an unresolved hang or nondeterminism.

## Completion report

Report requirement IDs, files changed, exact commands, results, tolerances,
conservation/consistency diagnostics, GPU arch and CUDA version used, limitations,
residual risks, and the next task. A task is incomplete while any acceptance
criterion or BLOCKER/MAJOR review finding is open. Traceability `status` is one of
`implemented`, `partial`, `planned`; never report `partial` as implemented.

## Build

Needs `nvcc` and a GPU; build from `src/` with GNU Make:

```bash
cd src
make CUDA_ARCH="-arch=sm_86" -j4   # default is sm_60, which fails on CUDA 13.x; run `make clean` after changing arch/flags
make -f makefile.hip               # AMD / ROCm
make gnep                          # optional target, not part of `all`
```

`make` (target `all`) builds `gpumd nep gpumd_replica tb`, outputs in `src/`. A CMake build (`CMakeLists.txt`, `-DPKG_NNAP=yes` for NNAP) is also available. Both build systems glob sources by directory (`src/force/*.cu` and `src/force/*/*.cu` for gpumd; `src/main_nep/**` for nep), so new files in existing dirs need no build edits, but nested deeper dirs do. The Linux makefile enables many `-W` flags; keep code warning-clean. Formatting uses clang-format (`src/apply_clang_format.sh`).

## Tests

All tests need a GPU and built binaries (`src/gpumd`, `src/nep`).

- `tests_pytest/` — main pytest suite (needs `ase calorine numpy pytest hypothesis`). Run from that dir:
  `pytest -q`, `pytest -m "not slow" -q`, `pytest -m fast -q`, single test: `pytest test_parsing.py::test_name -q`. `--update-golden` regenerates frozen references in `fixtures/golden/` (deliberate only).
- `tests_regression/` — manifest-driven behavior contract suite comparing a baseline vs candidate binary. Acceptance is `--suite full`; focused suites (e.g. `parsing_validation`, `pppm`) are diagnostics only. See `CONTRACT_SUITES.md`:
  `python3 run_regression.py --repo-root <repo> --baseline <gpumd_old> --candidate <gpumd_new> --suite <name> --keep-all`
  Unit tests of the runner itself: `pytest tests_regression/test_runner.py` etc.
- `tests_tb/` — tests for the `tb` executable.
- CI (`.github/workflows/`) builds with CUDA containers; the integration workflow uses calorine + pytest.

## Architecture (gpumd)

`Run` (`src/main_gpumd/run.cu`) owns `Atom`, `Velocity`, `Box`, `Group[]`, `Force`, `Integrate`, `Measure`. `RunInput` tokenizes `run.in`, and each keyword is dispatched to the module that owns it — implement new keywords in that module, not in `Run`, unless they change overall execution flow.

- `Force` (`src/force/`) holds `Potential` subclasses (NEP, qNEP/charge with PPPM/Ewald, Tersoff, EAM, LJ, ILP, DFT-D3, …). Keep per-model data and GPU workspaces in the potential object; avoid reallocating in `compute()`.
- `Integrate` (`src/integrate/`) holds the active `Ensemble` (`compute1`/`compute2`).
- `Measure` (`src/measure/`) holds `Action` objects (dumps, computes, observers) hooked into the MD loop: `pre_run, setup_force, post_integrate1, pre_force, post_force, end_of_step, post_run`. Use the matching hook rather than special-casing the loop.
- Other dirs: `model/` (atoms/box/groups), `minimize/`, `phonon/`, `mc/`, `utilities/` (`GPU_Vector`, `gpu_macro.cuh` CUDA/HIP abstraction, `error.cuh`).
- Polymorphic objects are owned via `std::unique_ptr`. Code shared between CUDA and HIP must go through `gpu_macro.cuh`.

The `nep` executable (`src/main_nep/`) has its own training `Potential` interface, separate from the MD `Potential` in `src/force/`; spin training in `main_nep/nep_spin/` shares headers with `src/force/nep_spin/`. `main_replica/` has its own copies of force/integrate/model subsets (see its makefile). When changing a user-facing keyword, update the manual in `doc/` too.

Other top-level dirs: `examples/`, `potentials/` (potential files), `tools/` (Python/format-conversion/analysis utilities), `scripts/`.
