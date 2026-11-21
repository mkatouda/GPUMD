# DR-009: optional `seed` for the stochastic thermostats (reproducible random streams)

- Status: **Approved 2026-11-21** (human approved all defaults of the "Decided points" and the scope nvt_bdp and npt_scr only).
- Origin: polymer-apps S-005. `nvt_bdp` seeds its generator from the clock unless the binary is built with `-DDEBUG` (`src/integrate/ensemble_bdp.cu:33-36`); the same holds for `npt_scr` (`ensemble_npt_scr.cu:35-38`), PIMD (`ensemble_pimd.cu:43-46`) and the MC ensemble (`mc_ensemble.cu:79-82`). `velocity <T> seed <n>` already exists. Without a seed keyword, a user cannot repeat a stochastic-thermostat run, which hampers bisecting a rare divergence (S-005 R6-2) and recording seeds per study as the polymer-apps rules require.
- Related: DR-008 (stability guard; a reproducible stream helps to repeat a failing run).
- Upstream behavior is a contract: without the new option the behavior is unchanged (clock seed in release builds, fixed seed in DEBUG builds).

## Requirement
R1. `ensemble nvt_bdp ... [seed <n>]` and `ensemble npt_scr ... [seed <n>]` accept an optional positive integer seed that initialises the thermostat generator deterministically. Without it: unchanged.
R2. PIMD and the MC ensembles get the same option only if the maintainers' owners agree (proposed: follow-up, not in this requirement).
R3. The documentation states what is and is not reproduced: the seed fixes the thermostat random stream only. GPU force accumulation (atomic additions) and other sources of run-to-run differences remain, so identical trajectories are not guaranteed (the project already records identical-source runs differing on this GPU).
R4. A negative, zero or non-integer seed is an input error.

## Acceptance criteria
- AC-1. Same seed, same input, same binary: the sequence of random numbers drawn by the thermostat is identical (tested on the first steps of a run in a way that does not depend on force noise: for example NVT of a non-interacting system, or a frozen-force toy potential such as a harmonic or zero potential, where the whole trajectory is then reproducible); different seeds give different sequences; no seed keeps the old clock-seeded behaviour (two runs differ).
- AC-2. Existing cases are unchanged (full regression: the same non-deterministic set).
- AC-3. Refusals (AC R4) tested; docs (`ensemble.rst`) and a note on R3.
- AC-4. Build warning-clean; reviews as for any source change.

## Out of scope
Deterministic force summation (a performance cost, separate decision), seeds for velocity creation (exists), Langevin-type ensembles unless the same pattern is found there (check during design; add only if the clock seed exists).

## Decided points (human, 2026-11-21: all defaults approved; scope nvt_bdp and npt_scr only)
1. Syntax: trailing optional `seed <n>` on the ensemble line (same word as `velocity ... seed`). Default: as stated.
2. Scope: `nvt_bdp` and `npt_scr` now (the two used by the fork's studies); PIMD and MC later. Default: as stated.
3. DEBUG builds: an explicit seed overrides the fixed DEBUG seed.

## ADR
No ADR: the change is opt-in (a new keyword / an optional token), existing keywords, file formats and default numerics are unchanged (AC-1 / AC-2 prove it by the full regression); design choices are recorded in the verification record.
