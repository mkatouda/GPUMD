# DR-008: run-time stability guard (`check_stability`) — stop and report when a simulation leaves the physical regime

- Status: **Approved 2026-11-21** (human approved all defaults of the "Decided points").
- Origin: polymer-apps study S-005 (PTFE qNEP, AC-field NEMD). Zero-field NVT of the 248-atom amorphous cell with the R6-2 model diverged (s2: F-F 0.67 A, T 2.5e8 K, CUDA error 700) or collapsed silently (s3: at about step 7300 T rose to 2145 K, U fell by about 1270 eV, the thermostat returned the temperature to 300 K and the run continued for the rest of its length with no error), starting from CF3 end groups (F-F 2.0 -> 1.6 A, energy monotonically falling). The failure is non-deterministic (4 of 4 identical-input runs collapsed, a fine-dump rerun did not). Evidence: `GPUMD-polymer-apps/studies/S-005-ptfe-qnep-prototype/README.md`, section "R6-2 seed 2 の発散の再現と T6 γ"; `outputs/.../retest/ffscan/`.
- Related: DR-007 (flexible ZBL for charge models) was tried first and is **not sufficient**: see `tmp/zbl_guard_plan_20261121/stage0_result.md` (F-F ZBL energy at 1.6 A is 0 / 0.50 / 1.48 eV for outer cutoffs 1.6 / 1.8 / 2.0 A against a no-ZBL energy fall of 21-65 eV in the trigger region; charges are unchanged by ZBL and diverge below F-F 1.6 A). `compute_extrapolation` does not support qnep and gave one frame of warning at most.
- Upstream behavior is a contract: the guard is opt-in (no keyword, no change); existing keywords, output files and numerics are unchanged without it.

## Problem
1. A run that has entered an unphysical state is either aborted by a CUDA error far from the cause (no diagnostic, last frames lost) or, worse, continues in a collapsed state under a thermostat and contaminates every later observable (the s3 case). A user can only detect it afterwards, from thermo output, if at all.
2. No keyword stops a run on a physical criterion. Existing thermo and dump outputs are written at a fixed interval and do not carry the information needed to find the trigger.

## Requirement
R1. New keyword `check_stability` (added to the `run.in` of a run, like other `Measure`/`Action` keywords; implemented in the module that owns it under `src/measure/`, registered the standard way). Syntax:

    check_stability <interval> [max_temperature <T>] [max_potential_change <eV/atom>] [min_distance <A>] [max_force <eV/A>] [max_charge <e>] [keep_frames <n>]

Every `<interval>` steps the guard evaluates the enabled criteria. At least one criterion must be given. The criteria are:
- `max_temperature`: instantaneous temperature (from the existing kinetic energy) above the value;
- `max_potential_change`: |U - U0| / N above the value, U0 the potential energy at the first evaluation of the current `run`. This is the only criterion that detects a collapse that the thermostat has already cooled (the s3 case);
- `min_distance`: shortest interatomic distance below the value;
- `max_force`: largest force component magnitude above the value;
- `max_charge`: largest |q| above the value, for charge models (qnep1/2/3) only; an error if the potential has no charges.
R2. On the first violated criterion the run stops with a non-zero exit status through the project's fatal-error helper. The message names the step, the criterion, the observed value, the threshold, and (where applicable) the atom indices.
R3. Before stopping, the guard writes the last `keep_frames` evaluated frames (default 3, including the violating one) with positions, velocities, forces and (charge models) charges to `stability_frames.xyz` in extended xyz, for diagnosis. With `keep_frames 0` it writes only the violating frame. A restart file is not written (the state is unphysical).
R4. Cost: with no `check_stability`, no cost and no code path change. With it, each evaluation is a few reductions (temperature, force, charge) plus, if enabled, one shortest-distance kernel; the target is below 1 % of the step time at interval >= 100 for 250 - 50 000 atoms on the reference GPU (measured, VER-0041).
R5. Scope: single-GPU `gpumd` runs. `multi_replica`, multi-GPU NEP and `minimize` are refused with a clear message if combined (the guard must never be silently inactive).
R6. A run that finishes without a violation behaves bit-identically to a run without the keyword (the guard only reads state).

## Acceptance criteria
- AC-1. Without the keyword the output of every case of the full regression suite and the existing pytest suite is unchanged (full `run_regression.py` against the pre-change develop: the same non-deterministic set only; the guard code is not reached).
- AC-2. Trigger plumbing: for each criterion a small constructed case violates the threshold at a known step: temperature (NVE with large initial velocity), potential change (a model or structure with a known large energy change: for example a two-atom Lennard-Jones or `nep` pair approaching an attractive minimum), minimum distance (two atoms driven together by a constant external force or initial velocities), force, charge (qnep with an imposed short contact; thresholds set below the known value of a static evaluation). The run stops at the first evaluation after the violation (within one interval), the exit status is non-zero, the message names the criterion, value and threshold, and `stability_frames.xyz` contains `keep_frames` frames, the last of which has the violating value.
- AC-3. No false alarm: the same cases with thresholds above the observed values run to completion with output identical (bit for bit in single-run determinism, otherwise within the repeat difference of the same binary) to the run without the keyword.
- AC-4. The s3 scenario: a case where the temperature rises and returns under a thermostat while U has dropped by >= 1 eV/atom is detected by `max_potential_change` and not by `max_temperature` evaluated only at the end (shown with a constructed case, not with the PTFE data).
- AC-5. Cost measured as in R4 on at least a 248-atom and a 40 000-atom case (nsys or step time, median of 3 alternated runs, idle GPU); the result is recorded even if above the target.
- AC-6. Refusals (R5, `max_charge` without charges, no criterion given, non-positive values) are tested.
- AC-7. Build warning-clean (sm_86; HIP paths in sync via `gpu_macro.cuh`), `doc/gpumd/input_parameters/check_stability.rst` and the keyword table in `run_in.rst`, an example under `examples/`, independent numerical-gpu and api-compat reviews without open BLOCKER/MAJOR, `specs/traceability.yaml`.
- AC-8 (outside the repository, polymer-apps): the guard configured for the 248-atom amorphous cell stops one of the known-divergent R6-2 s2/s3 inputs (same input, several tries; non-deterministic) with the criterion and the F-F pair named, and does not trigger on the stable models (R5, R6-1, R6-2 s1). Recorded in S-005; a failure to reproduce the collapse in the tries is not a failure of the guard.

## Out of scope
- Preventing the collapse (retraining, CF3 sampling, a charge clamp, ZBL tuning); recovery or automatic restart; the nep trainer; `compute_extrapolation` for qnep (a separate request); the guard for multi-replica or multi-GPU runs.

## Decided points (human, 2026-11-21: all defaults approved)
1. Criteria set: the five above; `max_potential_change` and `min_distance` are the ones the polymer-apps case needs. Default: all five implemented, each optional.
2. Shortest distance: use the force object's neighbour list if it exposes one for all supported potentials, otherwise a dedicated cell-list kernel. Default: dedicated kernel, independent of the potential, evaluated only at the check interval (simple, correct under any potential, cost bounded by the interval).
3. Action: stop with non-zero status and write `stability_frames.xyz`; no restart file. Default: as stated (R2, R3).
4. Thresholds have no default values; the user must give them (a default would silently change physics-relevant behaviour for some systems).
5. `keep_frames` default 3.
6. `U0` reference: first evaluation of the current `run` (not the step-0 energy of the whole script). Default: as stated.
7. Name: `check_stability`. Default: as stated.
8. Where it lives: `src/measure/` as an `Action`, so that it participates in the standard lifecycle (`post_force`/`end_of_step`), with a charge accessor added to the potential interface only if needed for `max_charge` (keep it a narrow, read-only accessor on the charge force classes). Default: as stated; the design note decides the accessor.

## ADR
No ADR: the change is opt-in (a new keyword / an optional token), existing keywords, file formats and default numerics are unchanged (AC-1 / AC-2 prove it by the full regression); design choices are recorded in the verification record.
