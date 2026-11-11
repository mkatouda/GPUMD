# DR-005: qNEP BEC consistent with the gradient of the dipole P = sum_a q_a r_a

- Status: DRAFT (2026-11-11). Not approved as a whole. Stage 1 (diagnosis, no `src/` change) was authorized and is DONE (VER-0028). D1 decided by the human on 2026-11-11: option (a) (see below). DES-0005 and ADR-0006 approved 2026-11-11; stage 2a (numpy reference, no `src/` change) authorized; later stages need their gate results first.
- Design: `specs/design/DES-0005-*.md` (to be written after stage 1)
- ADR: `specs/decisions/ADR-0006-*.md` (needed before any change of the BEC numerics)
- Verification: `specs/verification/VER-0028-qnep-bec-gradient/`
- Origin: work request from GPUMD-polymer-apps S-005 (`/home/katouda/work/dielectric-response_pj/GPUMD-polymer-apps/docs/work-requests/gpumd-bec-gradient.md`, 2026-11-11). Data tier of the requester: Tier P (prototype, not a physical prediction).
- Related: none of DR-001..004. Upstream behavior is a contract (AGENTS.md): this requirement changes a fork feature (qNEP BEC); the change must be shown not to affect energies, forces, and virials.

## Problem (from the request; corrected by stage 1, VER-0028)
In AC-field NEMD (`add_efield <gm> <gid> <file> bec`, `compute_dpdt`) of PTFE with a qNEP(BEC) model, a nearly
frequency-independent spurious dissipation appears (omega*chi'' about constant over 20-500 GHz). At zero field,
P = integral sum BEC.v dt does not saturate even for a crystal (<dP_z^2> = 1.58, 5.66, 22.9, 62.2 e^2 A^2 at lag 1/5/20/100 ps).
Stage 1 (VER-0028, reproduced the requester's numbers within 5 %) established:
- The kernels (normal, small-box; the training side by reading) compute BEC_k = s [ qn_k 1 + 0.5 T1_k + 0.5 T2_k ]
  from the derivative of the RAW charge (s = sqrt(eps_inf); qn = q - mean q). This formula is exactly translation invariant (sum_k Z_k = 0)
  but is not the gradient of P = sum qn r. The neutralization term is in the forces only.
- The request's "(0.5, 0.5) fits the amorphous system but not the oligomer" was an artefact of using the Jacobian of the neutralized charge
  (Jn = J - S/N, S_k = d(sum raw q)/dr_k, nonlocal). With the raw Jacobian both fit a = b = 0.500.
- A strictly local BEC is an exact gradient of a translation-invariant P only if S_k = 0, i.e. the model conserves total charge locally.
  The current model does not (S_k != 0, RMS 0.14-0.16 e/A).

## Requirement (target; definition fixed by D1)
The BEC computed by `gpumd` (`dump_xyz ... bec`, `bec.out`, `compute_dpdt`, `add_efield ... bec`) shall be the
derivative of one single-valued dipole function P(r) of the model, in all code paths: the normal kernels, the
small-box kernels, the `main_replica` copies, and the training side (`nep`, `src/main_nep/`), so that `nep` and `gpumd` give the same BEC
for the same structure and `nep.txt`.

## Decisions
D1 (decided by the human, 2026-11-11): option (a). The target is a model in which the raw charges conserve the total charge locally
    (S_k = d(sum_a q_a)/dr_k = 0 by construction), so that the local BEC Z_k = q_k 1 + sum_a (r_a - r_k)_MIC x dq_a/dr_k is the exact gradient of a
    single-valued, translation-invariant P. Note (VER-0028 1-d): even with S_k = 0 the present pair weights (0.5 T1 + 0.5 T2) are not the exact gradient
    (C1, T2 only, differs from the kernel by RMS 0.12-0.19 e), so the BEC kernels of the new model must also change to the exact form of DES-0005 section 3 (T2-only is exact only for rc_desc + rc_phi < L/2; VER-0029 stage 2a confirmed A + B against FD to 2e-9 in float64). This is a model change with retraining (A6), not a kernel coefficient change. Options not taken:
    (b) a direct dipole model (`dump_dipole`/nep4_dipole), (c) compensating the non-conservative part on the field side. The charge-conserving
    parametrization (e.g. antisymmetric pairwise charge transfer) and its `nep.txt` format are to be designed in DES-0005; ADR-0006 must state that the
    existing qNEP models and their BEC stay unchanged (new model/flag, no change of existing numerics).

## Decisions pending (human)
D2. Acceptance values for A1 and A4 (float arithmetic; fixed after stage 1).
D3. Whether to retrain (A6) and how to treat a large BEC accuracy loss.
D4. Whether a keyword or flag is needed to keep the old BEC (ADR-0006).

## Open hypothesis (stage 1, not a finding)
`zero_total_charge` (`src/force/nep_charge/nep_charge.cu`, after `find_descriptor`) subtracts mean(q), while the BEC kernels
use the derivative of the raw charge. If so, the exact gradient of the neutralized P contains a non-local term
(centroid-dependent, size N^-1). Stage 1 tests this (H1) against H0 = the current formula.

## Acceptance criteria (proposed, from the request; values fixed by the human, D2)
- A1. (of the NEW charge-conserving model) Agreement of the BEC with the finite-difference gradient of P = sum q r (with local MIC pair vectors; S_k = 0 checked) for amorphous 248 atoms, oligomer 59 atoms, and crystal 702 atoms
  (at least one structure each): RMS difference <= 1e-3 e (proposed).
- A2. Vacuum oligomer: correlation of dP_q/dt with BEC.v >= 0.99; <dP^2> of the integrated BEC.v saturates like P_q.
- A3. Zero-field crystal and amorphous: <dP^2> of the integrated BEC.v saturates up to 100 ps.
- A4. Same structure and `nep.txt`: BEC of `nep` and `gpumd` agree within 1e-5 e (proposed).
- A5. Existing regression and tests pass (`run_regression.py --suite full` against the `master` baseline: only the 37 known
  run-to-run cases fail, AGENTS.md); energies, forces, and virials of qNEP without BEC are unchanged (bit identity).
- A6. Report (not pass/fail): BEC RMSE after retraining N50 (10000 generations, 3 seeds) on the 20 held-out structures
  against 0.073 +/- 0.012 e (all components), 0.098 e (diagonal). The human fixed (2026-11-11, via the requester's reply) the guide at 2x the current value (about 0.15 e, all components); above it the human judges. E/F: BEC has priority (human, 2026-11-11); E/F accuracy may be looser than the current model, no numeric E/F gate; E/F RMSE of the retrained model is reported next to the BEC result (current N50: force 0.244 +/- 0.033 eV/A, energy 0.010 +/- 0.002 eV/atom). Training inputs (requester's `outputs/S-005-ptfe-qnep-prototype/`: `v3/bec_orb_N{20,50,100}.extxyz`, `bec_orb_heldout.extxyz`, models N20/N50, structures `demo/{pq_cryst,stab_amor1,pq_oligo}/model.xyz`) are for internal development/verification only: no redistribution, no commercial-model training (licence of the Orb-ft2 E/F labels not cleared).
- A7. Documentation: `doc/` pages of `add_efield`, `compute_dpdt`, `dump_xyz` and `traceability.yaml` updated.

## Out of scope
Comparison with experiment, Tier D, the treatment of epsilon_inf, the analysis code of the AC field, and the scripts of
GPUMD-polymer-apps. A2/A3 recovering a bounded P does not guarantee that the spurious dissipation disappears; that check belongs to the requester.

## Stages
1. (authorized) Diagnosis, VER-0028 1-0..1-d, no `src/` change. Gate per step; stop and report if a gate is not met.
2. DES-0005/ADR-0006 (charge-conserving parametrization, `nep.txt` format, training and gpumd kernels), then normal kernels and A1 (after the human approves them and decides D2).
3. small-box, replica, training side, A4.
4. Regression and tests (A5), documentation (A7).
5. Retraining (A6), only after stages 1-4 and the human's agreement (GPU hours).
