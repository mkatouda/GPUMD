# DES-0005: charge-conserving qNEP and its exact BEC (design for DR-005)

- Status: Approved by the human on 2026-11-11 (section 8 items accepted as proposed; ADR-0006). Stage 2a authorized; later stages follow the gates of section 6. No code written yet. Requirement: `specs/requirements/DR-005-qnep-bec-gradient-consistency.md` (D1 = option (a)). Evidence: `specs/verification/VER-0028-qnep-bec-gradient/`.
- ADR: ADR-0006 (to be written with this design; existing models and their BEC stay bit-identical).
- Items marked [CHECK] are read from memory of the code and must be confirmed in stage 2a before use.

## 1. What must hold (from VER-0028)
Let q_a be the charges the model uses (in the energy, the forces, `dump_xyz charge`) and P = sum_a q_a r_a. With a local model (q_a depends only on atoms
within a cutoff of a), the local, translation-invariant BEC Z_k = dP/dr_k = q_k 1 + sum_a (r_a - r_k)_MIC (x) dq_a/dr_k exists as an exact gradient only if
(i) sum_a q_a = 0 identically, with no mean subtraction (the mean subtraction makes dq_a/dr_k nonlocal, Jn = J - S/N), and
(ii) the BEC kernel uses the exact gradient form of section 3 (terms A + B), not 0.5 T1 + 0.5 T2.
   CORRECTION 2026-11-11 (stage 2a, VER-0029; pending the human's confirmation): the earlier wording "T2 only" is exact only when rc_desc + rc_phi < L/2; the A + B form is exact for
   any box that passes the usual per-cutoff checks. The diagonal q_k 1 is already contained in term B (phi 1 part): do not add it again in the kernel. Z is not symmetric: store all 9 components, index [3 beta + alpha] = dP_beta/dr_alpha.

## 2. Model: charges as a graph-Laplacian transfer of a per-atom potential
Keep the per-atom network of qNEP unchanged. Its charge output (today the charge q_a) is reinterpreted as a scalar potential h_a. New charges:

  q_a = sum_{b in N(a)} phi_ab (h_a - h_b),   phi_ab = phi_{t_a t_b}(r_ab) = phi_{t_b t_a}(r_ab) >= 0, r_ab <= rc_phi.

- sum_a q_a = 0 exactly (phi symmetric). Charge flows from high to low h, like electronegativity-driven transfer. No `zero_total_charge` is needed in this mode.
- Validity rule (VER-0029): the a-centered sums use only distances up to rc_desc (term A) and rc_phi (V_a, term B), so the usual per-cutoff box checks suffice; rc_phi + rc_desc < L/2 is NOT required.
- Locality: q_a depends only on a's own neighbors b and on h_b, which depend on b's neighbors. Hence q_a depends on atoms within rc_phi + rc_desc of a.
  For the BEC this is harmless, see section 3 (the sum is rewritten as a-centered sums).
- phi: one radial function per symmetric type pair, phi(r) = sum_k c_k^{pair} f_k(r) fc(r) on the existing radial basis (`basis_size_radial + 1` functions k = 0..basis_size_radial, as in the NEP radial descriptor; 2026-11-11 stage 2b, approved by the human), with a cutoff rc_phi
  (an extra number in nep.txt; default proposal rc_phi = radial cutoff). Positivity is not enforced (not needed for the math); note it for fitting.
- Why this form: (a) exact Sum q = 0 and locality by construction, (b) existing network and descriptors are reused, (c) NEP training is gradient-free (SNES), so the
  training side needs forward kernels only (no second derivatives).
- Known risk: q = L h with a local h limits what neutral charge patterns can be represented; BEC accuracy may degrade (A6). Alternatives in section 7.

## 3. Exact BEC without new derivative machinery
P = sum_a q_a r_a = sum_{pairs a<b} phi_ab (h_a - h_b)(r_a - r_b)   (MIC r_ab; every term depends only on the pair vector, so P is translation invariant and single valued)
  = sum_a h_a V_a,   V_a = sum_b phi_ab (r_a - r_b)   (a vector, MIC).
So dP_beta/dr_k alpha = sum_a (dh_a/dr_k alpha) V_a,beta + sum_a h_a dV_a,beta/dr_k alpha.
- Term A: for each Cartesian beta, it is the force-like contraction of the network derivative dh_a/dq_desc with the per-atom weight w_a = V_a,beta.
  This is exactly what `find_force_radial/angular` already do for the electrostatic part (weight `g_D_real`, `charge_derivative`, `sum_fxyz`; nep_charge.cu around lines 1015-1100 [CHECK]).
  Plan: call the same kernels three times with w = V_x, V_y, V_z and Fp = 0, writing dP_beta/dr_alpha instead of forces. Cost: 3 force-like passes.
- Term B: a pair loop over b within rc_phi: dV_a/dr_k = phi_ab' (r_ab/|r_ab|) (x) r_ab-type terms and phi_ab 1: closed form.
- The sums run over a with k a neighbor of a (descriptor derivatives) or of the pair (a,b), all centered on a: no two-hop MIC. Requirement: box thickness check for rc_phi and rc_desc as for the
  current kernels (small-box kernels get the same change; the small-box path is used when the box is thin).
- Energy and forces in this mode: D_c (= dE_el/dq_c from the long-range solver) enters through q_a = sum phi_ab (h_a - h_b): dE/dh_c = sum_b phi_cb (D_c - D_b) =: Dt_c,
  which is used as the weight `g_D_real` in the existing force kernels (so `zero_mean_D_real` is replaced by this Laplacian-transposed weight), plus the explicit pair term
  with phi'(r_ab)(h_a - h_b)(D_a - D_b) in the pair force. Forces and virial must pass the finite-difference check (energy conservation, AGENTS.md item 6).
- `dump_xyz bec`, `bec.out`, `compute_dpdt` and `add_efield ... bec` all use the same BEC array, so they change only through the new model [CHECK that no code path recomputes the BEC separately].
- The scaling by sqrt(eps_inf) (`scale_bec`) stays; it is a constant factor and does not change integrability.

## 3b. Diagnostic output of the direct dipole (added 2026-11-11 at the human's approval; interface to be fixed in stage 2c)
For `charge_transfer` models P = sum_a h_a V_a = sum_{pairs} phi_ab (h_a - h_b) r_ab (MIC) is single valued and continuous (phi -> 0 smoothly at rc_phi), also in periodic cells,
where sum q r with unwrapped positions is multivalued. Term B of section 3 already forms V_a, so P costs one extra reduction.
- Proposal: an optional extra column/file of `compute_dpdt` (name and syntax open; default off so existing outputs, files, and results of all models stay unchanged) giving the direct P(t) per group.
- Purpose: A2/A3 in periodic systems (crystal, amorphous): compare P_direct(t) - P_direct(0) with the integral of BEC.v dt; the difference must stay at the noise level (chain-rule identity),
  and <dP^2> of P_direct itself must saturate. It also serves the requester's metrics (chain rule, BEC vs gradient, <dD^2> saturation).
- Not available for models without the option (error if requested). Float vs double: accumulate in double.

## 4. File formats and keywords (all additive)
- `nep.in`: new keyword, proposed `charge_transfer <rc_phi>` (orthogonal to `charge_mode`/`model`; initial scope: charge_mode 2 and 3 (qnep2, qnep3), no vdw variants, no spin). Name and syntax are open (section 8).
- `nep.txt`: the first-line model name is kept; a new line `charge_transfer rc_phi` follows the cutoff block [CHECK the exact parser order, `NEP_Model_Input`], and the phi coefficients are appended after
  the existing parameter block. A file without that line is read exactly as today, so existing models give bit-identical results (ADR-0006).
- `dump_xyz ... bec`, `compute_dpdt`, `add_efield`: no keyword change; `doc/` pages state that BEC is the exact gradient of P for `charge_transfer` models.
- Training side: `src/main_nep/nep_charge/` and `common/nep_long_range.cu` get the same kernels as forward functions (definitions in one header shared by gpumd, `main_replica`, and `nep` where the
  build allows, to keep A4: BEC of `nep` and `gpumd` equal to 1e-5 e).

## 5. Files to change (stage 2 onwards; none changed yet)
- `src/force/nep_charge/nep_charge.cu`, `src/force/nep_charge/nep_charge_small_box.cuh` (charge formation, D-weight, BEC passes)
- `src/force/nep_charge/nep_charge.cuh`, a new `nep_charge_transfer.cuh` (phi evaluation, Laplacian, V_a) [new files in an existing dir need no build edits]
- `src/main_replica/force/nep_charge.cu`, `nep_charge_small_box.cuh` (mirror)
- `src/main_nep/nep_charge/nep_charge.cu`, `nep_charge_parameters`, `parameter_input.cu`, `common/nep_long_range.cu`, compiled kernels [CHECK whether specialized kernels are needed]
- `doc/gpumd/input_parameters/{add_efield,compute_dpdt,dump_xyz}.rst`, `doc/nep/input_parameters/` (new keyword page), `specs/traceability.yaml`
- Tests: new `tests_pytest/test_qnep_charge_transfer.py`, new regression cases only if human agrees; existing cases must be unchanged (A5).

## 6. Verification plan (gated; cheapest first)
| stage | content | gate (stop and report if not met) |
|---|---|---|
| 2a | numpy reference of the model on small systems (random weights; 8-59 atoms, thin and thick boxes): q, P, exact Z_k by autodiff/FD in float64; check sum q = 0, translation/rotation invariance, Z_k = dP/dr_k, and that T2-only with this L equals the pair-sum gradient | FD vs analytic <= 1e-8 (float64) |
| 2b | gpumd kernels (normal and small-box), `dump_xyz bec` vs FD of P of the gpumd model itself and vs the numpy reference | RMS <= value fixed by the human at D2 (proposed 1e-3 e as DR-005 A1); float vs double reported |
| 2c | forces/virial vs FD, the direct-P diagnostic output (section 3b) vs the numpy reference and vs the integral of BEC.v (chain rule), NVE energy conservation, determinism, atom-order invariance, existing qNEP unchanged (bit identity) | per AGENTS.md step 6 |
| 3  | replica copies, training-side kernels, A4 | 1e-5 e |
| 4  | `run_regression.py --suite full` once (only the 37 known cases fail), pytest, docs | A5, A7 |
| 5a | expressiveness check: short NEP training (small generations, same data and budget as the current N50 run, 1 seed) against the baseline of the same budget | held-out BEC RMSE <= 2x baseline; if not met stop and report to the human (alternatives, section 7) |
| 5b | full retraining N50, 10000 generations, 3 seeds (A6) | report only |

## 7. Alternatives considered
- Charge from a pair network with bond-charge increments only (t_ab = phi_ab(r), no h): also exact, but charges depend only on the pair distances; too weak (no environment dependence). Rejected as the first choice, may serve as a fallback.
- t_ab = phi_ab(r) (h_a - h_b) with richer environment-dependent phi (a pair network from two atom descriptors): more expressive, but higher cost and a new network input; considered if 5a fails.
- Direct dipole model (`dump_dipole`, nep4_dipole): option (b) of D1, not taken; remains the report item of the request (3-6).
- Keeping the mean subtraction and the 0.5/0.5 form: not a gradient (VER-0028). Not acceptable for DR-005.

## 8. Open points for the human
1. Name and syntax of the keyword and the nep.txt line (`charge_transfer rc_phi`); scope limited to charge_mode 2/3.
2. rc_phi: equal to the radial cutoff (proposal) or an independent value.
3. Whether 5a (short expressiveness training) may run before the kernels are finished on the training side (needs the training forward kernel; otherwise the gate comes after stage 3).
4. D2 values (A1: 1e-3 e proposed; A4: 1e-5 e proposed).
5. Whether the modes (vdw, charge_vdw, spin) are excluded in this task (proposed: yes).
