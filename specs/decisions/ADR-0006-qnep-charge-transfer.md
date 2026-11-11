# ADR-0006: `charge_transfer` option for qNEP; existing models and BEC unchanged

- Status: Accepted (approved by the human on 2026-11-11, "as proposed" for the items of DES-0005 section 8)
- Requirement: DR-005 (design DES-0005; evidence VER-0028)
- Needed because: AGENTS.md treats changes to existing keywords, file formats, and default numerics as an upstream contract change.

## Context
VER-0028 showed that the current qNEP BEC (kernel formula 0.5 T1 + 0.5 T2 with mean-subtracted charges) is translation invariant but not the gradient
of P = sum q r, and that a local exact gradient needs sum_a q_a = 0 by construction (DR-005 D1 = option (a)). Existing models (N20/N50 and any user model)
and their energies, forces, virials, and BEC must not change.

## Decision
1. New, additive option: `charge_transfer <rc_phi>` in `nep.in` (`nep`) and a matching `charge_transfer rc_phi` line plus appended phi coefficients in `nep.txt`.
   The charge definition becomes q_a = sum_b phi_ab (h_a - h_b), h_a = the charge output node of the per-atom network (DES-0005 section 2).
2. Scope: `charge_mode 2` and `3` (`qnep2`, `qnep3`) only. Not supported (input error naming the combination): `charge_mode 1`, `vdw`, `charge_vdw`, spin,
   `model_type` models. `rc_phi` defaults to the radial cutoff (proposal accepted); an explicit value larger than the radial cutoff is an error unless the box checks of DES-0005 section 3 hold.
3. Compatibility (the contract):
   - An `nep.in` without `charge_transfer` and an `nep.txt` without the new line are read and evaluated exactly as before: energies, forces, virials, charges, BEC bit-identical
     to the current binary (checked in stage 2c / A5; the `master` baseline and the 37 known non-deterministic regression cases excepted as in AGENTS.md).
   - No change of the BEC formula, `zero_total_charge`, or `scale_bec` for models without the option.
   - Old models cannot be converted to the new form (a different charge definition); a model with the option must be trained.
4. In `charge_transfer` models: no `zero_total_charge` and no `zero_mean_D_real`; the BEC kernels use the exact reference-k form of DES-0005 section 3 in all paths
   (normal, small-box, `main_replica`, training side); `dump_xyz ... bec`, `bec.out`, `compute_dpdt`, `add_efield ... bec` read the same array; no keyword change for those.
5. A binary that does not know the option must fail clearly on a new `nep.txt` (the unknown first-line/extra-line must not be silently ignored): to be checked in stage 3.
6. Documentation: new keyword page in `doc/nep/input_parameters/`, notes in the `add_efield`, `compute_dpdt`, `dump_xyz` pages (DR-005 A7).

## Consequences
- Existing users see no change. New models need retraining (A6) and may have lower BEC and E/F accuracy than the current ones (BEC within about 2x of 0.073 e is the guide; E/F not gated).
- A4 (training vs gpumd BEC, 1e-5 e) and A1 (FD, 1e-3 e) values are as proposed in DR-005 and approved with this ADR.
