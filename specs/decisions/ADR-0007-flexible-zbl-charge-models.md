# ADR-0007: flexible ZBL restored for charge models (qnep1/2/3), including `charge_transfer`; deliberate divergence from upstream

- Status: **Accepted 2026-11-20** (human approved the defaults of the 8 open points of DR-007; text as revised 2026-11-18, develop `fc67a8cc`; original 2026-11-13). Implementation not started.
- Requirement: DR-007
- Needed because: AGENTS.md treats changes to existing keywords, file formats, and default numerics as an upstream contract change, and this change reverses an upstream restriction.

## Context
Upstream removed the flexible-ZBL code from the charge-model kernels (`4508bba2`, 2026-10-29, "simplify kernels") and rejected flexible ZBL for every model but `model nep`
(`dbecce90`, 2026-10-29, "restric flexible zbl", PR #1866); the fork received both with VER-0022; the merge of upstream 5c67f365 (VER-0037) also removed the legacy nep.txt formats, so old-format v5.9.1 files can no longer be read by `gpumd` and must be converted. No reason is recorded in the commits. Upstream v5.9.1 supports
flexible ZBL for charge models, and a v5.9.1-trained PTFE qNEP with per-pair ZBL exists. The requester needs per-pair ZBL for PTFE (the wall distances of C-F bonds and F-F contacts
differ, so one universal outer cutoff or one typewise factor cannot serve both) together with the fork's `charge_transfer` and DFT-D3/D2 features, and has decided not to ask upstream.
The fork's own code assumes that flexible ZBL and `charge_transfer` never coexist (`nep_model_file.cu:326-329` in `validate_md`, `src/main_nep/common/nep_io.cu:376-377`, the replica model code under `src/force/replica/`, and the record VER-0035).

## Decision
1. Flexible ZBL is accepted for qnep1/2/3 (old names `nep4_zbl_charge1/2/3`) in `nep`, `gpumd`, and replica dynamics (`multi_replica`), in the new `nep.txt` format (`zbl 0 0` plus a `zbl_parameters` block, ten values per unordered type pair, `nep_model_format.md` section 4.3), with and without `charge_transfer`; old-format v5.9.1 files are converted by `nep_model_converter.py`; no new keyword.
2. Block order: new-format blocks are named, so no order has to be decided for gpumd, replica, and the trainer (the writer's order and its comment are updated). The converter keeps the legacy value order `parameters`, `q_scaler`, `zbl_parameters`, `phi` (as in `nep_model_converter.py:197-222`).
3. Unchanged restrictions: `nep_dipole`, `nep_polarizability`, NEP-spin reject flexible ZBL; typewise factor and flexible ZBL are exclusive; at most 10 active elements. `nep_charge_vdw` is out of scope unless decided otherwise.
4. Compatibility (the contract): models without flexible ZBL give bit-identical results to the pre-change binary; v5.9.1 flexible-ZBL qNEP files run unchanged.
5. Divergence management: the change is confined to `src/force/nep_charge/`, `src/force/replica/` (charge model and kernels), the reader/`validate_md`/compact code in `src/utilities/nep_io/`, the trainer files and the converter listed in DR-007, and tests; each changed block carries a comment `DR-007`; a guard test fails if a future upstream merge re-inserts the restriction for these models; `specs/upstream-issues/` records that upstream restricts this on purpose and that the fork diverges. No upstream PR is planned.
6. Documentation: the pages listed in DR-007 R7 state that flexible ZBL is available for qNEP in this fork (and that upstream accepts it only for `model nep`).

## Consequences
- The fork keeps flexible ZBL for qNEP at the cost of conflicts at each upstream merge in the touched places (after the 2026-11 release cleanup, upstream's direction is to keep the restriction; the divergence is larger than at the time of the original draft); the guard test makes the loss visible.
- `charge_transfer` + flexible ZBL has no v5.9.1 reference; verification is analytic (equivalence to universal ZBL, independent float64 reference, finite differences, charge/BEC independence of ZBL).
- Alternatives considered: universal ZBL only (cannot serve C-F and F-F at once; the requester's training of a charge-free NEP on PTFE data gave force errors of 51.9/68.4 meV/A (train/test) without ZBL, 60.8/72.1 (+17%/+5%) with a universal `zbl 1.25`, and 52.5/68.1 (+1%/+-0) with per-pair ZBL, so per-pair ZBL was adopted for the production NEP and the same short-range setting is planned for the qNEP; universal ZBL stays the fallback); asking upstream to restore it (not chosen by the requester); backporting the fork's D3/D2 to v5.9.1 (large verification burden, gives up `charge_transfer`).
