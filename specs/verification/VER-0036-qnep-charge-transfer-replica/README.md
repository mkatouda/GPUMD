# VER-0036: charge_transfer qNEP in gpumd_replica (DR-005 stage 3b)

- Status: code written and compiled (sm_86, CUDA from /usr/local/cuda, warning-free in `src/main_replica`); **not yet run on the GPU** (all gates below open). Type: regression of gpumd_replica against gpumd on identical configurations (same synthetic nep.txt, VER-0032 generator). Not validation.
- Requirement DR-005; design DES-0005; ADR-0006; kernels VER-0032/0034 (gpumd side).

## What was done
- `src/main_replica/force/nep_charge_transfer.cuh`: copy (not shared) of the gpumd charge_transfer kernels `ct_find_phi`, `ct_find_charge_{large,small}_box`, `ct_find_Dt_and_pair_force_{large,small}_box` (the header of the main tree includes a different `NEP_Charge`). The BEC kernels are NOT ported.
- `nep_charge.cuh/.cu` (replica): `ParaMB::{charge_transfer, rc_phi, num_phi_pairs, num_phi_per_pair}`, `ANN::phi_coeff`, `NEP_Charge_Data::{phi_parameters, ct_h, ct_Dt, ct_V}`; the parser accepts `charge_transfer rc_phi` right after `cutoff` (charge_mode 2/3 only, 0 < rc_phi <= rc_radial, same error texts as gpumd), reads the phi block after the q_scaler block and requires EOF; flexible ZBL together with charge_transfer is rejected (layout undefined). Buffers are allocated in the constructor, none in `compute()`.
- Force path (large and small box): for charge_transfer models `zero_total_charge` is replaced by h-copy (`copy_from_device_async`, replica stream) + `ct_find_charge_*`; the BEC kernels (`find_bec_*`, `scale_bec`) are skipped; `zero_mean_D_real` is replaced by `ct_find_Dt_and_pair_force_*`, and `ct_Dt` is the weight of `find_force_radial` / `find_partial_force_angular` / small-box equivalents; the pair kernel adds force and virial. All launches and copies use the replica's `stream_`. Models without charge_transfer take the old branch (same launches, same order).
- Batched integration (`replica_batch.cu`) uses `NEP` (energy models) only; qNEP names (`nep4_charge*`) are excluded in `Replica_Runtime::initialize_batches` ("batched integration: off (not supported for qNEP)"), so charge_transfer models never reach it.
- gpumd_replica knows `nep4_charge1/2` only (no `nep4_charge3`): a charge_transfer model with `charge_mode 3` is rejected by the existing model-name check, not by the new code.

## Checks done without a GPU
- `make CUDA_ARCH="-arch=sm_86" -j4` in `src/`: all four binaries build; no new warnings (pre-existing: eam null-dereference, nvlink pthread/dl).
- `cuobjdump -sass` of the old kernels, base commit 93d203d4 vs this tree (`nep_charge.o`): 20 of 22 pre-existing kernels identical after normalising `c[0x0][..]`; `find_descriptor` and `find_descriptor_small_box` differ only in three indexed constant loads each (`LDC c[0x0][R+0x1d8/0x4c8/0x7b8] -> +0x1e8/0x4d8/0x7c8`, +0x10 = growth of `ParaMB`). Four new functions (`ct_*`).
- `git clang-format --diff`: clean.

## Gates (to be run by the parent; GPU free; one test file, ~1-3 min)
`pytest tests_pytest/test_replica_charge_transfer.py -q` (needs `src/gpumd` and `src/gpumd_replica`; env `GPUMD_REPLICA` overrides the path).
Method: `GPUMD_REPLICA_STATE_DUMP` (existing testing aid) writes position, force, potential, virial of every replica at the end of a 1-step REMD run (time_step 0.001 fs); gpumd evaluates the dumped positions statically (`time_step 0`, `dump_xyz precision double`) with the same nep.txt. REMD needs >= 2 replicas, so there is no 1-replica run; every replica of the 2- and 4-replica runs is compared with gpumd.
| gate | structure | criterion |
|---|---|---|
| 3b-3 (i)/(ii) | thick n45/L14 rc 3.5 (large box), thin n12/L7.5 rc 4 (small box), 2 replicas; thin also 4 replicas | |dE|/N <= 1e-4 eV/atom; RMS dF <= 1e-4 eV/A; RMS dW / RMS W <= 1e-4 (six independent components of the total virial); non-vacuous: RMS F > 0.05, RMS W > 0.5 |
| triclinic | same two structures, triclinic cell | same gates |
| rejection | mode 3 model, malformed/extra phi block, rc_phi > rc_radial | non-zero exit, message |
| old path | thin n12 without charge_transfer | loose guard only (|dE|/N <= 1e-3, RMS dF <= 5e-3; not a spec gate) |
Mode 3 cannot be compared (not supported by gpumd_replica). Gate 3b-4 (parent): `run_regression.py --suite full` (replica cases) and `python3 tests_regression/replica/test_replica.py`, to show the old models are unchanged.
Results: not yet run.

## Decisions of the human (2026-11-11), confirmed against the code
1. charge_mode 3 is not available in `gpumd_replica`: the replica accepts only the model names `nep4_charge1` and `nep4_charge2` (src/main_replica/force/nep_charge.cu, lines 74-80; there is no `nep4_charge3`), so a mode-3 `charge_transfer` model is rejected by the model-name check before the charge_transfer line is read. This is a documented limitation (mode 3 would be a separate port); the test only asserts the clean rejection.
2. The comparison is "each replica of a 2- and 4-replica REMD run vs gpumd evaluating the dumped positions statically" (REMD needs at least 2 replicas; PRD with one replica starts with a FIRE quench), with the approved gates (|dE|/N <= 1e-4 eV/atom, RMS dF <= 1e-4 eV/A, RMS dW / RMS W <= 1e-4).
3. A flexible ZBL potential together with `charge_transfer` is rejected in the replica (nep_charge.cu line 217), consistent with the trainer (`get_md_input` errors on the same combination).
Docs to update after the GPU gates pass: doc/gpumd-replica/input_parameters/potential.rst (currently "not supported": change to mode 2 supported, mode 3 not supported), and the VER-0034 README sentence that says the replica rejects charge_transfer.
