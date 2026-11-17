# VER-0037: merge of upstream/master 5c67f365 into the fork develop (5de76eb5)

- Status: verified 2026-11-18 on RTX A6000 (sm_86), CUDA 12.3, `make CUDA_ARCH="-arch=sm_86"`. Type: regression + build + focused tests. Requirements DR-001..DR-006 (fork features kept), DR-005 (replica part revised, see below).
- Merge: upstream 3b2c8db8 -> 5c67f365 (225 commits, 2026-11-01..17). Integration branch `merge-upstream-5c67f365`; merge commit f9ada8de, then the fork porting commits listed below.
- Baselines (separate worktrees, md5 of binaries kept in the session scratchpad): DEVELOP = 5de76eb5 (fork before merge), MASTER = upstream 5c67f365, MERGE = integration branch tip.

## What was resolved / ported
- 16 conflicts. dftd3: the fork version is kept wholesale (DR-001/DR-006). Upstream's own dftd3 changes are NOT adopted: box-switch allocation fix c79ed9a4 (same defect as DR-006, redundant), per-atom weight cache d5bf3f7f (not adopted), C6 normalization 5c104c42 (not adopted now; human decision 2026-11-18, to be reconsidered as a separate commit with its own VER).
- charge_transfer (DR-005) ported to upstream's `NEP_Model_File` reader: `charge_transfer rc_phi` line after the cutoff line and `@ phi_parameters N` block (N = num_pairs*(basis_size_radial+1)); qnep2/qnep3 only; flexible ZBL + charge_transfer refused.
- Replica (now inside `gpumd`, `src/force/replica/`): ct force path ported (`replica_nep_charge_transfer.cuh`, h copy via `gpuMemcpyAsync`; upstream's `GPU_Vector` has no `copy_from_device_async(ptr,N,stream)`). The replica computes no BEC.
- Upstream removed legacy nep.txt formats 1-6: legacy fork charge_transfer files (`nep4[_zbl]_charge2`, `nep4_charge3` with `charge_transfer`) are converted by the extended `potentials/nep/nep_model_converter.py` (human-approved policy; `energy_reference` becomes 0 in the conversion).
- Other build fixes: duplicate `gpuGetDevice` macro removed (9625b892). Docs, generators in specs/verification and tests updated to the new format (commits a43d0a1f ... 586436ff).

## Gates
| gate | content | result |
|---|---|---|
| G1 | build `src/` (gpumd, nep, tb), warning-clean | passes; only warning: upstream `eam_alloy.cu:42` (null dereference, upstream code) |
| G2 | focused pytest (dftd2/dftd3 suites, qnep charge_transfer parse/bec/forces/field, nep trainer ct, replica ct, rc_phi parse, converter 23 CPU tests) | pass after test-infra fixes (2eda50cf, converter commits). Not fork-related: upstream `test_replica::test_restart_velocity_layouts` fails on upstream master as well; `test_qnep_large_box_concurrent_resume` (pppm) is flaky |
| G3 | pre/post-merge identity DEVELOP vs MERGE on 87 generated cases (D3/D2 variants, qNEP, BEC, EF), each run 2 to 40 times per binary; criterion fixed in advance: bit-identical, or max difference <= 2x the same-binary repeat difference of DEVELOP (repeat 0 => bit identity) | 66 bit-identical, 19 within noise, 2 outside the criterion at first (below); criterion unchanged |
| B2 extra | the two strict-criterion cases re-measured, 100 runs per binary | `C_big3000_m2_pppm_ef`: passes (all groups within noise; 32 runs had failed energy/virial/force). `C_thick45_m3_ewald` (BEC): still outside the criterion: max diff MERGE-vs-DEVELOP 3.0e-8, DEVELOP-vs-DEVELOP 7.5e-9, MERGE-vs-MERGE 3.0e-8 (BEC scale 0.63, float 1 ulp). Distinct BEC arrays: DEVELOP 2 (52/48), MERGE 3 (93/6/1); one array common to both; elements varying: DEVELOP 5 of 405, MERGE 62 of 405; mean difference 3.6e-9. |
| B3 | replica gates (displacement identity; self-calibrated against ordinary `gpumd`) | all pass: floor 1.8-2.0e-8, RMS dF <= 1.1e-8 eV/A (gate 1e-4), dE/N = 0 (gate 1e-4). Virial gate: not verified (human decision 2026-11-11 c). charge_mode 3 + ct verified for energy/force at small cases only |
| B4 | `run_regression.py --suite full` (3 runs, GPU memory max 4239 MiB) | see below |

### B2 decision (human, 2026-11-18)
The `C_thick45_m3_ewald` BEC difference is ACCEPTED by the human with this reason: the kernel sources are identical (`nep_charge_transfer.cuh` byte-identical to DEVELOP; `nep_charge_small_box.cuh` differs by one null check of `energy_reference`; `nep_charge.cu` differs in model parsing only), the model parameters are identical as a multiset (max difference 0.0 between the legacy and the converted file), and the difference is float-atomicAdd ordering noise of 1 ulp. The criterion itself was not changed.

### B4 (`--suite full`, baseline -> candidate)
| run | baseline -> candidate | passed / failed | relations | wall |
|---|---|---|---|---|
| R1 | MASTER -> MERGE | 216 / 41 | 25/25 | 464 s |
| R2 | MERGE -> MERGE (nondeterminism set N) | 220 / 37 | 25/25 | 436 s |
| R3 | DEVELOP -> MERGE | 173 / 84 | 16 pass, 9 skip | 437 s |
- R2: 37 failures, same count as the known 37 of AGENTS.md (14 non-train cases listed there all fail again; 23 `train_*` cases cannot be matched by id). These are run-to-run differences.
- R1: 41 = the 37 of N + 4: `dftd2_basic`, `dftd2_dftd3_exclusive` (fork-only feature, MASTER cannot run it); `dftd3_single_late`, `dftd3_kspace_one_each` (reproduced 3x). Cause of the last two: upstream's C6 rewrite (5c104c42) changes float rounding. Checked directly on `dftd3_single_late`: MERGE and DEVELOP agree (PE -132.05107804, same stress), MASTER differs (PE -132.05107783, relative 1.6e-9).
- R3: 84 = N (37) + 47, identical in 3 reruns: 32 stdout only (NEP info lines added/removed); 11 ensemble_mttk family (`nvt_mttk`, `npt_mttk_*`, `mttk_couple_*`, `adaptive_npt_mttk`, `adaptive_nphug`, `nphug_short`) and 2 `ti_as`/`ti_rs` whose final/restart/thermo differ: attributed to upstream 111cf8b2 (initializes MTTK chain forces in `ensemble_mttk.cu`), since MASTER -> MERGE passes in R1; 2 `dump_*_invalid_model` where DEVELOP lacks the diagnostic text of upstream. The numerical values of the `ti_*` cases were not analysed beyond this attribution.
- Not verified: `-DDEBUG` build of `nep`, `tests_tb`, HIP build (makefile.hip default arch changed upstream: HIP_ARCH gfx1100, `-std=c++17`).

## G5 independent reviews
- numerical-gpu-reviewer (static read of the fork-specific charge_transfer port; nothing built or run): no BLOCKER, no MAJOR. Replica ct kernels are the gpumd kernels minus the BEC kernels (same math); the h copy is `gpuMemcpyAsync` on the replica stream, ordered with the kernels before and after it; phi count and rc_phi handling agree in reader, gpumd and replica; no allocation in compute(); no change on the non-ct path in the fork code. Not verified by the reviewer: launch configs, multi-GPU device selection by the caller, full bit-identity of the upstream-only non-ct path. MINOR: `ct_V` allocated in the replica (kept), plain `gpuMemcpyAsync` instead of a helper (style).
- api-compat-reviewer (static read; no build or run): no BLOCKER, no MAJOR. Verified: new nep.txt reader checks (qnep2/qnep3 only, 0 < rc_phi <= radial cutoff, phi block size), `validate_md` refusals, converter rules and nep_model_format.md section 7.1, dftd2/dftd3 in all build lists (makefile, makefile.hip, CMake) and in `run.cu`, 233 unique case ids in `tests_regression/gpumd/manifest.json` with existing inputs. MINOR: (1) the doc sentence "batched integration is not used for qNEP" is backed by `src/integrate/replica/replica_common.cu:785` (checked by the parent); (2)(3) replica rejects `add_efield`, `compute_dpdt`, `dump_xyz ... bec` and `dftd2` through the generic unknown-keyword/syntax error, not a dedicated message; (4) stale ignored `.o` files in the worktree (`make clean` before a final build); (5) `docs_ja/` (fork-only) still describes `gpumd_replica`/`main_replica`: to be updated in stage C4; comments in `src/force/replica/` still say "gpumd_replica" (harmless). Not verified: the exact message of gpumd on an old-format file; whether the replica test covers qnep3 + charge_transfer; no fork ADR exists for the legacy-format drop (upstream decision; open item).

## Known limitations / residual risks
- Replica: no BEC, no virial gate, qNEP batching off, flexible ZBL + charge_transfer refused.
- Legacy charge_transfer `nep.txt` must be converted; `energy_reference` is 0 after conversion (models that used a reference need a check).
- Upstream pre-existing: `test_restart_velocity_layouts`, `eam_alloy.cu:42`.
- Minor findings carried: sum of q in float; unchecked MN_radial overflow in upstream code; unused `ct_V` allocation in the replica.
