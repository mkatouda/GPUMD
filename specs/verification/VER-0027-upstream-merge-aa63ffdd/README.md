# VER-0027 Merge of upstream/master aa63ffdd into develop (pre-merge develop aba6b28b)

Machine: RTX A6000 (sm_86), CUDA /usr/local/cuda, `make CUDA_ARCH="-arch=sm_86" -j4`. Range: 29 upstream commits beyond ddc6e2d2 (upstream moved sources into subfolders; Spin features, replica batching, ILP fix).
Baseline for regression = pre-merge develop aba6b28b (gpumd, nep built from `git archive`). Pure upstream aa63ffdd was built only to attribute the spin test failures.

## Conflicts and resolutions
- `src/force/nep/nep.cuh`, `src/force/nep_charge/nep_charge.cuh`: include block. Resolved as the union in upstream style: `force/dftd2.cuh`, `force/dftd3/dftd3.cuh`, `force/neighbor.cuh`, `force/potential.cuh`. No behaviour change.

## Path fixes (silent breakage)
- `git mv src/force/dftd3zero_table.cuh src/force/dftd3/` (next to dftd3.cu; its `#include "dftd3zero_table.cuh"` is same-directory and still valid).
- Fork-only dftd2.cu, dftd2.cuh, dftd2para.cuh, dftd2qe_table.cuh stay in `src/force/` (human decision). They are picked up by `$(wildcard force/*.cu)` (src/makefile, src/makefile.hip) and `src/force/*.cu` (CMakeLists.txt); the dftd3 dir by `force/*/*.cu`. All other includes of fork headers already use `force/dftd2.cuh`.
- Stale path mentions updated (`src/force/dftd3*.cu*` -> `src/force/dftd3/...`; `nep.cu`/`nep_charge.cu` -> subfolders in docs_ja/04): doc/gpumd/input_parameters/dftd3.rst, docs_ja/dftd/{01,02,04,05,A3}*.md, docs_ja/dftd/scripts/check_d3_derivation.py, tests_pytest/test_dftd3_reference.py, tests_pytest/fixtures/dftd3_zero/{README.md,generate_*.py}. Historical specs/ records and AGENTS.md were not changed (no wrong path in AGENTS.md). tests_pytest/test_dftd2_src_table.py and tools/Miscellaneous/dftd2 use dftd2*.cuh paths (unchanged).

## Results
| Stage | Result |
|---|---|
| Build gpumd nep gpumd_replica tb | OK; only compiler warning: upstream `force/eam/eam_alloy.cu:42` -Wnull-dereference (as in VER-0022); 0 from fork files; nvlink libpthread/libdl warnings benign. `make gnep` and HIP build not run. |
| CMake configure (sm_86) | configures and generates; not built |
| Focused: test_dftd3_zero, test_dftd3_reference, test_dftd2_qe, test_dftd2_qe_extended, test_dftd2_consistency, test_dftd2_src_table (not slow) | 248 passed, 21 deselected (1304 s) |
| tests_regression test_runner.py test_manifest.py | 36 passed |
| `pytest -m "not slow" -q` | 1145 passed, 4 failed, 480 skipped, 32 deselected (2842 s). The 4 failures are upstream spin `test_spin_force_modifiers.py::test_corrector_preserves_force_modifiers` (small/large box x nve_sib/nve_glsd), reproduced with a pure upstream aa63ffdd build (same 4 fail): not caused by the merge |
| `run_regression.py --suite full` (baseline pre-merge develop) | 220 passed, 37 failed, relations 25/25. The 37 failures are exactly the AGENTS.md known non-deterministic list (nvt_bdp, npt_scr_iso, heat_bdp, deform_npt_tri, pimd_scr, replicate_default_velocity, replicate_velocity_unseeded, graphene_hac, graphene_hnemd_shc, graphene_nemd_shc, silicon_liquid_transport, ilp_{nep,tersoff,tmd_sw}_compute, 23 train_*); no failure outside the list; dftd2/dftd3 cases pass |

Limitations: HIP and gnep not built; `slow` pytest not run; failing cases of the known list were not re-analysed individually (identity with the known list only); no timing measurement; the spin test failure on upstream is reported, not fixed.
