---
paths:
  - "src/**"
---
- Keep CUDA and HIP paths in sync; shared code goes through `src/utilities/gpu_macro.cuh`.
- Replica dynamics (`multi_replica`, inside `gpumd`) keeps its own copies of the NEP force code in `src/force/replica/` and `src/integrate/replica/` (e.g. `replica_nep_charge*.cu*`, `replica_nep_charge_transfer.cuh`); mirror force/model-I/O changes there where applicable. The `gpumd_replica` executable and `src/main_replica/` no longer exist.
- Both build systems glob sources by directory (`src/force/*.cu`, `src/force/*/*.cu` for gpumd;
  `src/main_nep/**` for nep): new files in existing dirs need no build edits, deeper nested dirs do.
- Keep per-model data and GPU workspaces in the potential object; do not reallocate in `compute()`.
- clang-format only on code changed in the current task, never the whole tree or
  `src/apply_clang_format.sh`. Revert reformatting of lines you did not edit; keep diffs small.
- User-facing keyword changes also need the `doc/` manual updated.
