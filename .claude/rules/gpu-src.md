---
paths:
  - "src/**"
---
- Keep CUDA and HIP paths in sync; shared code goes through `src/utilities/gpu_macro.cuh`.
- Mirror changes into `src/main_replica/` copies where applicable (see its makefile).
- Both build systems glob sources by directory (`src/force/*.cu`, `src/force/*/*.cu` for gpumd;
  `src/main_nep/**` for nep): new files in existing dirs need no build edits, deeper nested dirs do.
- Keep per-model data and GPU workspaces in the potential object; do not reallocate in `compute()`.
- clang-format only on code changed in the current task, never the whole tree or
  `src/apply_clang_format.sh`. Revert reformatting of lines you did not edit; keep diffs small.
- User-facing keyword changes also need the `doc/` manual updated.
