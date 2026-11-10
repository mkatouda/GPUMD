---
name: regression
description: Run the GPUMD full regression suite (baseline vs candidate binary) and classify failures against the known non-deterministic list. Use before the final completion report of any task that changes src/.
disable-model-invocation: true
---
Arguments: optional baseline binary path (default: build of `master`), optional `--suite`.

Stop and ask the human if any step is unclear (SPEC_CONFLICT rules in AGENTS.md apply).

1. Preconditions
   - `git status` clean of generated files; candidate = freshly built `src/gpumd` (+ `src/nep`)
     from `make CUDA_ARCH="-arch=sm_86" -j4`, warning-clean.
   - Baseline: a `master` build (e.g. worktree `.claude/worktrees/master-base`, `3b2c8db8`).
     Never use the candidate as its own baseline for acceptance.
   - Only one GPU-heavy job at a time (no other subagent running builds/tests).
2. Run (Python needs numpy; use the conda env), output piped short:
   ```
   /home/katouda/miniforge3/envs/py312-gpumd/bin/python tests_regression/run_regression.py \
     --repo-root "$PWD" --baseline <baseline>/src/gpumd --candidate src/gpumd \
     --baseline-nep <baseline>/src/nep --candidate-nep src/nep \
     --suite full --keep-all 2>&1 | tail -40
   ```
   Focused suites (`--suite pppm`, `--case <glob>`) are diagnostics only; acceptance is `full`.
3. Classify failures
   - Known non-deterministic cases (37) and the reference result (218 passed / 37 failed,
     relations 25/25) are listed in AGENTS.md, section "Known test results"; that list is canonical.
   - SUSPECT = any failing case outside the list, or a listed case failing by more than
     small numeric differences. Inspect only those (report under `tests_regression/reports/`).
   - Never edit tolerances or goldens to make a case pass.
4. Report (concise): command, binaries (commit hashes), GPU/CUDA version, passed/failed
   counts, list of suspect cases (or "none"), relations result.
5. Cleanup: delete `tests_regression/.work/` and `tests_regression/reports/`; confirm
   `git status` shows no generated files.
