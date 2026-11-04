# VER-0022 Merge of upstream/master ddc6e2d2 into develop (merge commit 6791f834)

Machine: RTX A6000 (sm_86), CUDA /usr/local/cuda, `make CUDA_ARCH="-arch=sm_86" -j4`. No source fix was needed (no extra commit).
Baseline = detached worktree at upstream/master ddc6e2d2 (gpumd, nep). Pre-merge reference = develop 8fe8ab99 (S6 only).

| Stage | Result |
|---|---|
| S1 build (gpumd nep gpumd_replica tb) | builds; 1 warning, upstream code only: `force/eam_alloy.cu:42` -Wnull-dereference; none in dftd2/dftd3/gpu_macro; nvlink libpthread/libdl warnings are benign (results/build_warnings.txt) |
| S3 `test_dftd3_reference.py test_dftd2_*.py` | 128 passed (1853 s), tolerances unchanged |
| S4 `run_regression.py --suite full` (baseline = master-new) | 216 passed, 41 failed, relations 25/25. 37 failures = exactly the known non-deterministic list of AGENTS.md; 4 more = fork-only `dftd3_single_late`, `dftd3_kspace_one_each` (baseline D3 differs numerically), `dftd2_basic`, `dftd2_dftd3_exclusive` (baseline lacks dftd2). No unexplained failure. results/fails.txt, regr.log |
| S4 dftd3 AC-2 (compare_ac2.py, rtol 1e-5 atol 1e-12) | PASS: single_late thermo 4.1e-7 abs / 2.6e-7 rel, restart 3.7e-11 / 1.6e-7; kspace_one_each thermo 3.7e-7 / 1.5e-6, restart 1.7e-12 / 2.6e-8 (results/ac2_dftd3.txt) |
| S5 `pytest -m "not slow" -q` | 868 passed, 480 skipped, 10 deselected, 0 failed (2072 s); earlier 859 passed / 480 skipped |
| S6 D3/D2 kernel time per step (nsys, median of 3, premerge vs merged) | w648 D3 0.504/0.501 ms (0.995); w648 D2 0.577/0.575 (0.997); w5184 D3 0.728/0.738 (1.015); w5184 D2 1.899/1.886 (0.993). Gate 3% met |

S6 note: the first attempt (scripts/timing.py, wall-clock slope with-minus-without D3, 5 reps) was noisy (ratios 0.99-1.12; spread of individual runs up to
+-20% for w648) and is NOT used for the gate; nsys kernel sums are reproducible to <1%. Both are kept in results/ (timing.*, nsys.*).
Not verified: HIP build; `slow` pytest; 41,472-atom point; wall-clock total time.
