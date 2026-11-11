# VER-0030: dftd3 cell-list buffers when the box crosses the small/large-box threshold (DR-006)

- Status: 2026-11-11. Verification type: reproduction + regression (baseline vs candidate). Environment: RTX A6000 (sm_86), CUDA 12.3.103, build `make CUDA_ARCH="-arch=sm_86"`, warning-free.
- Requirement: DR-006 (AC-1..AC-5). Record: UPSTREAM-0003 (not reported upstream, by the human's decision). Everything is uncommitted.

## Change
`src/force/dftd3/dftd3.cu`, `DFTD3::compute_large_box`: the guard `if (cn.size()==0)` is split in two; the six `cell_*` buffers are allocated under `if (cell_count_radial.size()==0)`, the cn/cn_weight/dc6_sum/dc8_sum buffers stay under `cn.size()==0`. 2 lines inserted. No other copy (`main_replica` none; HIP builds the same file).

## Tests (`tests_pytest/test_dftd3_cell_buffers.py`, 4 cases, `dftd3 pbe 4 2`, threshold 10.0 A, water27, `deform` 0.005 A/step x 200, `time_step 0`)
D3 energy = (run with D3) - (run without D3), compared at each segment end with a static single-path run of the dumped frame; cap 2e-6 |E_D3| (DR-001 AC-1, not loosened). Each segment must cross the threshold (asserted).
- AC-1 (stage 0): before the fix 2 of 4 fail (small->large, small->large->small) with the `force/neighbor.cu:193` "invalid argument" abort; the two large-first cases pass. After the fix 4/4 pass.
- AC-2: 4/4 pass (agreement ~5e-9 eV). Repeated 10 times: 10/10 passed (about 62 s each), no flakiness.
- Sensitivity (scratch script, not in the repo): a one-step (0.005 A) box shift changes E_D3 by 3.9e-3 eV (L=9.3, E_D3=-1.849 eV) and 3.2e-3 eV (L=10.3, E_D3=-1.282 eV); the caps are 3.7e-6 and 2.6e-6 eV, i.e. the comparison resolves a one-step difference by a factor ~1000.
- Collaborator's 368-atom case: 4000 steps run (box 28.87 -> 32.87 A).
- `test_dftd3_reference.py` + `test_dftd3_zero.py` (not slow): 101 passed.

## Regression (`run_regression.py --suite full`, baseline = master-base 3b2c8db8)
216 passed, 41 failed, relations 25/25. The 37 known cases fail as listed in AGENTS.md. Four cases outside the list:
- `dftd2_basic`, `dftd2_dftd3_exclusive`: the baseline exits with "'dftd2' is an invalid keyword" (master-base predates the fork's dftd2 and the upstream merge 8cce94a6). Not caused by the change.
- `dftd3_single_late`, `dftd3_kspace_one_each`: differences ~1e-10 in thermo/restart vs master-base. Candidate vs candidate: 3/3 PASS (deterministic). Candidate vs an unpatched build of HEAD (`git worktree`, removed): 5/5 dftd3/dftd2 cases PASS (bit-identical output).
Not done: the full suite with the unpatched HEAD as baseline (the master-base in AGENTS.md is stale after 8cce94a6; the known-failure list should be re-measured).

## Review
Independent numerical-gpu-reviewer (static): no BLOCKER/MAJOR; two MINOR (threshold crossing assertion; flakiness/sensitivity) both addressed above.

## Limitations
- HIP build NOT verified (no hipcc on this machine); the file is shared by both builds and the change uses no API-specific code.
- AC-3 is shown by the 5 dftd cases and the test files above, not by a dedicated no-transition comparison of forces/virial.

## Addendum (2026-11-11): AC-3 against a control build, from the first regression agent's report
- Control binary: scratch full rebuild with `dftd3.cu` of `b90c2920^` (before the fix), everything else as the candidate.
- water27 (9.3 A, `dftd3 pbe 12 6`, small-box path) and water_large (27.9 A, large-box path), NVE 100 steps, `velocity 300 seed 7`, dump_thermo 10, forces in double; models: all-zero NEP and NEP89. Candidate vs control: `thermo.out` and `f.xyz` byte-identical in all 4 configurations.
- The `dftd3_single_late` / `dftd3_kspace_one_each` regression cases: candidate vs control byte-identical in `restart.xyz` and `thermo.out` (first output); candidate and master-base 3b2c8db8 are each reproducible run to run (3/3) but differ from each other by ~1e-10 (other develop changes since 3b2c8db8, e.g. the source restructuring; the cause was not pinned down further).
- Candidate vs master-base direct comparison is therefore not byte-identical and is not evidence about the fix (all-zero NEP step-0 PE: water27 -2.7082682 vs -2.7082678 eV).
- Still open: `--suite full` against an unpatched HEAD baseline (to be started by the human), HIP build.
