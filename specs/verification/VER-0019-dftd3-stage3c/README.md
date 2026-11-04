# VER-0019: D3 stage 3c, small fixes after the independent numerical review (DR-001)

Environment: RTX A6000 (sm_86), CUDA from /usr/local/cuda, worktree d3-stage3b (HEAD bafbb6b9 + this change).

| item | change | evidence |
|---|---|---|
| F1 | `get_cn_list_capacity`: the estimate is clamped (in double, to 4096) before the int cast; byte check in size_t | list sizes of normal systems unchanged (clamp never active below 4096). Test `test_dftd3_non_periodic_degenerate_cell_matches_reference` (volume 1e-6 A^3): passes with the HEAD and the new binary alike (the undefined cast gave a harmless fallback on this machine), so it is a guard, not a before/after discriminator |
| F2 | test `test_dftd3_small_box_130_atoms_many_threads_per_atom` (130 atoms, 18.6 x 9.3 x 9.3 A, num_sub 1/64/128 via GPUMD_DFTD3_SMALL_BOX_NUM_SUB): bit-reproducible run to run, agrees with num_sub = 1 within the AC-1 caps | passes with HEAD and new binary. No independent reference exists for this structure (python `dftd3` is not installed); the stored references cover the fixtures only. compute-sanitizer memcheck and racecheck on this case with num_sub = 128: 0 errors / 0 hazards |
| F3 | test `test_dftd3_large_box_changing_box_list_matches_no_list` (water_large, cu_large; `npt_ber` + `deform`, compression and expansion, 100 steps, `velocity ... seed`): finite, first 5 steps of U with the list vs capacity 0 within 2e-6 x \|E_D3\| | passes. `npt_scr` was not usable: its stochastic cell rescaling makes two identical runs differ by ~1e-3 eV after a few steps, and without `velocity` even NEP-only runs differ by 0.2 eV at step 1 (uninitialised velocities, not investigated further) |
| F4 | `cn_list` grows with 25% headroom (capped at MAX_CN_LIST_BYTES) | water_large at -0.069 A/step: debug print (not kept) showed growth at the start (cap 10) and once more (cap 13, needed 454,896 > 437,400); without headroom it would have grown at cap 11, 12 and 13. Cu compression does not reach a second growth (the box cannot shrink below 2.5 x rc_radial = 20 A). memcheck on this run (60 steps) was clean, but 60 steps do not reach the second growth |

Timing (F4, nsys D3 kernel total, `scripts/run_nsys_total_s3c.py`, 3 alternations, HEAD binary vs new, median of medians; `results/nsys_total_s3c.json`):
water 5,184: 0.726 vs 0.730 ms (1.006x); water 41,472: 5.830 vs 5.845 ms (1.003x); gate <= 1.03x met.

Regression: `run_regression.py --suite full --case 'dftd3_*'` vs master-base: the runner's own comparison (atol 1e-17) reports 2 numeric diffs as in VER-0018; `scripts` of VER-0018 `compare_ac2.py` (rtol 1e-5, atol 1e-12): PASS (`results/compare_ac2.txt`).
Out of scope, not touched: the pre-existing float weight underflow at extreme CN.
