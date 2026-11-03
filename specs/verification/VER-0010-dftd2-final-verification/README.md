# VER-0010: DFT-D2 final verification (DES-0002 Phase 6 stage 3; DR-002 AC-4 at rc = 30, AC-6, AC-7)

- Date: 2026-11-03. Status: verification (self-consistency, regression vs master baseline, timing). NOT validation.
- Binary: worktree HEAD a012b76c (Phase 6 stage 2b), `make CUDA_ARCH="-arch=sm_86" -j4`, RTX A6000, CUDA 12.3. Model NEP89.
- Methods and acceptance criteria are those of VER-0006 (not changed or relaxed): G1 force error <= 2e-4 max|F_D2| at h = 0.01 A, G2 virial error <= 2e-4 max|W_D2| at h = 1e-3, N1 NVE not worse. Scripts: `scripts/fd_check.py`
  (VER-0006 copy + Cu/Au systems + FD_SYSTEMS selection), `scripts/nve.py` (unchanged), `scripts/timing.py` (VER-0008 version).

## A. AC-4 at rc = 30 A (`results/fd_s6_75_rc30.txt`, `results/nve.txt`)
FD: s6 = 75 (as VER-0006; the code is linear in s6), rc = 30 A. Systems: water27 cubic/triclinic/open/slab (81 atoms; forces on atoms 0, 1, 40) and new heavy-element
systems: Cu fcc 2x2x2 (32 atoms, rattled 0.1 A) periodic and open, Au fcc 2x2x2 rattled with a sheared (triclinic) cell (forces on atoms 0, 1, 20). Virial: all six strain components.
On-minus-off differences are bit-reproducible (repeat diff 0 for all).
| system | max abs F_D2 [eV/A] | force err rel (G1) | max abs W_D2 [eV] | virial err rel (G2) |
|---|---|---|---|---|
| water cubic | 3.694 | 2.6e-5 | 408.6 | 3.4e-5 |
| water triclinic | 3.365 | 1.7e-5 | 469.8 | 2.5e-5 |
| water open | 3.354 | 2.0e-5 | 223.0 | 1.0e-5 |
| water slab TTF | 3.642 | 1.3e-5 | 371.7 | 2.9e-5 |
| Cu32 periodic | 20.79 | 1.9e-5 | 1637.6 | 1.7e-5 |
| Au32 triclinic | 46.25 | 4.0e-5 | 6470.0 | 7.9e-6 |
| Cu32 open | 18.25 | 1.9e-5 | 321.4 | 4.1e-5 |
All nominal-step errors <= 4.1e-5 < 2e-4: G1, G2 met. Over the h scan (not gated) the maximum relative error is 3.9e-4 (truncation/noise at the extreme steps, same as VER-0006's 3.9e-4).
NVE (648 water, NEP89, 10 ps, dt 0.5 fs, same protocol as VER-0006, default rc = 30):
| run | E_tot std [eV] | max-min [eV] | drift [meV/ps] | drift [meV/atom/ps] |
|---|---|---|---|---|
| NEP89 only | 0.0843 | 0.296 | +27.3 | +0.0421 |
| NEP89 + dftd2 0.75 | 0.0594 | 0.261 | +19.0 | +0.0293 |
N1 met. As in VER-0006 this is one trajectory per run; the difference is within float-NEP trajectory scatter and is not evidence that D2 improves conservation.

## B. AC-7 timing (`results/timing.txt`, `results/timing.json`, `results/timing_table.txt`; slope method as VER-0006/0008)
Extra = (run with the term) - (run without), ms/step of the whole MD step; D3 = `dftd3 pbe 12 6`. Measurement noise ~0.5-1 ms on the small differences.
| N | NEP only | D3 extra | D2 rc12 extra: VER-0006 -> VER-0008 -> now | D2 rc30 extra: VER-0006 -> VER-0008 -> now | D2(12)/D3 | D2(30)/D3 |
|---|---|---|---|---|---|---|
| 81 | 1.40 | 1.38 | 4.27 -> 0.63 -> 0.34 | 30.6 -> 1.79 -> 0.90 | 0.25 | 0.66 |
| 648 | 1.61 | 1.86 | 3.04 -> 0.17 -> 0.19 | 54.3 -> 2.92 -> 1.81 | 0.10 | 0.98 |
| 5184 | 1.72 | 1.11 | 3.15 -> 1.56 -> 1.12 | 55.9 -> 15.96 -> 10.01 | 1.01 | 9.0 |
| 41472 | 6.75 | 7.40 | 12.03 -> 12.01 -> 7.06 | 211.4 -> 116.9 -> 68.6 | 0.95 | 9.3 |
AC-7 (D2 <= 0.5 x D3), informative: met at rc = 12 for N = 81 and 648 (0.25, 0.10); NOT met at rc = 12 for N = 5184, 41472 (1.0, 0.95); NOT met at rc = 30 for any N (0.66, 0.98, 9.0, 9.3).
Compared with the original (3-54x D3), D2 cost is now 0.1-1.0 x D3 at rc = 12 and 0.7-9.3 x D3 at the default rc = 30. The rc = 30 cost for N >= 5184 is dominated by the number of pairs within 30 A (about 14 x that of rc = 12 at the same density): not reducible by this scheme.

## C. AC-6 (`results/regression_full_master_baseline.log`, `results/pytest_not_slow.txt`)
`run_regression.py --suite full`, baseline = master 3b2c8db8 build (`gpumd`, `nep`), candidate = this build (gpumd, nep), once: 218 passed, 39 failed, relations 25/25 passed.
The 39 failures = the 37 known run-to-run cases (AGENTS.md "Known test results": nvt_bdp, npt_scr_iso, heat_bdp, deform_npt_tri, pimd_scr, replicate_*, ilp_*_compute, graphene_*, silicon_liquid_transport, all train_* cases) + 2 expected ones:
`dftd2_basic` and `dftd2_dftd3_exclusive` fail because the master baseline does not know the `dftd2` keyword (baseline exit 1). With baseline = candidate = this build, `--case 'dftd2_*'` gives 2 passed (byte-exact).
No case outside the known list failed, so no re-runs were needed. The exact list of 37 train_* names in AGENTS.md is described by category, so the comparison of the train_* names is by pattern, not by an enumerated list.
`pytest -m "not slow" -q` (tests_pytest): 841 passed, 480 skipped, 10 deselected, 0 failed, 1050 s (reference 738 passed before the D2 tests were added).

## Limitations
- One GPU (sm_86). FD at rc = 30 uses s6 = 75 (not 0.75); float noise floor sets the 1e-5 relative level. NVE: single trajectories. Slow-marked tests were not run here. Heavy elements covered by FD: Cu, Au only.
