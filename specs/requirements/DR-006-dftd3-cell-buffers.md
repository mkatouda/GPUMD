# DR-006: dftd3 must not abort when a run starts in the small-box path and the box later exceeds 2.5 x cutoff

- Status: Approved by the human on 2026-11-11 (scope: fix + test + upstream-issue record; the upstream report/PR is NOT sent now).
- Origin: bug report from a collaborator, `/home/katouda/work/dielectric-response_pj/tmp/request_to_fork_20261110/REPORT_dftd3_cell_buffers_DRAFT_2026-11-09.md` (2026-11-09). Record: `specs/upstream-issues/UPSTREAM-0003-dftd3-cell-buffers.md`.
- Related: DR-001 (dftd3 performance), DR-003 (zero damping). Upstream behavior is a contract: only the crash is removed; no result changes.

## Problem (confirmed by code reading, 2026-11-11; reproduced, VER-0030)
`DFTD3::compute_large_box` (`src/force/dftd3/dftd3.cu` ~1702) allocates `cell_count_*`, `cell_count_sum_*`, `cell_contents_*` only inside `if (cn.size() == 0)`.
`compute_small_box` allocates `cn` (guard `NN_radial.size() == 0`) but not the cell-list buffers. A run whose first call is small-box and whose box later grows past
the threshold (thickness > 2.5 x rc_radial; NPT, `deform`) enters the large-box path with empty cell-list buffers and aborts at
`force/neighbor.cu:193` (`gpuMemset(cell_contents.data(), 0, sizeof(int)*N)`): "CUDA Error ... invalid argument".

## Requirement
R1. The large-box path allocates its cell-list buffers independently of `cn` (guard on the cell-list buffers themselves).
R2. Any sequence of box paths (small -> large, large -> small, small -> large -> small, ...) completes; results unchanged otherwise.

## Acceptance criteria
- AC-1. A test that starts small-box and crosses the threshold fails on the pre-change binary with the report's error and passes after the fix.
- AC-2. large -> small and round trips complete; energies before and after the transition agree with single-path runs within the DR-001 AC-1 energy cap (2e-6 relative), no loosening.
- AC-3. Runs without a path transition are bit-identical to the pre-change binary; `test_dftd3_reference.py`, `test_dftd3_zero.py` unchanged.
- AC-4. Build warning-clean (sm_86); `run_regression.py --suite full`: only the known 37 cases fail (AGENTS.md); independent numerical-gpu review without open BLOCKER/MAJOR.
- AC-5. `specs/upstream-issues/UPSTREAM-0003-*` and `specs/traceability.yaml` updated.

## Out of scope
Banner version string, `std::mt19937` clock seeding of `nvt_bdp`/`npt_scr`, `thermo.out` comment lines (collaborator's observations; no action requested).
