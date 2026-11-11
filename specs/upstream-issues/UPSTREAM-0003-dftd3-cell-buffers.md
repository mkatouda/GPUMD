# UPSTREAM-0003: dftd3 aborts (CUDA invalid argument at neighbor.cu:193) when the first call is small-box and the box later grows past 2.5 x cutoff

- Status: recorded 2026-11-11; NOT reported upstream (human decision 2026-11-11: keep the record, report/PR later). Fix in this fork: DR-006.
- Fork-only note (do not include in an upstream issue/PR). Reported by a collaborator using this fork (report dated 2026-11-09, measured with fork `8afa074` and upstream v5.9.1 on A100, CUDA 12.6, NEP89); original text: `/home/katouda/work/dielectric-response_pj/tmp/request_to_fork_20261110/REPORT_dftd3_cell_buffers_DRAFT_2026-11-09.md` (it also holds a `reproduction/` directory with model.xyz md5 efb866e4... and run.in).
- Affected upstream code: `brucefan1983/GPUMD` `src/force/dftd3.cu` (v5.9.1) and the fork `src/force/dftd3/dftd3.cu`; `src/force/neighbor.cu` is identical. Verified by code reading in this fork (2026-11-11); upstream master not re-checked.

## Problem
`compute_large_box` allocates the cell-list buffers (`cell_count_radial/angular`, `cell_count_sum_*`, `cell_contents_*`) only `if (cn.size() == 0)`, i.e. only when the first call takes the large-box path.
`compute_small_box` allocates `cn` but not those buffers. A run that starts with box thickness <= 2.5 x cutoff (30 A for `dftd3 pbe 12 6`) and later exceeds it (`npt_*`, `deform`) calls
`find_cell_list` with empty buffers: `CHECK(gpuMemset(cell_contents.data(), 0, sizeof(int) * N))` fails, "invalid argument", abort.

## Reported evidence (collaborator; not yet reproduced here)
- amorphous PTFE 368 atoms, `npt_mttk`, start 18 A: 8 of 8 runs crash (v5.9.1 and fork). Minimal: same system, NVE + `deform xx/yy/zz 0.001`, start 28.87 A: crash at box 30 A, after ~1100 steps.
- Same input without `dftd3` or with `dftd2`: runs. Large-first round trip 35.87 -> 29.87 -> 35.87 A: runs.

## Fix direction
Two independent guards: `cn` group `if (cn.size() == 0)`, cell-list group `if (cell_count_radial.size() == 0)`; `find_cell_list` already grows `cell_count*` when needed.
