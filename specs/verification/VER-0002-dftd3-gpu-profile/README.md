# VER-0002: Kernel-level profile of the GPUMD DFT-D3 implementation (Phase 0 of DES-0001)

- Status: Completed (profile only; no `src/` change)
- Date: 2026-11-03
- Requirement: `DR-001`; plan: `specs/design/DES-0001-dftd3-gpu-optimization-plan.md`
- Code profiled: `src/force/dftd3.cu` at `develop` commit `e48454ba` (`src/` identical to `master`)

## Setup
RTX A6000 (84 SMs, sm_86), CUDA 12.3, NEP89 + `dftd3 pbe <rc> <rc_cn>`, `nsys` 2023.3.3,
100 NVE steps (0.5 fs, 300 K), times are summed GPU kernel time per step (launch gaps and
host time not included). Scripts: `scripts/prof_all.py` (runs the cases under `nsys`),
`scripts/analyze.py` (authoritative classification; also writes `results/prof_summary.json`).
D3 kernels are the ones in the anonymous namespace of `dftd3.cu`
(`find_dftd3_*`, `add_dftd3_*`, and the small-box `find_neighbor_list_small_box`, whose
name coincides with NEP's kernel of the same name). The first version of the classification
missed this last kernel; `analyze.py` fixes it. The "graphite" case is 4,608 atoms.

## Results (ms per step; `results/prof_summary.json`)
| Case | N | GPU total | D3 total | D3 share | Dominant D3 kernel |
|---|---|---|---|---|---|
| water, 12/6, small-box path | 81 | 3.93 | 2.55 | 65% | force 1.33 + neighbor list 1.12 |
| water, 12/6, small-box path | 648 | 5.93 | 4.37 | 74% | neighbor list 2.83 (65% of D3) |
| water, 12/6, large-box path | 5,184 | 5.60 | 3.86 | 69% | force 3.37 (87% of D3) |
| water, 12/6, large-box path | 41,472 | 14.69 | 8.17 | 56% | force 6.54 (80% of D3) |
| water, 25/15, small-box path | 5,184 | 42.43 | 40.75 | 96% | neighbor list 22.49 + force 15.19 |
| graphite, 12/6, small-box path | 4,608 | 7.70 | 6.03 | 78% | neighbor list 3.25 + force 2.57 |
| fcc Cu, 12/6, large-box path | 4,000 | 4.96 | 3.65 | 74% | force 3.10 (85% of D3) |

The D3 totals agree with the VER-0001 "D3 extra" times (2.5 ms at N=81, 4.4 ms at 5,184 vs
3.9 here, 8.0 vs 8.2 at 41,472, 40.7 vs 40.75 at 25/15).

Path selection (`dftd3.cu:265-284`): the small-box path (explicit pair-vector lists of
`N x MN`, `MN = 10000`) is used when ANY periodic thickness is <= 2.5 x `rc_potential`, i.e.
for cells up to 30 A at rc=12 A, which includes typical NEP-training-size cells. The large-box
path (cell lists, per-atom register accumulation) is used otherwise.

## Findings
1. Large-box path: the force kernel `find_dftd3_force_large_box` is 80-87% of D3 time; the cell
   list build is negligible (about 0.015 ms/step) and the CN kernel is about 6-10%.
2. Under-filled GPU in the large-box force kernel (thread per atom, block 64): time per pair
   `t / (N x pairs within 12 A)` is 0.89 ns at N=5,184 (about 62 threads per SM) but 0.22 ns at
   N=41,472 (about 494 threads per SM): about 4x lower. Saturating the GPU at N=5,184 would take
   the force kernel from 3.4 ms to about 0.8 ms (estimate from the per-pair ratio).
3. Small-box path: building the explicit neighbor list is as expensive as, or more expensive
   than, the force kernel: 44% of D3 time at N=81, 65% at N=648, 55% at 25/15 A, 54% for graphite.
   This corrects the assumption in DES-0001 that cause D (neighbor list) is conditional: for
   cells <= 2.5 rc it is a first-order cost.
4. Element dependence (cause B, C6 interpolation): per pair, Cu costs 1.27 ns vs water 0.89 ns
   at similar N (about 1.4x); consistent with, but not proof of, the cost of the per-pair
   reference-CN double loop.
5. Static resources (`cuobjdump --dump-resource-usage`): `find_dftd3_force_large_box` 72
   registers, `find_dftd3_coordination_number_large_box` 42, `add_dftd3_force_small_box` 48,
   neighbor-list small-box 40; no stack, no local-memory spill. With 72 registers the
   theoretical limit is about 14 blocks of 64 threads per SM (about 58% occupancy), so
   register pressure is not the limiter; the thread count is.

## Gate of DES-0001 Phase 0
Criterion: D3 >= 20% of a NEP+D3 step in at least one typical case, and the force kernel is
the dominant D3 cost. Result: PASS (D3 is 56-96% of the GPU step in every case). The force
kernel is dominant in the large-box path; in the small-box path force and neighbor-list build
share the cost.

## Not measured (limitations)
- `ncu` (achieved occupancy, stall reasons, memory vs compute bound) could not be used:
  `ERR_NVGPUCTRPERM` (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`). Enabling
  needs admin rights (e.g. `NVreg_RestrictProfilingToAdminUsers=0`). By the owner's security
  policy (2026-11-03) GPU performance counters are NOT to be enabled with admin rights, so
  `ncu` counter metrics are out of scope for this project, not merely pending.
  Occupancy statements above are from register counts and thread counts, not from counters.
  Alternatives used or available: `nsys` kernel times, `cuobjdump --dump-resource-usage`,
  theoretical occupancy from register counts, CUDA-event timing, and micro-benchmarks.
- Host-side time and launch gaps are not included; GPU-time sums only.
- One GPU, one NEP model, H/C/O/Cu systems; the 25/15 A case is in the small-box path because
  the box (37 A) is below 2.5 x 25 A.

## Consequences proposed for DES-0001 (need human approval; DES-0001 not yet edited)
- Phase 4 item D (neighbor list for the small-box path) is promoted from conditional to a
  planned phase, because it is 44-65% of D3 time for cells <= 30 A at rc=12 A.
- Phases 2 (factorized C6 weights) and 3 (parallelism) apply to the large-box path first;
  both kernels (small-box and large-box) need the same treatment.
- A reproducible acceptance case set should include both paths (small box: N=81/648, large
  box: N=5,184/41,472, plus a Cu case).
