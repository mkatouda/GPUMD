# VER-0012: DFT-D3 small-box path, Phase 7 Stage 1 diagnostic (DES-0001 Phase 7)

- Date: 2026-11-04. Status: verification (timing diagnostic; NOT validation). Requirement / design: `DR-001` AC-4, `DES-0001` Phase 7 Stage 1.
- Code under test: worktree `wt-d3-phase7`, HEAD `e0a6918d`, `src/force/dftd3.cu` unchanged (no file under `src/` was modified;
  timing-only variants are built from a copy of `src/` in a scratch directory).
- Environment: RTX A6000 (sm_86, 84 SMs), CUDA 12.3 (`/usr/local/cuda`), NEP89 (`nep89_20250409`), `dftd3 pbe 12 6`
  (rc 12 / CN cutoff 6), `velocity 300 seed 1`, NVE 0.5 fs, 100 steps, `make CUDA_ARCH="-arch=sm_86" -j4`. One GPU job at a time.
- Method: `nsys profile --export=sqlite`; per-kernel duration from `CUPTI_ACTIVITY_KIND_KERNEL`; the first 5 calls are dropped;
  median over calls; then median of 3 runs (variant stages: base and variant binary alternated, 3 times each, base first).
  "D3 total" = neighbor list + CN + force + extra kernels (small box); large box adds the `find_cell_*` and cub scan kernels
  (this reproduces the VER-0004 definition: 6.86 ms here vs 6.96 ms there for water 41,472).
  D3 kernels are separated from the NEP kernel of the same name (`find_neighbor_list_small_box`) by `DFTD3::ExpandedBox` in the signature.
- ncu counters are not available (not allowed on this machine); the compiler report and nsys replace them.

## Result summary (gate: Stage 2 may start if at least one of (a) (d) (e) (f) holds)
| item | gate | measured | verdict |
|---|---|---|---|
| (a) small-box force time vs large-N throughput, N <= 648 | >= 2x | N=81: 101x, N=162: 50x, N=324: 25x, N=648: 12.6x (N=972: 8.8x, 2187: 4.2x, 8748: 1.4x) | MET |
| (d) C6 interpolation share of the force kernel | >= 25 % | small box: 47 % (N=81), 38 % (648), 34 % (2187), 12 % (8748); large box: 59 % (cluster 972), 24 % (water 41,472) | MET (N <= 2187) |
| (e) float variant | >= 1.2x | float atomics 0.98-1.08x, float coordinate differences in the list build 1.01-1.10x of the D3 total (N <= 648); large-box force kernel 1.02-1.06x | not met |
| (f) launch/latency floor at N=81 | >= 40 % of D3 total | empty four-kernel sequence 6.4 us (4.0 us kernels + 2.4 us gaps) = 0.42 % of 1.517 ms | not met |

Verdict: the gate for Stage 2 is passed by (a) and (d). Stage 2 was NOT started (this record ends at Stage 1).
Full tables: `results/summary.md` (built by `scripts/summarize.py` from `results/*.json`).

## (a) N dependence of the small-box force kernel
Water (water27 repeats, `scripts/make_systems.py`), fixed rc 12/6, density constant, small-box path for every N (one periodic
thickness 27.9 A <= 30 A; no thickness above 120 A). Throughput reference = the largest N (34,992: 5.865 ms = 0.1676 us/atom);
it is itself probably not saturated (547 blocks of 64 threads on 84 SMs), so the ratios are conservative.

| N | force [ms] | expected from throughput [ms] | measured / expected | NL [ms] | CN [ms] | extra [ms] | D3 total [ms] |
|---|---|---|---|---|---|---|---|
| 81 | 1.378 | 0.014 | 101 | 0.039 | 0.033 | 0.067 | 1.517 |
| 648 | 1.364 | 0.109 | 12.6 | 0.375 | 0.052 | 0.098 | 1.887 |
| 972 | 1.438 | 0.163 | 8.8 | 0.419 | 0.055 | 0.099 | 2.009 |
| 2,187 | 1.557 | 0.367 | 4.2 | 0.534 | 0.056 | 0.101 | 2.247 |
| 8,748 | 2.111 | 1.466 | 1.4 | 7.747 | 0.058 | 0.189 | 10.106 |
| 34,992 | 5.865 | 5.865 | 1.0 | 123.0 | 0.087 | 0.635 | 129.6 |

- The force kernel is 1.34-1.44 ms for N = 81...972 (N independent): one thread per atom, 64-thread blocks, so N=81 launches 2 blocks and
  N=648 launches 11 blocks on 84 SMs. About 740 pairs per atom are processed serially: 1.37 ms / 740 = 1.9 us per pair iteration
  (a dependent chain of global loads: list entry, `g_type[n2]`, `g_cn[n2]`, `c6_ref`; plus 15 double atomics) - latency bound, not throughput bound.
- The neighbor-list kernel is O(N^2) (one block per atom scans all N x images candidates): 0.375 ms at N=648 (20 % of the total) and larger than
  the force kernel from N ~ 2,900 on.
- The "cluster 972" system of the task list (non-periodic) does NOT take the small-box path: `get_expanded_box` returns false without a
  periodic direction, so it uses the large-box kernels (force 1.10 ms, D3 total 1.71 ms here).

## (b) Registers, spills, occupancy (`results/ptxas_sm86.txt`, `scripts/ptxas_report.sh`)
| kernel (small box) | registers | smem | spills |
|---|---|---|---|
| `add_dftd3_force_small_box` | 48 | 0 | 0 |
| `find_neighbor_list_small_box` | 36 | 1,024 B | 0 |
| `add_dftd3_force_extra_small_box` | 40 | 0 | 0 |
| `find_dftd3_coordination_number_small_box` | 29 | 0 | 0 |

Force kernel theoretical occupancy (sm_86: 65,536 registers, 1,536 threads, 16 blocks per SM; 64-thread blocks): register limit 42 warps,
block limit 16 blocks = 32 warps, so 32/48 = 67 %. Achieved occupancy is determined by the grid, not by registers: N=81 gives 3 warps
on 84 SMs, N=648 gives 21 warps (0.5 % of the 84 x 48 resident-warp capacity). No spills, so registers are not a problem.

## (c) Candidates, accepted pairs, list length vs MN = 10000 (variant `count`, exact counts from the real kernel)
`results/count.txt`. cand = candidates visited per atom by the list kernel (N x number of images); accepted = list length (r < rc).
| system | rc | cells | cand/atom | accepted radial mean / max | accepted angular mean / max | max / MN |
|---|---|---|---|---|---|---|
| water 81 | 12 | 3x3x3 | 2,187 | 725 / 740 | 85 / 92 | 7.4 % |
| water 648 | 12 | 2x2x2 | 5,184 | 725 / 740 | 85 / 92 | 7.4 % |
| water 8,748 | 12 | 1x1x1 | 8,748 | 725 / 740 | 85 / 92 | 7.4 % |
| water 648 | 20 | 3x3x3 | 17,496 | 3,389 / 3,412 | 85 / 92 | 34 % |
| fcc Cu 864 | 12 / 20 | 2x2x2 | 6,912 | 626 / 626 ; 2,898 / 2,898 | 78 | 6.3 % / 29 % |
| diamond C 1,728 | 12 / 20 | 2x2x2 | 13,824 | 1,290 / 1,290 ; 5,850 / 5,850 | 158 | 13 % / 58.5 % |

- Accepted / visited = 33 % (N=81), 14 % (N=648), 8 % (N=8,748 is 725/8748): the list kernel rejects 67-92 % of what it visits.
- The list length scales as (4/3) pi rc^3 rho. Extrapolated limit (continuum estimate from the measured counts): MN = 10,000 is reached at
  rc ~ 28.8 A (water), 30.3 A (fcc Cu), 23.8 A (diamond C). Realistic use (rc 12-20) leaves a margin of 1.7x (dense C at rc 20) to 13x.
- As VER-0004 states, `MN` is not checked: `find_neighbor_list_small_box` writes slot `pos * N + n1` with no bound (lines 432-443 of `dftd3.cu`).
  A dense system with a large cutoff (diamond-like, rc >= 24 A) would write out of bounds silently. This is pre-existing, not changed here.
- Memory of the explicit list: N x MN x 32 B (2 int lists + 6 float arrays) = 320 KB per atom, e.g. 11 GB at N = 34,992.

## (d) C6 interpolation share (variants `noc6`, `noc6L`: `if (num_cn_1 == 1 && num_cn_2 == 1)` replaced by `if (true)`, so c6 = first reference, dc6 = 0)
Force-kernel time base -> variant (ms) and D3 total speedup (3 alternated runs each; `results/variants.json`, `variants_large.json`):
| system | force base -> noc6 | share removed | D3 total x |
|---|---|---|---|
| water 81 (small) | 1.377 -> 0.733 | 47 % | 1.72 |
| water 648 (small) | 1.440 -> 0.888 | 38 % | 1.40 |
| water 2,187 (small) | 1.599 -> 1.060 | 34 % | 1.28 |
| water 8,748 (small) | 2.124 -> 1.870 | 12 % | 1.03 |
| cluster 972 (large-box kernel) | 1.104 -> 0.454 | 59 % | 1.55 |
| water 41,472 (large-box kernel) | 5.386 -> 4.121 | 23.5 % | 1.23 |
The share is an upper bound of what an exp-reduction could give (the variant also removes the `g_cn[n2]` gather and 6 loads of `c6_ref`).
The interpolation weight is separable, L_ij = exp(-4 diff_i^2) exp(-4 diff_j^2); a per-atom table of exp(-4 diff_i^2) (<= 5 values) would
remove most of the exp calls (not in the Stage 2 list of DES-0001; noted as an observation, it changes results at round-off level).

## (e) FP64 weight (timing only; accuracy not checked)
In the small-box force kernel the list holds float differences already, so the FP64 work there is the 15 double `atomicAdd` per pair.
| variant | water 81 | 648 | 2,187 | 8,748 |
|---|---|---|---|---|
| `f32atomic` (float atomics): force x / total x | 0.98 / 0.98 | 1.08 / 1.06 | 1.07 / 1.05 | 1.15 / 1.03 |
| `n1only` (n2 and virial atomics removed): force x | 1.22 | 1.15 | 1.21 | 1.72 |
| `regacc` (no pair-loop atomics, register sums): force x | 1.14 | 1.03 | 1.16 | 1.57 |
| `nlf32` (list build: float coordinate differences): NL x / total x | 1.21 / 1.02 | 1.45 / 1.10 | 1.28 / 1.01 | 1.42 / 1.29 |
| large box `xf32L` (cluster 972, water 41,472): force x | 1.02, 1.06 (total 1.01, 1.04) | | | |
Atomics matter only when the GPU is filled (N = 8,748: removing them gives 1.6-1.7x); at N <= 648 even removing all atomics gives <= 1.14x,
which confirms the latency (serial pair loop) explanation. The float list build helps only the NL kernel (1.2-1.45x), 1.10x of the total at N=648.

## (f) Launch / latency floor (variant `empty`: all four small-box kernels return at once)
nsys timeline of one D3 step: sum of the 4 empty kernels 3.97 us (N=81) / 4.35 us (648), 3 gaps 0.7-0.8 us each (2.4 us), span 6.4 / 6.7 us =
0.42 % / 0.35 % of the base D3 total (1.517 / 1.887 ms). In the real run the gaps are also 0.7 us (2.2 us per step, 0.15 %), i.e. the kernels
run back to back. CPU-side launch cost is hidden by the 1.4 ms force kernel. The floor would only matter after the force kernel is shortened
by about 20x (AC-4 target 1.0 ms at N=81 leaves a very large margin over 6.4 us).

## Stage 2 outlook (not started; each candidate needs its own gate)
1. (i) several threads per atom with fixed-order reduction on the small-box force kernel: most promising. The force kernel is 12.6x (N=648) to
   100x (N=81) slower than its own large-N throughput and independent of N, i.e. a fill/latency problem. If it reached the throughput time the D3 total
   would be about 0.5-0.7 ms at N=648 (now 1.89, dominated afterwards by NL 0.375 ms) and about 0.15 ms at N=81 (now 1.52): this is an estimate from the
   measurements, not a measured result. Atomics (15 per pair) must be reconsidered when threads per atom increase (they cost 1.6x at N=8,748).
2. (ii) cost-based cell/list parameters / cell-based candidates for the list build: second priority. NL is 20 % of the total at N=648 and
   dominates from N ~ 2,900 (O(N^2); visits 5,184 candidates for 725 accepted at N=648). Matters mostly after (i).
3. (iii) float candidate loop: low value alone (list kernel only 1.2-1.45x, total <= 1.10x at N <= 648); needs the AC-1 accuracy study.
4. (iv) kernel fusion: not justified (floor 0.4 %).
5. Possible addition (needs a human decision, outside the listed items): separable per-atom exp table for the C6 interpolation (up to 47 % of the
   force kernel at N=81, 24 % at water 41,472, i.e. also the only lever seen for AC-4 at 41,472: total 6.88 ms vs target 6.0; `noc6L` gives 5.59 ms).
   The large-box path was not otherwise examined (Phase 3 `num_sub` already applies); the 41,472 target is unmet by 15 % and C6 share is
   its only lever found here.

## Not verified / limitations
- Single GPU (sm_86), CUDA 12.3; HIP not exercised. Variants change results and are timing only.
- The first `w81 noc6` row of `variants.json` was measured while a short unrelated nsys job (the `empty` stage) ran on the same GPU for part of the
  time; its base runs are normal (1.377 ms) and the variant runs (0.743/0.698/0.733) agree with the N=648 trend, but it was not repeated.
- No hardware counters (ncu unavailable): the latency-bound explanation rests on N independence, the atomic/FP64/interpolation removals and the grid size,
  not on stall reasons. The multi-thread-per-atom speedup is an estimate, not a measurement.
- `count` limits: only the listed systems; the MN extrapolation is a continuum estimate. Throughput reference (N=34,992) may not be saturated.
- Large-box path: only variants (d)(e) of the force kernel were timed; no scan over N or over cell parameters.

## Reproduction (scratch dir S outside the repository; PY = a Python with ase)
```
R=<repo root>; S=<scratch>
$PY scripts/make_systems.py $R $S/xyz
scripts/build_variants.sh $R $S base noc6 f32atomic nlf32 n1only regacc empty count noc6L xf32L   # copies src/ to $S/src, repo untouched
scripts/ptxas_report.sh $R $S > results/ptxas_sm86.txt
$PY scripts/run_study.py $S scan scanbig variants empty large count                               # writes results/*.json, count.txt
$PY scripts/summarize.py                                                                          # results/summary.md
```
`scripts/patch_variants.py` documents every variant (each is a few lines of change to `dftd3.cu`).
