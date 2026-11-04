# VER-0016: float pre-filter of the large-box D3 CN kernel (DES-0001 Phase 7 item (vii))

Status: Stage S1 gate NOT MET; no `src/` change kept (prototype patch only in the session scratchpad).
Platform: RTX A6000 (sm_86), CUDA 12.3. Timings: nsys kernel median over 100 steps (5 dropped), median of 3 alternating runs
(`results/*.json`, driver `specs/verification/VER-0015-dftd3-cn-extra-kernels/scripts/run_6a.py`).

## Prototype (reverted)
Float copies of the positions in angular cell-list order (float4, built by an extra kernel), per-candidate float difference with
float minimum image (orthogonal box only), reject if the float r^2 >= T, else the unchanged double path. Bound: per axis
e_a = 3e-7 (2(|x1_a| + M_a) + L_a) (M_a = largest |coordinate|; 5 float roundings of 2^-24), E = sum e_a,
T = (rc + E)^2 (1 + 1e-6); candidates with | d_a | within e_a of L_a/2 are never rejected (minimum image decision may differ).
v1 used a per-pair bound, v2 the per-thread constant bound above.

## Result (CN kernel [ms], base -> prototype)
| system | v1 | v2 |
|---|---|---|
| water 41,472 | 0.706 -> 0.759 (0.93x) | 0.709 -> 0.760 (0.93x) |
| water 5,184 | 0.068 -> 0.079 (0.86x) | 0.069 -> 0.075 (0.92x) |

Gate (>= 1.2x at both): not met, the prototype is slower.

## Why (diagnosis)
The timing-only variant `cn_xf32` of VER-0015 is reproduced (CN 0.711 -> 0.375 ms at 41,472, 0.069 -> 0.052 at 5,184, `results/s1_6a_variants_recheck.json`),
but it has no double arithmetic at all. A pre-filter that keeps the exact double path for accepted candidates cannot skip it
in SIMT execution: 22 % of the candidates are accepted, so almost every warp iteration (32 lanes) has at least one accepted
lane (1 - 0.78^32 > 99.9 %) and executes the double difference for all lanes (predicated). The extra float work then only adds cost.
A gain would need a warp-level compaction of the accepted candidates or a float (non-identical) pair test; the first is beyond a
minimal prototype, the second changes the numbers (not allowed by the item's identity requirement).
