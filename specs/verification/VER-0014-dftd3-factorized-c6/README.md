# VER-0014: DFT-D3 factorized C6 interpolation weights (DES-0001 Phase 7 Stage 2 item (v))

- Date: 2026-11-04. Status: verification (timing, round-off agreement); NOT validation. Requirement / design: `DR-001` AC-1, AC-2, AC-4;
  `DES-0001` Phase 7 (v). Outcome: **S1 met, S2 NOT met (water 41,472: 6.32 ms vs gate 6.0 ms)**; S3 (factorization-specific tests) not started
  because the S2 gate failed. The change is left uncommitted in the worktree for the human decision.
- Code under test: worktree `wt-d3-phase7c` (base HEAD `db2bf407`, item (i) included), changed `src/force/dftd3.cu`, `src/force/dftd3.cuh`.
- Environment: RTX A6000 (sm_86), CUDA 12.3, NEP89, `dftd3 pbe 12 6`. Timing base = unchanged HEAD built in a scratch copy of `src/`;
  AC-2 baseline = `master-base` binary. Harness = VER-0012 `bench.py` via `scripts/run_s2.py` (D3 total = sum of nsys kernel durations, VER-0004 section 3).

## Change
The interpolation weight separates, L_ij = w_1[i] w_2[j] with w[k] = exp(-4 (CN - CN_ref_k)^2). The CN kernels (`find_dftd3_coordination_number_small_box`,
`find_dftd3_coordination_number_large_box`) now also store `w[k]` (`max_cn` = 5 floats per atom, zero beyond the element's reference count) via
`store_cn_weights` into the new `DFTD3::cn_weight`. The force kernels (`add_dftd3_force_small_box`, `find_dftd3_force_large_box`) call
`load_cn_weights` once per central atom (w, dw = w(-8 diff), sums) and `get_c6_factorized` per pair: t_a = sum_b c6_ab w2_b; Z = sum_a w1_a t_a;
dZ = sum_a dw1_a t_a; W = W_1 W_2, dW = dW_1 W_2. No `exp` in the pair loop. The extra (angular) kernels are unchanged (they do not use C6 weights).
- Normalization / fallback: identical rule to before (`W < 1e-30` -> c6 of the last reference pair, dc6 = 0; `num_cn == 1` for both -> first reference).
  W = W_1 W_2 equals the sum of the unfactorized L_ij up to round-off, so the fallback condition is unchanged; individual denormal weights only
  matter when W_1 W_2 is below 1e-30 (fallback). Underflow of `exp(-4 diff^2)` to zero for one reference point is harmless (weight 0, as before).
- Memory: 20 bytes per atom (N x 5 floats): 0.83 MB at N = 41,472, 0.08 MB at N = 4,000.
- Tried and rejected (timing, `w41472`, same harness): (a) stride 8 with two float4 loads: 6.49 ms (worse); (b) per-atom normalized weights n = w/W
  (division removed from the pair loop, dn = -8 n (diff - sum n diff)): force 4.84 ms vs 4.81 ms, 80 vs 72 registers, no gain; (c)
  `__launch_bounds__(128, 8)` on the force kernel: 6.90 ms (worse). The kept variant is the plain one above. 0 register spills (`-Xptxas -v`);
  force kernel registers 72 -> about 79-80.

## S1 gate (prototype; w41472; base/new alternated 3x, median of medians; `results/s1.json`)
| quantity | value | gate |
|---|---|---|
| D3 total, water 41,472 | base 6.856 -> new 6.239 ms (first plain variant; later repeats 6.28-6.32) | <= 6.4 ms: MET |
| AC-1 accuracy (`scripts/accuracy.py`, `results/accuracy.txt`) | largest |new - simple-dftd3| / tolerance: graphite dE 0.58, dF 0.52, others <= 0.26 | "within 1/2 of tolerance": see note |
Note: graphite already has dE 3.94e-6 (0.50), dF 4.6e-7 (0.46) with HEAD (0.59 / 0.53 with the item (i) build recorded in VER-0013); this is the float
error of the existing implementation against the double-precision reference, not caused by the factorization. The change itself (new - HEAD, force-kernel
per-atom outputs) is at most: dE 4.2e-7 (cu_large, tol 1.2e-3), dF 3.1e-8 (graphite, tol 1e-6), per-atom virial 1.2e-7 (cu_large), i.e. <= 4 % of every tolerance;
new errors against the reference are equal to or slightly smaller than HEAD's.

## S2 gate (`scripts/run_s2.py`, `results/s2.json`, `results/s2.log`; D3 total [ms], base = HEAD, new = factorized)
| system (path) | base | new | speedup | gate |
|---|---|---|---|---|
| water 81 (small) | 0.101 | 0.097 | 1.04x | no slowdown: ok |
| water 648 (small) | 0.519 | 0.504 | 1.03x | (NL kernel 0.40 ms dominates) |
| water 2,187 (small) | 0.785 | 0.726 | 1.08x | ok |
| water 8,748 (small) | 8.288 | 8.149 | 1.02x | ok |
| cluster 972 (large) | 1.827 | 1.484 | 1.23x | ok |
| water 5,184 (large) | 0.930 | 0.767 | 1.21x | >= 1.2x at 648 or 5,184: MET (margin 1 %) |
| water 41,472 (large) | 6.931 | 6.316 | 1.10x | <= 6.0 ms: NOT MET |
| Cu 4,000 (large) | 0.586 | 0.541 | 1.08x | no slowdown: ok |
No point is slower than 1.03x. Force kernel at 41,472: 5.36 -> 4.81 ms; the CN kernel (0.71 ms) and extra kernel (0.75 ms) are unchanged. The measured
upper bound with C6 replaced by a constant (VER-0012) was 5.59 ms total; the factorization recovers about half of the C6 share (0.55 of 1.26 ms).

## Other checks
- AC-1: `pytest -q test_dftd3_reference.py`: 14 passed (137 s), tolerances unchanged.
- AC-2: `run_regression.py --suite full --case 'dftd3_*'` (baseline `master-base`): strict manifest tolerance fails 2 of 3 as for any round-off change
  (`results/regression_dftd3.log`); under the AC-2 rule (rtol 1e-5, atol 1e-12, `scripts/compare_ac2.py`, `results/ac2_dftd3.txt`) all pass; largest thermo.out
  rel difference 1.5e-6 (dftd3_kspace_one_each), restart.xyz rel 1.6e-7.
- Determinism: small-box (water 648), 100-step NVE thermo.out bit-identical run to run. Large-box (water 5,184, Cu 4,000, cluster 972): thermo.out differs
  run to run for BOTH the HEAD build and the new build (NEP alone is bit-reproducible), i.e. the pre-existing cell-list ordering non-determinism of the
  large-box path; not worse than now. Per-`num_sub` bit reproducibility of the force kernels was not re-run.
- Build: warning-clean for `dftd3.cu` (an unrelated `eam_alloy.cu` warning exists); clang-format of the changed regions clean.

## Not done / limits
- S3 not started (S2 gate failed): no factorization-specific per-atom table vs HEAD for extreme CN, no extra tests; only the existing reference cases
  and per-atom candidate-HEAD differences above (`accuracy.txt`) were evaluated. Underflow behaviour is argued, not tested with isolated/extreme-CN cases.
- HIP not built; only sm_86; `--suite full`, NVE energy conservation, and independent review not run.
