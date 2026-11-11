# VER-0031: decomposition of the non-conservative BEC work of the current qNEP BEC (DR-005 side analysis, diagnosis only)

- Status: done 2026-11-11. No `src/` change. Verification type: finite-difference decomposition (not regression, not validation). Data tier: P (prototype model, internal use only).
- Follows VER-0028 (reuses its FD-Jacobian method, scratch raw-charge build, C4 definition) and the requester's note `GPUMD-polymer-apps/docs/work-requests/reply-to-gpumd-4-7-decomposition.md`.
- Environment: parent `src/` at `d7806f1b`, RTX A6000 (sm_86), CUDA 12.3, `make CUDA_ARCH="-arch=sm_86" -j4 gpumd`, both binaries built in scratch (copies of `src/`): `src_real` unmodified, `src_raw` with both `zero_total_charge` launches commented out (only change; one expected "declared but never referenced" warning there).
- Inputs (read-only): `pq_oligo/model.xyz` (59 atoms, 40 A cube), N50 `nep.txt` (`v3/nscan2/N50_g10000_s1`), from GPUMD-polymer-apps.

## Math
J[a,k] = dq_a/dr_k (raw charges), S_k = sum_a J[a,k], Jn = J - S/N, rel_ak = u_a - u_k (plain unwrapped).
BEC_kernel/s = qn 1 + 0.5 T1(J) + 0.5 T2(J); C4 = qn 1 + T2(Jn);
BEC_kernel/s - C4 = W + S_term, W = 0.5 (T1(J) - T2(J)) [cause ii: 0.5/0.5 weights], S_term = T2(J) - T2(Jn) = (1/N) sum_a rel_ak x S_k [cause i: non-local conservation of the raw total charge].
Work rate per atom: x[k,beta] = sum_alpha Z[k,beta,alpha] v[k,alpha] (index b[3 beta+alpha] = dP_beta/dr_alpha). All quantities are divided by s (the s = 0.989074 factor is removed from the measured BEC).
Units: BEC/charges in e, positions in A, velocity in A/fs as written by `dump_xyz ... velocity` (the writer multiplies by 1/TIME_UNIT_CONVERSION, `measure/dump/dump_xyz.cu`), so work rates are e*A/fs. "RMS" = RMS over atoms x 3 components of x[k,beta]; "sum" = sum over atoms (3-vector, the contribution to dP/dt).

## Trajectory (own, zero field)
`velocity 300 seed 4711; time_step 1; ensemble nvt_nhc 300 300 100; kspace pppm; dump_xyz 1500 precision double unwrapped_position velocity charge; run 20000` (20 ps, real binary, 18 s). 13 frames written (1.5..19.5 ps), the first 12 (1.5..18 ps) used. Unwrapped extent of the molecule 10-23 A (< 40 - rc), so plain differences = MIC for the local terms. Mean T 300 K scale (thermo.out 302-331 K at the printed points).

## Results (delta = 0.02 A, 12 frames; units e*A/fs)
Check of the kernel reading per frame: RMS(BEC_real/s - kernel formula(J)) = 3.2e-4..3.9e-4 e (as VER-0028); max|q_real - (q_raw - mean)| <= 1.2e-6.

| frame | RMS W.v | RMS S.v | RMS (W+S).v | RMS (BEC/s-C4).v meas | closure RMS/meas | RMS BEC.v/s | RMS C4.v | corr(W.v,S.v) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0007 | 0.0035 | 0.0035 | 0.0035 | 6e-4 | 0.0024 | 0.0045 | -0.14 |
| 1 | 0.0012 | 0.0050 | 0.0050 | 0.0050 | 4e-4 | 0.0027 | 0.0055 | -0.17 |
| 2 | 0.0020 | 0.0052 | 0.0045 | 0.0045 | 4e-4 | 0.0022 | 0.0049 | -0.51 |
| 3 | 0.0018 | 0.0051 | 0.0047 | 0.0047 | 5e-4 | 0.0023 | 0.0055 | -0.40 |
| 4 | 0.0025 | 0.0055 | 0.0045 | 0.0045 | 7e-4 | 0.0031 | 0.0058 | -0.58 |
| 5 | 0.0020 | 0.0041 | 0.0041 | 0.0041 | 5e-4 | 0.0027 | 0.0045 | -0.25 |
| 6 | 0.0027 | 0.0061 | 0.0056 | 0.0056 | 4e-4 | 0.0025 | 0.0061 | -0.39 |
| 7 | 0.0032 | 0.0087 | 0.0076 | 0.0076 | 3e-4 | 0.0026 | 0.0084 | -0.49 |
| 8 | 0.0029 | 0.0066 | 0.0061 | 0.0061 | 4e-4 | 0.0028 | 0.0062 | -0.40 |
| 9 | 0.0030 | 0.0075 | 0.0069 | 0.0070 | 4e-4 | 0.0032 | 0.0070 | -0.39 |
| 10 | 0.0045 | 0.0074 | 0.0077 | 0.0078 | 3e-4 | 0.0035 | 0.0078 | -0.24 |
| 11 | 0.0036 | 0.0094 | 0.0092 | 0.0092 | 3e-4 | 0.0032 | 0.0095 | -0.25 |
| mean | 0.0025 | 0.0062 | 0.0058 | 0.0058 | - | 0.0028 | 0.0063 | -0.35 |

(Per-frame sum 3-vectors are printed by `decomp_summary.py`; not repeated here.)

Gate G1 (closure RMS((W+S).v - meas.v) <= 10 % of RMS(meas.v) on >= 10 of 12 frames): met on 12 of 12 frames, max ratio 7e-4 (the identity is exact up to FD/float noise). delta = 0.01 on frame 0: W, S, meas RMS identical to 0.02 to <= 8e-6 (closure 9e-4); kernel-formula residual 5.6e-4 e (vs 3.3e-4, the 1/delta float noise).

Cell-summed work rate, mean +/- standard error over the 12 frames (x, y, z; e*A/fs):
- sum W.v: (-0.0065, +0.0036, -0.0039) +/- (0.0054, 0.0039, 0.0049)
- sum S_term.v: (+0.0040, +0.0045, +0.0032) +/- (0.0072, 0.0066, 0.0165)
- sum (W+S).v = sum (BEC/s - C4).v: (-0.0025, +0.0081, -0.0006) +/- (0.0030, 0.0052, 0.0140)
All are consistent with zero within 1-2 SE. 12 frames 1.5 ps apart (one trajectory) cannot decide drift vs diffusion.

## Which cause dominates
S_term (cause i) dominates the per-atom work rate: mean RMS 0.0062 vs 0.0025 for W (ratio 2.5; mean of RMS_W^2/(RMS_W^2+RMS_S^2) = 0.14, i.e. ~86 % of the squared RMS from S_term). The two are anticorrelated (corr -0.14..-0.58, mean -0.35), so RMS(W+S) (0.0058) < sqrt(RMS_W^2+RMS_S^2) = 0.0067. The S_term share grows with the molecular extent change: RMS|S| = 0.15 e/A (frame 0, extended) to 0.43 (compact frames).

## Interpretation (limited to the numbers)
The measured (BEC/s - C4).v is reproduced to <= 7e-4 relative by W + S_term, so the two named causes account for all of the difference on this trajectory. Per atom, the non-local total-charge term S_term is about 2.5 times larger than the weight asymmetry W, with a partial cancellation (negative correlation); neither cell sum differs from zero beyond about 2 standard errors (and the S_term sum has the largest scatter, SE up to 0.017). Whether the accumulated work is a drift or a diffusive fluctuation is not decided by these 12 frames. Note that RMS(BEC.v/s) here (0.0028) is a per-atom measure and is NOT comparable to the requester's 0.038 (cell-total dP/dt, different normalisation, other trajectory); the per-atom ratio RMS(meas.v)/RMS(C4.v) = 0.92 here is likewise not the requester's 0.70 (cell-total, other trajectory) and no claim is made about agreement. No statement is made about the causal link to the pseudo-dissipation.

## Files / commands
- `scripts/decomp_frame.py` (FD Jacobian with the raw binary, real-binary BEC/q, W, S_term, C4, work rates), `scripts/decomp_summary.py` (table/means). Scratch outputs (npz, binaries, trajectory) stay in the session scratchpad.
- `SC=<this dir>/scripts; PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python`; per frame i: `$PY $SC/decomp_frame.py traj/f.xyz i nep.txt $S/src_raw/gpumd $S/src_real/gpumd res/fII_d0.02.npz 0.02 4 $S/work` (about 55 s each, 354+2 single-step runs); `$PY $SC/decomp_summary.py res d0.02`.

## Not verified
Other models/structures, periodic systems, a second trajectory/seed, a longer sample (drift vs diffusion), time integration of the work (only instantaneous rates), the C4 chain-rule check on frame pairs (not requested), training-side numerics.
