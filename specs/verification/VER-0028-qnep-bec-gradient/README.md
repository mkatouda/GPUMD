# VER-0028: qNEP BEC vs the gradient of the dipole (DR-005 stage 1, diagnosis only)

- Status: stage 1 done (2026-11-11). No `src/` change. Verification type: analytic/finite-difference (not regression, not validation).
- Requirement: DR-005 (draft). Data tier of the inputs: Tier P (prototype model).
- Environment: GPUMD worktree HEAD `3b2c8db8` (= master), RTX A6000 (sm_86), CUDA 12.3.103, `make CUDA_ARCH="-arch=sm_86" -j4`.
- Inputs (read-only, from GPUMD-polymer-apps): model N50 `outputs/S-005-ptfe-qnep-prototype/v3/nscan2/N50_g10000_s1/nep.txt`
  (sha256 `ca6fd522...`; `cutoff 6 5`, `sqrt_epsilon_inf` = 0.989073634 = parameter index 6080); amorphous `demo/stab_amor1/model.xyz`
  (248 atoms, 15.90 A cube) and oligomer `demo/pq_oligo/model.xyz` (59 atoms, 40 A cube).
- Scratch: all runs in the session scratchpad (`.../scratchpad/{in,raw_*,base_*}`); nothing generated in the repo.

## Kernel facts (read from the code)
- Normal path `src/force/nep_charge.cu` (`compute_large_box`) and `nep_charge_small_box.cuh`: after `find_descriptor`,
  `zero_total_charge` sets `qn = q - mean(q)` (float). `find_bec_diagonal` writes `qn*1`; `find_bec_radial/angular`
  use `charge_derivative` (derivative of the RAW charge, no neutralization term) and add `+0.5 r12 x f12` to n1 and `-0.5 r12 x f12` to n2;
  `scale_bec` multiplies everything by `sqrt_epsilon_inf` (s = 0.98907).
  Hence BEC_k = s [ qn_k 1 + 0.5 T1_k + 0.5 T2_k ], T1_k = sum_j rel_jk x dq_k/dr_j, T2_k = sum_a rel_ak x dq_a/dr_k
  (rel = (r_a - r_k) MIC, q raw, index b[3 beta + alpha] = dP_beta/dr_alpha). This is "H0".
- The neutralization chain-rule correction exists only for the force side (`zero_mean_D_real`); it is not in the BEC.
- Box path: oligomer (thickness 40 A > 2.5(rc+1) = 17.5 A) uses the normal kernels; amorphous (15.9 A) uses the small-box kernels.
- Training side (`src/main_nep/nep_charge/nep_charge.cu` `find_bec_radial/angular`, `common/nep_long_range.cu`): same structure.
  Diagonal is the shifted charge `q + (q_ref - sum q)/N` (`zero_total_charge` of the training side, `charge_ref` = structure total charge,
  normally 0), pair terms `0.5 r12 x f12` from the raw derivative, then `scale_bec`. The pair vectors come from the training neighbor list
  (`x12/y12/z12`), not `apply_mic`. Same formula as gpumd (A4 is a question of numerics, not of definition); `compiled_kernel_` variants not read.
  `main_replica/force/nep_charge.cu` has the same `zero_total_charge` and BEC kernels (diff of the BEC kernel region against `src/force`: formatting only in the lines checked; not exhaustively verified).

## Method
FD Jacobian of the RAW charges, J[a,k,alpha] = dq_a/dr_k,alpha, central difference, 6N gpumd single-step runs
(`dump_xyz ... charge bec`, double precision, `kspace pppm`). Raw charges come from a SCRATCH build (copy of `src/`) in which the
`zero_total_charge` launch is disabled (the only change; the repo binary is untouched). The Jacobian of the charges seen by the real
gpumd is Jn = J - S/N with S_k = sum_a J[a,k] = dM/dr_k, M = sum of raw charges (checked: real q equals q_raw - mean(q_raw) to 1.7e-6).

## Results
Stage 1-0 (build): `cd src && make CUDA_ARCH="-arch=sm_86" -j4`; builds; recompiling `nep_charge.cu` alone gives no compiler warning
(only the pre-existing nvlink `libdl` skip message at link time).

Stage 1-a (reproduction, real binary, Jn, as in `bec_code_formula.py`/`bec_formula_fit.py`; `bec_repro_1a.py`):

| quantity | oligomer 59 | amorphous 248 | requester |
|---|---|---|---|
| RMS(GPUMD - exact gradient) [e] | 0.4301 | 0.1877 | 0.430 / 0.188 |
| fit a, b (BEC - q1 = a T1 + b T2) | 0.682, 0.049 | 0.493, 0.483 | 0.68, 0.05 / 0.493, 0.483 |
| fit residual [e] | 0.0727 | 0.0227 | 0.073 / 0.023 |
| RMS BEC [e] | 0.391 | 0.442 | 0.391 / 0.442 |

Gate (+-5 %): met. (`bec_integrability.py` with the worktree binary, oligomer, 8 atoms: RMS 0.463 e, consistent.)

Stage 1-b (`bec_hypotheses.py`; H0 = kernel formula with RAW J; H1 = H0 + exact neutralization term -(1/N) sum_a rel_ak x S_k):

| delta [A] | oligomer RMS(GPUMD - H0) [e] | amorphous RMS(GPUMD - H0) [e] |
|---|---|---|
| 0.002 | 0.0029 | - |
| 0.005 | 0.0011 | 0.0021 |
| 0.01 | 0.0006 | - |
| 0.02 | 0.0003 | 0.0005 |

- H0 reproduces GPUMD on both structures; the residual falls as 1/delta (float noise of q in the FD), 3e-4 and 5e-4 e at delta = 0.02 (gate 1e-3: met).
  Fit with the raw J: a = b = 0.500 on both structures (residual 0.0011 / 0.0021 e).
- H1 does not: RMS(GPUMD - H1) = 0.431 e (oligomer), 0.029 e (amorphous). H1 is rejected; the BEC has no neutralization term.
- Explanation of the oligomer/amorphous difference (request 2.2): the formula is the same and correct for both (and for both kernel paths).
  The "(0.68, 0.05)" fit was an artefact of using the Jacobian of the neutralized charge: Jn = J - S/N is nonlocal (every pair nonzero, 3481/3481 oligomer pairs),
  RMS|S| = 0.140 / 0.156 e/A, and the nonlocal term -(r_c - r_k) x S_k is large for a 33 A long molecule in a 40 A cell (RMS 0.43 e), small for the 248-atom
  cell (0.03 e). The sum of raw charges, 0.742 (oligomer) and -0.680 (amorphous), is the same size, but the effect scales with the distance to the centroid / N.
- Stage 1-c: not needed (1-b gate met). scale_bec (1 %) is included in H0; small-box vs normal path both follow H0.

Stage 1-d (numbers only; all x s; rel = MIC unless "plain"; the kernel is the reference column):

| candidate | RMS vs kernel [e] oligo / amor | RMS(Z) oligo / amor | max abs sum_k Z_k [e] oligo / amor |
|---|---|---|---|
| kernel (0.5 T1 + 0.5 T2, qn on diagonal) | 0 / 0 | 0.391 / 0.442 | 0 / 0 |
| C1: qn 1 + T2(raw J), local MIC, reference k, no neutralization term | 0.122 / 0.186 | 0.359 / 0.373 | 8.9 / 14.4 |
| C2: q_raw 1 + T2(raw J) | 0.126 / 0.186 | 0.361 / 0.373 | 9.6 / 15.1 |
| C3: qn 1 + T2(Jn), MIC (the requester's "exact") | 0.425 / 0.188 | 0.600 / 0.374 | 0.95 / 13.9 |
| C4: qn 1 + T2(Jn), plain unwrapped differences = grad of sum qn_a r_a | 0.648 / 0.905 | 0.796 / 1.001 | 0.011 / 0.12 |

Notes: the requirement for a translation-invariant dipole current is sum_k Z_k = (sum of the charges on the diagonal) 1, i.e. 0 for qn.
The kernel satisfies it exactly (pair terms cancel). C1/C2 fail it (sum = -sum_k r_k S_k x ..., S_k = dM/dr_k != 0), C3 fails because MIC of
the nonlocal term wraps, C4 satisfies it (FD noise level) but depends on the unwrapped coordinates (centroid term, multivalued under wrapping)
and is 0.65-0.90 e away from the kernel. Any strictly local BEC equals an exact gradient of a translation-invariant P only if S_k = 0, i.e. if the
total raw charge is locally conserved by the model.

## Observations for D1 (not a decision)
- Cheapest kernel change reproducing the requester's "local exact" T2-only form (C1) is a coefficient change (n1: 0, n2: -1 instead of +-0.5), but C1 breaks
  sum_k Z_k = 0 (8.9-14.4 e) and is therefore not recommended on its own.
- The root of the non-integrability is that neutralization by mean subtraction makes S_k nonzero; a definition that is local, a gradient, and
  translation invariant needs S_k = 0 by construction (a model change, with retraining, A6) or accepts the centroid term (C4). Human decision.

## Files
- `scripts/bec_code_formula.py`, `bec_integrability.py`, `bec_formula_fit.py`: copies of the S-005 scripts; only the binary path changed (`$GPUMD_BIN`, default `<repo>/src/gpumd`); `bec_code_formula.py` also saves `q0.npy`, `b0.npy`.
- `scripts/bec_base.py`: real-binary base run, writes `q.npy`, `bec.npy`, `J_neut.npy`. `scripts/bec_repro_1a.py`: stage 1-a numbers. `scripts/bec_hypotheses.py`: stages 1-b and 1-d.

## Commands (scratch dir S; `PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python`, `SC=<repo>/specs/verification/VER-0028-qnep-bec-gradient/scripts`)
```
cp -a <repo>/src $S/src_raw   # then comment out both zero_total_charge launches in force/nep_charge.cu; make CUDA_ARCH="-arch=sm_86" gpumd
GPUMD_BIN=$S/src_raw/gpumd $PY $SC/bec_code_formula.py in/{oligo,amor}.xyz in/nep.txt $S/raw_{oligo,amor} 0.005 4   # also delta 0.002 0.01 0.02
$PY $SC/bec_base.py in/X.xyz in/nep.txt raw_X base_X          # real binary (worktree src/gpumd)
$PY $SC/bec_repro_1a.py in/X.xyz base_X
$PY $SC/bec_hypotheses.py in/X.xyz raw_X base_X/bec.npy 0.989073634
$PY $SC/bec_integrability.py in/oligo.xyz in/nep.txt integ_oligo 8 0.005
```

## Not verified
Crystal (702 atoms), other models (N20), other structures, replica binary numerics, training-side numerics (A4), compiled-kernel variants on the training side, 
FD noise reduction beyond delta = 0.02, time-evolution checks (A2/A3).
