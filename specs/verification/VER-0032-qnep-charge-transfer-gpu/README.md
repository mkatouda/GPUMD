# VER-0032: charge_transfer qNEP on the GPU (DR-005 stage 2b)

- Status: steps 0-5 of stage 2b done (generator, parser, charge/BEC kernels in the large-box and small-box paths). Forces/virial/D-weight for charge_transfer models are NOT done (stage 2c): the model prints a warning at start-up and its forces are known wrong. Data tier: synthetic (random weights); no external model or data.
- Requirement DR-005; design DES-0005; ADR-0006; CPU reference VER-0029.

## Step 0: `scripts/make_ct_model.py OUTDIR [--seed N] [--rc-radial 4] [--rc-phi 4] [--no-ct]`
Writes `nep.txt` (nep4_charge2, types Na Cl, n_max 3 2, basis_size 4 4, l_max 4 0 0, ANN 10 neurons, dim 16; w1 ~ N(0,1.5), phi coefficients uniform in [0,0.3]) and five periodic cubic structures
(extxyz): n8/L13, n45/L14, n59/L16 (large-box kernels) and n12/L7.5, n10/L6 (small-box kernels). gpumd uses the small-box path when thickness <= 2.5 (rc_radial + 1) = 12.5 A for rc 4.

## nep.txt layout (legacy text format, `src/force/nep_charge/nep_charge.cu`)
```
nep4_charge2 2 Na Cl
cutoff rc_radial rc_angular MN_radial MN_angular
charge_transfer rc_phi          <- optional, IMMEDIATELY after cutoff (old binaries stop at the n_max line)
n_max / basis_size / l_max / ANN ...
values, one per line:
 1. ANN block, (dim+3)*num_neurons*num_types + 2:  per type w0 (neurons*dim), b0 (neurons), w1 (neurons*2); then sqrt_epsilon_inf, b1
 2. descriptor c, num_types^2 * ((n_max_radial+1)(basis_size_radial+1) + (n_max_angular+1)(basis_size_angular+1))
 3. q_scaler, dim values
 4. phi block (only with the charge_transfer line), at the very end: num_pairs * (basis_size_radial+1) values, pair-major
```
phi pair ordering (symmetric, upper triangle, row-major, t1 <= t2): `p = t1*nt - t1*(t1-1)/2 + (t2-t1)`; 2 types: p(0,0)=0, p(0,1)=1, p(1,1)=2.
Coefficient k of pair p is line `p*(basis_size_radial+1)+k` of the block; phi(r) = sum_k c_k f_k(r) fc(r), cutoff rc_phi.
NOTE (to confirm): the number of coefficients per pair is `basis_size_radial + 1` (k = 0..basis_size_radial, as for the descriptor `c`), not `basis_size_radial`.

## Step 1: parser (`NEP_Charge` constructor)
Stores rc_phi (`paramb`), phi coefficients (host `phi_coeff_host`, device `nep_data.phi_parameters`, `annmb.phi_coeff`); errors: line malformed, charge_mode not 2/3,
rc_phi <= 0, rc_phi > rc_radial, too few / too many numbers after the parameter block. Prints one summary line (and, after parsing, the warning that forces are not valid in this build). vdw / charge_vdw / spin models use the new-format reader, which rejects the unknown field (`unhandled field 'charge_transfer'`).
Tests: `tests_pytest/test_qnep_charge_transfer_parse.py` (marked fast).

## Steps 2-5: kernels (`src/force/nep_charge/nep_charge_transfer.cuh`, wired in `nep_charge.cu`)
- h = charge output of `find_descriptor` (copied to `nep_data.ct_h`); q_a = sum_b phi_ab (h_a - h_b) (double accumulation per atom, float result written to `nep_data.charge`, replaces `zero_total_charge`), V_a = sum_b phi_ab (r_a - r_b) (`nep_data.ct_V`).
  phi uses `find_fc_and_fcp`/`find_fn_and_fnp` with rc_phi; neighbours = the radial list.
- BEC: the Term B kernel writes all nine components (includes q 1; no `find_bec_diagonal`), then the Term A kernels (radial, angular) add V (x) dh/dr with atomicAdd: ONE pass gives all nine components
  (equivalent to three passes with weights V_x, V_y, V_z), then the unchanged `scale_bec`. The old `find_bec_radial/angular` do not run for these models. Array index [3 beta + alpha] = dP_beta/dr_alpha.
- Models without the line take the old code path (branches only on `paramb.charge_transfer`).

## Verification (RTX A6000, sm_86; float kernels, double dumps)
Reference (`scripts/ct_bec_fd.py`): the task text suggested Z_k = q_k 1 + sum_a (r_a - r_k)_MIC dq_a/dr_k with FD of gpumd's own charge. VER-0029 showed this holds only when rc_desc + rc_phi < L/2 (not true for the cells below except the wide one),
so the main reference is the FD of the single-valued pair-form P = sum_a h_a V_a built on the host (all lattice images) from h of the same weights without the line (q_noct = h - mean h; only differences of h enter).
The task formula is used additionally on a wide cell (n59, L18; 8 < 9) with FD of gpumd's charge dump. Static runs: `time_step 0`, `run 1`, `kspace ewald`, `dump_xyz 1 d.xyz precision double charge bec` (fresh directory per run: dump_xyz appends).
delta 0.02 A, central differences; "all" = every atom displaced in turn, else 12 evenly spaced atoms.
| structure, rc_radial/rc_phi, path | atoms | sum q (float dump) | max abs q | RMS(BEC/sqrt(eps) - Z_FD) | RMS Z |
|---|---|---|---|---|---|
| n8 L13, 4/4, large | all | 0 | 2.9e-3 | 3.1e-6 | 8.3e-3 |
| n45 L14, 4/4, large | all | -1.7e-8 | 0.61 | 7.2e-5 | 0.143 |
| n59 L16, 4/4, large | all | 4.9e-9 | 0.43 | 8.0e-5 | 0.128 |
| n12 L7.5, 4/4, small | all | -3.0e-7 | 0.56 | 1.8e-4 (delta 0.01: 4.6e-5) | 0.34 |
| n10 L6, 4/4, small | all | 3.2e-7 | 1.03 | 1.2e-4 | 0.29 |
| n8 L13, 5.5/4, small (forced) | all | 0 | | 1.2e-5 | 0.040 |
| n45 L14, 5.5/4, small (forced) | 12 | 1.6e-7 | | 7.1e-5 | 0.166 |
| n59 L16, 5.5/4, small (forced) | 12 | 1.3e-7 | | 5.7e-5 | 0.134 |
| wide n59 L18, 4/4, large; task formula with gpumd charge FD | 12 | 2.4e-8 | 0.64 | 9.2e-5 (pair-P reference: 8.9e-5) | 0.145 |
- Gates: sum q <= 1e-6 e (float) passes on all rows (the host double sum of the dumped floats agrees to 1e-8); max |q_gpumd - q_host| <= 3e-7 e; RMS(BEC - FD) <= 1e-3 e (A1) passes on all rows (max 1.8e-4).
- Noise: the error scales as delta^2 (n12: 1.8e-4 at 0.02, 4.6e-5 at 0.01), so it is the FD truncation error, not float noise. Float vs double: not run (kernels are float only).
- The first generator version used larger weights and gave |q| up to 5.7 e in the dense thin cells, where the float sum of q reached 2e-6 (float representation of |q| ~ 3); the model scale was reduced (PHI_MAX 0.3, w1 std 1.5) rather than the gate changed.
- Models without charge_transfer: static runs (water mode 2 and mode 1, BaTiO3 mode 2; `kspace ewald`, `time_step 0`) with the base build and the new build, each twice: `charge` and `forces` bit-identical in all 12 runs; `bec` differs by up to 1.8e-7 (water), 1.4e-6 (BaTiO3), 3e-7 (water mode 1)
  both base-vs-base and base-vs-new (float atomicAdd order), i.e. run-to-run noise of the existing kernel. `cuobjdump -sass` of `nep_charge.o`: all 22 pre-existing kernels identical to the base build after normalising constant-bank offsets (ParaMB grew).

## Commands
```
cd src && make CUDA_ARCH="-arch=sm_86" -j4
PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python; S=specs/verification/VER-0032-qnep-charge-transfer-gpu/scripts
$PY $S/ct_bec_fd.py src/gpumd /scratch/fd --structure thick_n45_L14 [--max-atoms 12] [--delta 0.02] [--rc-radial 5.5 --rc-phi 4.0]   # structure: a name in make_ct_model.STRUCTURES, or "wide"
$PY -m pytest -q tests_pytest/test_qnep_charge_transfer_parse.py tests_pytest/test_qnep_charge_transfer_bec.py
```

## Not verified
Forces, virial, energy derivative (D-weight) and NVE conservation for charge_transfer models (stage 2c); float vs double; realistic trained weights; training-side (`nep`) and `main_replica` kernels (stage 3); non-cubic cells.

## Review fixes (MAJOR 1/2 and minor items)
- Test-only opt-in: a charge_transfer model refuses to start (`PRINT_INPUT_ERROR`, message names "DR-005 stage 2c (forces/virial not implemented)") unless the environment variable
  `GPUMD_CHARGE_TRANSFER_BEC_ONLY=1` is set. It exists only to verify charges and BEC (VER-0032 scripts and tests set it); it is NOT a user feature and has no `doc/` page. With it the start-up banner
  ("forces/virial/energy-derivative terms are NOT valid in this build") is printed and forces remain invalid. Models without the line do not read the variable.
- Tests (`tests_pytest/test_qnep_charge_transfer_bec.py`): dense n45/L14 large-box case (RMS Z ~ 0.14), thin n12/L7.5 small-box case, rc_phi=3 < rc_radial=4, and a 3-type model (generator `nt=3`, pair index p = t1*nt - t1(t1-1)/2 + (t2-t1), 6 pairs);
  each gated by RMS error <= 1e-3 e, max abs error <= 1e-3 e, relative RMS error <= 1e-2 (and RMS Z > 0.02, so a near-zero BEC cannot pass). Also atom-permutation invariance of q and BEC (<= 1e-5) and bit-identical q of two static runs.
- Float neutrality limit: sum q is limited by the float representation of q, about 1e-7 sqrt(N) |q|_typ (e.g. 3e-7 for |q| ~ 1, N = 10); the 1e-6 e gate holds for |q| <~ 1 and is the reason the test model scale is kept moderate.
- Not covered by these tests: non-cubic cells, 4/5-body angular descriptor terms (`l_max` second/third entries and has_q_*; the generator writes l_max 4 0 0), float vs double, trained weights.
