# VER-0034: charge_transfer qNEP forces and virial on the GPU (DR-005 stage 2c)

- Status: implemented and verified 2026-11-11 (RTX A6000, sm_86, CUDA from /usr/local/cuda; float kernels, double accumulation of the new sums). Type: finite difference of gpumd's own energy (verification) + regression against the stage-2b build. Synthetic data (VER-0032 generator, random weights). Not validation.
- Requirement DR-005; DES-0005 sections 2, 3, 6; ADR-0006; reference formulas VER-0033; charge/BEC kernels VER-0032.

## Implementation (`src/force/nep_charge/nep_charge_transfer.cuh`, wired in `nep_charge.cu`, large-box and small-box paths)
After the long-range solver (ewald/pppm, charge_mode 1 real-space part, mode 3 self energy) has produced D_a = dE/dq_a (`nep_data.D_real`), for charge_transfer models
`zero_mean_D_real` is NOT run; the new kernel `ct_find_Dt_and_pair_force_{large,small}_box` (own radial list, rc_phi) computes per atom, in double:
Dt_k = sum_b phi_kb (D_k - D_b) -> `nep_data.ct_Dt` (separate array; neighbours' D are read), and the explicit pair force/virial
F_k += G v/d, W -= 0.5 G v (x) v / d with G = phi'(d)(h_k - h_b)(D_k - D_b), v = r_b - r_k (added to `force_per_atom` and `virial_per_atom` of atom k, index map 0 3 4 / 6 1 5 / 7 8 2).
`ct_Dt` is passed as the weight `g_D_real` of the unchanged `find_force_radial` / `find_partial_force_angular` (`find_properties_many_body` does forces and virial of the angular part from f12) and of the small-box `find_force_radial_small_box` / `find_force_angular_small_box`.
Models without charge_transfer use the old branch (`zero_mean_D_real`, `nep_data.D_real`), unchanged. The BEC kernels of stage 2b are untouched. The stage-2b refusal (`GPUMD_CHARGE_TRANSFER_BEC_ONLY`) and the "forces NOT valid" banner are removed.

## Gate 1: force vs central FD of the total energy (`scripts/ct_forces_fd.py`, `kspace ewald`, `time_step 0`, `dump_xyz precision double`, delta 0.005 A, all atoms)
"Old" = the same generator without the charge_transfer line (ordinary qNEP, same structure, rc 4/4): its error is the float/FD noise floor.
| structure (path) | N | CT rms_F | CT RMS err (max) | OLD rms_F | OLD RMS err (max) |
|---|---|---|---|---|---|
| thick_n8_L13 (large) | 8 | 0.096 | 5.6e-6 (1.3e-5) | 0.152 | 1.0e-5 (3.1e-5) |
| thick_n45_L14 (small, rc 4) | 45 | 0.70 | 1.7e-4 (5.9e-4) | 12.7 | 3.3e-3 (9.8e-3) |
| thick_n59_L16 (small, rc 4) | 59 | 0.56 | 9.8e-5 (2.3e-4) | 7.8 | 3.0e-3 (1.0e-2) |
| thin_n12_L7.5 (small) | 12 | 1.1 | 1.2e-4 (3.1e-4) | 8.27 | 7.9e-4 (1.9e-3) |
| thin_n10_L6 (small) | 10 | 1.47 | 1.5e-4 (3.5e-4) | 4.56 | 5.2e-4 (1.5e-3) |
Gate (RMS error <= 3x old noise floor and <= 1e-3 eV/A): met in all rows (CT error is below the OLD error in every row; the OLD model has larger |F|, hence a larger absolute float noise).
Large-box kernels need L > 2.5 (rc_radial + 1): with rc 4 none of the five structures except n8/L13 is large-box; the large-box path is therefore additionally exercised by rc 3.5/3.5 on n45/L14 in the pytest file (see Tests).
Other runs: `kspace pppm`, n45/L14, 8 atoms: CT 2.4e-4 (max 8.8e-4) vs OLD 3.3e-3; rc_radial 5.5 / rc_phi 4 (forced small-box path), n45/L14, 8 atoms: CT 1.3e-4 vs OLD 7.9e-3. All five VER-0032 structures were run with ewald; thick_n45_L14 only with pppm.

## Gate 2: virial vs FD of the energy under homogeneous strain (r -> F r, lattice -> F lattice; F = 1 + e E_ab, six components xx yy zz xy xz yz; compare with the dumped total virial, symmetrised)
| structure | rms_W | CT RMS err (max) | OLD RMS err (max) |
|---|---|---|---|
| thick_n8_L13 | 0.46 | 5.9e-5 (1.4e-4) | 4.4e-5 (8.9e-5) |
| thick_n45_L14 | 13.4 | 6.5e-4 (1.3e-3) | 2.2e-2 (5.0e-2) |
| thick_n59_L16 | 13.3 | 1.1e-3 (2.2e-3) | 4.9e-3 (9.6e-3) |
| thin_n12_L7.5 | 4.1 | 2.3e-3 (5.7e-3) | 8.3e-3 (1.5e-2) |
| thin_n10_L6 | 12.3 | 1.5e-3 (2.7e-3) | 8.3e-3 (2.0e-2) |
Interpretation of the gate: the virial is in eV with magnitude 4-13, so the force bound 1e-3 eV/A is not meaningful in absolute terms; the gate "<= 3x the old-model noise floor" is met everywhere (CT <= OLD in 4 of 5 rows; the exception thick_n8 is 1.3x, both tiny). Relative RMS error CT: 1.3e-4 (n8), 4.9e-5 (n45), 8.1e-5 (n59), 5.7e-4 (n12), 1.3e-4 (n10).
The absolute 1e-3 would be exceeded by thick_n59 (1.1e-3), thin_n12 (2.3e-3), thin_n10 (1.5e-3): this is flagged to the owner; I did not loosen anything, and the virial test in pytest uses the relative bound 3e-3 |W|_RMS.
Thermo pressure: thermo.out P V / 160.2177 = dumped virial + kinetic (diff 0.26-0.36 on the thin cell, 1.0-1.3 on n45, equal to 2K/3 per direction up to the velocity anisotropy; the OLD model shows the same offset), i.e. thermo P is the dumped virial/V plus the kinetic term, so the FD check of the dump carries over.

## Gate 3: NVE (2000 steps, 0.5 fs, `velocity 300`, `kspace ewald`, thermo every 20 steps, E = K + U; linear-fit drift over the run, max |E - E0|)
| structure | CT drift / max dev (eV) | OLD drift / max dev (eV) |
|---|---|---|
| thick_n45_L14 | -5.8e-5 / 1.2e-3 | 2.9e-3 / 5.4e-2 |
| thin_n12_L7.5 | 1.8e-4 / 1.7e-3 | 6.4e-3 / 2.1e-1 |
(The random-weight models heat up, <K> is 12.9 eV (CT) and 82 eV (OLD) for 45 atoms; the OLD run is much more violent, so it is only an upper reference.)

## Gate 4: invariance (tests_pytest/test_qnep_charge_transfer_forces.py, thin n12/L7.5 and n45/L14 rc 3.5)
Static run twice: forces and energy bit-identical; atom permutation: max |dF| <= 1e-5, |dE| <= 1e-5; translation by (0.37, -1.21, 2.05) A: |dF| <= 1e-4, |dE| <= 1e-4; |sum F| <= 1e-5 (all asserted, all pass).

## Gate 5: old models unchanged (base = `git archive 480ba096` built in a scratch dir, same flags)
Static runs (`time_step 0`, `kspace ewald` and `pppm`, dump_xyz double charge/force/virial/energy) of qnep_mode1/mode2 water (63 atoms) and BaTiO3 (40 atoms): forces, charge, energy bit-identical in all 8 runs; virial bit-identical in 7 of 8 (mode1 BaTiO3 ewald differed once; a rerun of base vs base and base vs new gave identical virial, i.e. atomicAdd order noise of the existing kernels, as in VER-0032). thermo.out columns differ run to run also between two runs of the base build (random initial velocities). `cuobjdump -sass` of `nep_charge.o`: all 30 pre-existing kernels identical to the base build after normalising constant-bank offsets (`c[0x0][..]`); the only new functions are the two `ct_find_Dt_and_pair_force_*` kernels.

## Tests and mutation checks
`tests_pytest/test_qnep_charge_transfer_forces.py` (8 tests, ~23 s): force FD (RMS <= 1e-3, max <= 3e-3, RMS F > 0.1), virial FD (RMS <= 3e-3 RMS W, RMS W > 1), NVE 600 steps (|drift| <= 2e-3, max dev <= 5e-3 eV), permutation/translation/determinism, on thin n12/L7.5 (small-box kernels, rc 4/4) and n45/L14 with rc 3.5/3.5 (large-box kernels).
Mutations applied locally to `ct_pair_force` / `ct_store_force` (each rebuilt, then restored): (1) Dt = 0 (descriptor weight dropped), (2) pair force and virial dropped, (3) pair term negated: each makes 6 of the 8 tests fail (force FD, virial FD, NVE, on both structures); the permutation/determinism tests pass (they are not sensitive to a consistent wrong force).
`test_qnep_charge_transfer_parse.py` was updated (opt-in tests replaced by: an NVE run of a charge_transfer model works without any environment variable); `test_qnep_charge_transfer_bec.py` and `ct_bec_fd.py` no longer set the variable. All 31 `test_qnep_charge_transfer_*.py` tests pass.

## Review follow-up (2026-11-11)
### A. Electric-field tests (`scripts/ct_field_tests.py`, pytest `test_qnep_charge_transfer_field.py`)
Convention (src/measure/add_efield.cu): `add_efield ... bec` adds F_alpha = sum_beta bec[3 beta + alpha] E_beta, bec = sqrt(eps_inf) Z, Z[3 beta + alpha] = dP_beta/dr_alpha, so Z is NOT symmetric and F = grad_r(sqrt(eps_inf) E.P_direct); the thermo energy does NOT contain -E.P (add_efield adds forces only).
P_direct = sum_a h_a V_a = -sum_pairs phi (h_a - h_b) v (all lattice images, float64 host); h = gpumd charge dump of the same weights without the charge_transfer line (P depends on differences of h only).
1. Field force vs central FD (delta 0.02) of sqrt(eps) E.P_direct, E = (0.3, -0.2, 0.5) V/A, static, 6 atoms (pytest: 3): thin n12 rms F_field 0.33, RMS error 1.2e-4 (max 2.6e-4), BEC error 1.7e-4; thick n45/rc 3.5 0.050, 2.5e-5 (8.9e-5), 3.2e-5; triclinic thin 0.37, 1.3e-4 (3.0e-4), Z 3.2e-4; triclinic thick 0.060, 3.0e-5 (9.9e-5), 3.9e-5; mode 3 thin and thick identical to mode 2 (the charge does not depend on the mode).
2. NVE with the constant field (600-1000 steps, 0.5 fs, `velocity 300`, frames every 20 steps, h of every frame from a twin static run): span (max-min) of K + U - sqrt(eps) E.P over 1000 steps: thin 4.7e-4 eV, thick 3.1e-4 eV (drift of a linear fit -3.4e-6 / 1.7e-5), compared with the field-off run 9.4e-4 / 2.5e-4 and with K + U alone in the field: span 1.20 / 0.28 eV (drift -2.9e-2 / -3.7e-2). So the energy that is conserved in the field is K + U - sqrt(eps) E.P_direct, with the same quality as field off. (Old model with field: not compared, its BEC has no P_direct.)
3. Closed loop (24-32 points, trapezoid on the periodic integrand, random loop of amplitude 0.15 A in all atoms): |loop integral| / scale of the integrand, ct: thin 1.4e-5 / 2.6e-4 / 8.3e-4, thick 2.2e-4 / 2.7e-4 / 2.8e-4 (the three E directions); OLD qNEP BEC: thin 0.50 / 0.19 / 0.16, thick 0.32 / 0.04 / 0.27.

### B. charge_mode 3 and triclinic cells (accepted, verified; no parser change needed)
mode 3 header `nep4_charge3` (generator `make_model_text(..., charge_mode=3)`, same weights): force RMS error 1.2e-4 (thin, all 12 atoms) / 9.0e-5 (n45 rc 3.5, 12 atoms) vs OLD mode 3 9.5e-4 / 1.7e-3; virial RMS error 4.6e-3 (rms_W 7.5) / 1.8e-3 (3.4) vs OLD 2.7e-3 / 1.5e-3 (CT/OLD 1.7 and 1.2, within the 3x gate).
triclinic (lattice rows 1 0 0 / 0.12 1 0 / -0.08 0.10 1 times L, same fractional coordinates; small-box thin n12 and large-box n45 rc 3.5): force error 1.1e-4 / 6.4e-5 vs OLD 1.4e-3 / 1.7e-3; virial error 3.6e-3 (rms_W 7.1) / 4.6e-4 (8.4) vs OLD 2.2e-2 / 6.5e-3; BEC (field force test above) 3.2e-4 / 3.9e-5. Strain FD uses all six components with the lattice rows deformed.
pppm: n45 rc 3.5, pytest force/virial FD (and the earlier n45 rc 4 run).

### C. Numerical items
- `gpumd_replica`: `src/main_replica/force/nep_charge.cu` now stops with "charge_transfer is not supported by gpumd_replica yet (DR-005 stage 3)." (pytest `test_gpumd_replica_rejects_charge_transfer`).
- Force FD test now uses all 12 atoms of the thin n12 case, plus a 3-type case (nt = 3, 6 phi pairs), mode 3, triclinic and pppm cases (pytest `test_extended_force_and_virial_fd`).
- Pair-virial-only mutation (0.5 -> 1.0 in `ct_pair_force`, forces untouched): 8 of the 8 virial tests fail (the two basic and six extended cases); force tests unaffected.
- Static energy (`scripts/ct_checks.py energy`): `kspace ewald` for qNEP mode 2 adds only the reciprocal-space part (alpha = pi/rc_radial, |k|^2 <= (2 pi alpha)^2, src/force/ewald.cu). E(ct) - E(same ct model with all phi = 0) = E_el[q] exactly (q = 0 there, so the network energy cancels); compared with the float64 host sum K_C (2 pi/V) sum_k exp(-k^2/4a^2)/k^2 |S|^2 of gpumd's dumped charges: |dE| = 5e-7 eV (thin n12, E_el 1.78), 3.4e-8 (n45 rc 3.5, 0.73), 1.4e-11 (n8, 6.8e-5); the no-line twin (old path) 6e-6, 1e-5, 3e-8 eV. max |q_gpumd - q_host| <= 2.1e-7. What this checks: E_el of the reported energy corresponds to the reported charges and q equals the host formula; the network energy E_NN is common to ct and non-ct models and is not independently recomputed. (The VER-0033 numpy reference uses a toy K and cannot ingest nep.txt weights.)
- Large N (`ct_checks.py big`, N = 3000, L = 59, large-box, rc 4/4): |sum q| = 1.5e-7 e (limit 1e-5; float sum of |q| up to 1.2), two identical static runs: q bit-identical, max |dZ| = 1.2e-7 (atomicAdd order), max |sum_k Z_k| = 1.3e-6 (translation invariance of P; rms Z 0.154).
- Float neutrality limit: sum q is limited by the float representation of q (about 1e-7 sqrt(N) |q|_typ); gates assume |q| <~ 1.
- DR-005 stage 3 (not done): nep trainer kernels (`src/main_nep`), `main_replica` force kernels (rejected with the message above), trained weights, charge_vdw / spin models (rejected by the new-format reader).

## Commands
```
cd src && make CUDA_ARCH="-arch=sm_86" -j4
PY=/home/katouda/miniforge3/envs/py312-gpumd/bin/python; T=specs/verification/VER-0034-qnep-charge-transfer-forces-gpu/scripts/ct_forces_fd.py
$PY $T src/gpumd /scratch/w --structure thin_n12_L7.5 [--old] [--kspace pppm] [--rc-radial 5.5 --rc-phi 4] [--max-atoms 8]   # force and virial FD
$PY $T src/gpumd /scratch/w --structure thick_n45_L14 --nve 2000 [--old]
$PY -m pytest -q tests_pytest/test_qnep_charge_transfer_forces.py tests_pytest/test_qnep_charge_transfer_parse.py tests_pytest/test_qnep_charge_transfer_bec.py
```

## Not verified
Trained weights; nep trainer and main_replica kernels (stage 3); float vs double kernels; pppm only on one structure; charge_mode 1 with charge_transfer (rejected by the parser); NVE with realistic dynamics (the random models are unphysical); `run_regression.py --suite full` (not run).

## Decision of the human (2026-11-11): virial gate
The virial gate is: RMS error <= 3x the measured finite-difference floor of an old qNEP model on the same structure AND relative error <= 3e-3 |W| (measured 5e-5 to 5.7e-4). The absolute 1e-3 (which suits forces in eV/A) is not applied to the virial (eV, magnitude 4-13; the old-model floor itself reaches 8.3e-3). Forces keep the absolute 1e-3 eV/A gate. (Triclinic cells: virial errors 3.6e-3 / 4.6e-4 eV against old-model floors 2.2e-2 / 6.5e-3 eV.)
