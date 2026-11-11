# VER-0033: float64 numpy reference of forces and virial of the charge_transfer qNEP (DR-005 stage 2c, step 1)

- Status: reference done 2026-11-11; gate (analytic vs FD <= 1e-8 absolute, forces and virial, float64) met in all 5 gated cases with a 5-point 4th-order FD stencil (the 2nd-order delta = 1e-5 FD missed it for the virial in 3 cases by truncation). No `src/` change, no GPU. Type: analytic / finite difference (verification). Synthetic data (VER-0029 random weights and structures).
- Requirement DR-005; design DES-0005 section 3 ("Energy and forces in this mode"), VER-0029 (model code imported unchanged), VER-0032 (stage 2b kernels). Python 3 + numpy + scipy (erfc), CPU only.

## Model and conventions
E(r, H) = E_el(q(r,H), r, H), q_a = sum_b phi_ab (h_a - h_b) (VER-0029). The NEP side needs only D_a = dE_el/dq_a and the explicit pair force of the solver at fixed q.
Toy solver: E_el = 0.5 sum_{a,b,s} q_a q_b K(|r_a - r_b - s|) + 0.5 s0 q^2, K = erfc(r/1.5)/r * 0.5(cos(pi r/rcK)+1), rcK = min(4, 0.49 L), s0 = 0.3, so D_a = sum_{b,s} q_b K + s0 q_a.
The NEP-side formulas below are independent of K (only D_a and the pair weight g^K_ab = q_a q_b K'(r) enter).
Notation: neighbour vector v = r_b + s - r_a (image b,s), d = |v|, h-difference weights below; forces F = -dE/dr.
Virial: W_ab = sum_a r_a,a F_a,b (periodic: sum over pairs of (r_a - r_b) (x) F_a) = -dE/d eps_(ba) for r -> (1+eps) r, H -> H (1+eps)^T. This is the GPUMD `virial_per_atom` convention
(index 3 = xy = r_x F_y; the existing kernels add r12 (x) f21 with r12 = r2 - r1 per ordered pair, f21 = minus the force on atom 1 from atom 2's descriptor term).

## Derived formulas (all a-centred: atom a only needs its own rc_desc and rc_phi loops; checked, including the descriptor part)
1. Dt_c = dE/dh_c = sum_b phi_cb (D_c - D_b) (rc_phi loop of c). Derivation: sum_a D_a q_a = sum_unordered phi_ab (h_a - h_b)(D_a - D_b) with D fixed.
2. Descriptor part: F_k -= sum_a Dt_a dh_a/dr_k. Identical to the existing electrostatic force path with weight Dt: for a and each b in its rc_desc list, G_ab = dh_a/dr_b (VER-0029 Term A kernel, radial + angular), F_b -= Dt_a G_ab, F_a += Dt_a sum_b G_ab. No two-hop loop.
3. Pair term (from phi(r_ab) at fixed h, D): for each ordered pair in atom k's rc_phi loop, g_kb = phi'(d)(h_k - h_b)(D_k - D_b),
   F_k += g_kb v/d  (v = r_b - r_k; equivalently -g x_hat with x_hat = (r_k - r_b)/d). Newton's third law holds because g_kb = g_bk and v flips; each atom only adds to itself.
4. Fixed-q explicit force of the solver, same form with g^K_kb = q_k q_b K'(d) (in GPUMD this is already inside the ewald/pppm/real-space force; shown here only to make the total FD-testable).
5. Virial (total): W_ab = -sum_{a,b in desc lists} v_a Dt_a G_ab,b  -  0.5 sum_{ordered pairs} (g + g^K) v_a v_b / d. (descriptor part: sum_b (r_b - r_a) (x) F_b from the descriptor force; pair parts symmetric, 0.5 per ordered pair.) The solver's own k-space virial at fixed q is unchanged.
6. Old weight (`zero_mean_D_real`: D_c - mean(D), no pair term) is NOT the gradient for this model (below).

## Kernel-level recipe for stage 2c (from reading `nep_charge.cu` compute_large_box / compute_small_box, read-only)
- Existing path: after `ewald/pppm.find_force` (writes `nep_data.D_real`, plus the real-space/self parts for charge_mode 1/3), `zero_mean_D_real<<<1,1024>>>` is applied in place, then `find_force_radial` (weight `g_Fp + g_charge_derivative * g_D_real[n1]`
  (own) and `[n2]` (neighbour), writes f12 - f21 to g_f and r12 (x) f21 (9 comps, index map 0 3 4 / 6 1 5 / 7 8 2) to g_virial), `find_partial_force_angular` (Fp += charge_derivative * D_real, then `find_properties_many_body`, which does forces and virial).
  The same weight array `nep_data.D_real` is used by all three; the ewald/pppm explicit force and virial stay as they are.
- charge_transfer: replace `zero_mean_D_real` by a kernel writing Dt_c = sum_b phi_cb (D_c - D_b) in place of D_real (needs a copy of D, since D_b of neighbours is read: write into a second array, e.g. new `ct_Dt`, and pass it as `g_D_real` to the radial and angular kernels;
  no other change to those kernels, `g_charge_derivative` must be dh/d(descriptor) of the charge output used as h, as in the VER-0032 BEC kernels). D has to be final (after `find_charge_self_energy` / real-space D for mode 1/3) before Dt is formed.
- New pair kernel (own rc_phi loop over the radial list, same ct_find_phi): F_k += phi' (h_k - h_b)(D_k - D_b) v/d, virial 9 components -0.5 phi'(h_k - h_b)(D_k - D_b) v (x) v/d added to g_virial[k + i N]; needs h (`nep_data.ct_h`) and D (final). Large-box: r12 by MIC from positions; small-box: stored r12. No other new virial pieces.
- Accumulation: sums per atom in float as now are within the existing noise; advice for new kernels: accumulate Dt and the pair sums in double (cancellations D_c - D_b, h_a - h_b), store results as before (forces/virial are double arrays already).
- Edge check [CHECK on GPU]: rc_phi <= rc_radial so the radial list contains all phi neighbours (enforced by the parser); Dt of an atom needs D of its neighbours (read-only, no two-hop positions).

## Results (`python scripts/ct_forces_reference.py`, ~60 s; gate FD = 5-point 4th-order central stencil (-f(2d)+8f(d)-8f(-d)+f(-2d))/(12 d), d = 5e-5 (positions in A, strain eps dimensionless), float64; all 9 eps components)
| case (N, L, rc_desc, rc_phi) | F err (4th) | W err (4th) | F / W err (2nd order, d=1e-5, diagnostic) | max abs sum F | RMS F | RMS W | RMS(F_old - F_FD) | RMS(F with Dt, no pair term - F_FD) | gate |
|---|---|---|---|---|---|---|---|---|---|
| 8, 10, 4, 4 | 1.3e-12 | 2.0e-12 | 1.5e-11 / 1.3e-10 | 3e-17 | 0.034 | 0.15 | 0.043 | 0.038 | PASS |
| 45, 12, 4.5, 4 | 6.3e-10 | 5.2e-10 | 2.8e-9 / 1.4e-8 | 5e-15 | 2.44 | 29.6 | 1.92 | 0.94 | PASS |
| 59, 14, 5, 5 | 9.1e-10 | 1.9e-9 | 4.5e-9 / 5.9e-8 | 3e-15 | 1.56 | 57.8 | 1.27 | 0.91 | PASS |
| 12, 7.5, 3.6, 3.7 (thin, rc < L/2) | 1.3e-10 | 1.7e-10 | 1.1e-9 / 1.5e-8 | 1e-15 | 2.73 | 20.3 | 2.13 | 1.85 | PASS |
| 59, 22, 3.5, 3.5 (rc_d + rc_p < L/2) | 5.5e-12 | 1.7e-11 | 4.5e-11 / 3.7e-10 | 3e-17 | 0.036 | 0.33 | 0.016 | 0.015 | PASS |
| 10, 6, 4, 4.5 (rc > L/2, images; info) | 1.8e-10 | 6.1e-10 | 4.5e-9 / 2.8e-8 | 2e-15 | 6.38 | 18.7 | 5.27 | 0.85 | info |
Gate (unchanged): analytic vs FD <= 1e-8 absolute, forces and virial, float64: PASS in all five gated cases (max force 9.1e-10, max virial 1.9e-9). The 2nd-order numbers (delta = 1e-5) are kept as the earlier diagnostic: its virial residual was 1.4e-8, 5.9e-8, 1.5e-8 in the three dense cases (O(delta^2) truncation at |W| ~ 20-60), which is why the 4th-order stencil is used for the gate.
Invariances (20-atom cluster, L = 60, rc 4/4): sum F 5e-16, rotation (F -> RF, W -> R W R^T) 5.7e-14, translation 6.8e-14, image shift 2.8e-14, atom permutation 3.6e-15 (all <= 1e-12); W symmetric to 2e-15 in all cases; |sum q| <= 1e-14. FD on the cluster (4th order): force 7.3e-11, virial 2.3e-10.
Item 4 (old weight): RMS(F_old - F_FD) is 0.04 to 5.3, comparable to RMS F: the `zero_mean_D_real` weight without the pair term does not reproduce the FD force. Dt without the pair term (not the old weight) is also wrong (0.015 to 1.85), so both Dt and the pair term are required.
"RMS W" for the thick cases reflects strongly charged random models (|q| up to a few e); W parts: descriptor ~10, ct pair ~20-40, K ~1-3.

## Finite-difference step scan (`scripts/delta_scan.py`; max abs error vs analytic, force / virial)
- 2nd order (N59 case): d = 2e-5, 1e-5, 5e-6, 2.5e-6: virial 2.3e-7, 5.9e-8, 1.5e-8, 5.4e-9 (falls 4x per halving = O(d^2)); force 2.0e-8, 4.5e-9, 6.0e-9, 1.5e-8 (round-off onset below 5e-6).
- 4th order (N59): d = 1e-2 ... 5e-5: virial 1.3e-2, 3.2e-3, 1.6e-3, 5.6e-4, 1.8e-4, 1.3e-5, 2.0e-5, 1.9e-9; force 5.7e-4, 2.4e-4, 9.1e-5, 9.2e-6, 7.5e-6, 1.4e-7, 4.3e-10, 9.1e-10.
  (N12): virial 8.6e-4, 1.8e-3, 4.7e-4, 3.1e-5, 1.4e-5, 1.1e-6, 1.6e-10, 1.7e-10; force 1.8e-4, 3.4e-5, 2.4e-6, 3.5e-6, 1.1e-11, 3.4e-11, 7.9e-11, 1.3e-10.
- Honest reading: a clean delta^4 law is NOT observed. The model energy is only C1 at the cutoffs (fc'' jumps at rc_desc, rc_phi, rcK), so a stencil that straddles a pair crossing a cutoff has an error governed by the jump, not by delta^4; this gives the large, non-monotonic errors at d >= 2e-4 (a few pairs lie within 2 d (1+rc) of a cutoff).
  Below d ~ 1e-4 no crossing remains and the residual is small and flat (1e-10 .. 2e-9: round-off floor ~ 1e-16 |E| / d with |E| ~ 10-100, plus remaining truncation). d = 5e-5 was chosen from this scan (a posteriori; it is the largest step in the scan that is below the crossing regime for all cases), so the choice is scan-informed, not blind.
  `ct_forces_reference.py` prints the minimum |r - rc| over all pairs and cutoffs ("cutoff gap") per case; for the N59 case it (2.6e-4) is below the conservative stencil reach (5e-4, = 2 d max(1, rc)) although the observed error is 1.9e-9, i.e. no actual crossing inside this stencil (the reach is an upper bound). The analytic force/virial are exact for configurations away from cutoff crossings, as for all existing NEP kernels.

## Not verified
GPU kernels (stage 2c itself), float accumulation, real Ewald/PPPM D (only a toy K), charge_mode 1/3 details (self energy / real-space D enter only through D), small-box mixed images beyond the informational case, NVE energy conservation, trained weights, a pure delta^4 convergence law (limited by the C1 cutoffs, see above).
