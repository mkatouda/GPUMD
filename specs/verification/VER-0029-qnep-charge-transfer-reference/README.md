# VER-0029: float64 numpy reference of the charge-transfer qNEP and its exact BEC (DR-005 stage 2a)

- Status: stage 2a done (2026-11-11). No `src/` change, no GPU. Verification type: analytic / finite difference (not regression, not validation).
- Requirement: DR-005; design DES-0005 sections 2, 3, 6; ADR-0006. Data tier: synthetic (random float64 weights, random structures).
- Environment: numpy 2.5.4 (conda env `py312-gpumd`), CPU only. Scratch runs only; nothing generated in the repo.

## Model (as implemented, `scripts/ct_reference.py`)
- Cubic periodic box L. Neighbours are (atom, lattice image) pairs; for rc < L/2 this is the minimum image convention (the gated cases), rc > L/2 works too (informational case).
- 2 atom types. Basis f_k(r) fc(r), k < 4 (NEP form 0.5(T_k(2(r/rc-1)^2-1)+1), fc = 0.5(cos(pi r/rc)+1)).
- Descriptor (5 components): 3 radial sums_b g_n(t_a,t_b,r_ab) and 2 angular-like invariants |sum_b g_m(r_ab) r_ab|^2 (the sum_fxyz path); cutoff rc_desc; coefficients symmetric in the type pair.
- h_a = w1[t_a] . tanh(W0[t_a] d_a + b0[t_a]) - b1[t_a] (10 hidden nodes, per type, random weights).
- phi_{tt'}(r) = sum_k c_k^{tt'} f_k(r) fc(r), symmetric in the types, cutoff rc_phi (coefficients in [0,1]).
- q_a = sum_b phi_ab (h_a - h_b); P = sum_a q_a r_a = 0.5 sum_(a,b,s) phi_ab (h_a-h_b)(r_a - r_b - s) = sum_a h_a V_a, V_a = sum_b phi_ab (r_a - r_b).
- Convention: Z[k,beta,alpha] = dP_beta/dr_k,alpha.

## Analytic a-centred BEC (what the GPU kernels will do; `bec_analytic`)
Z_k = Term A + Term B.
- Term A = sum_a V_a,beta dh_a/dr_k,alpha. Loop over a and its rc_desc neighbours b: with G_ab = dh_a/dr_b (chain rule through the network, radial part
  sum_n (dh/dd_n) g_n'(r) u and angular part 2 sum_m (dh/dd_m)[g_m' u (S_m . x) + g_m S_m], x = r_b - r_a, S_m = sum g_m x), add outer(V_a, G_ab) to Z_b and subtract it from Z_a.
  This is the existing force-like kernel with weight V_a,beta, three passes.
- Term B = sum_b M_kb (h_k - h_b), M_kb = phi'(r) xhat (x) x + phi(r) 1, x = r_k - r_b, loop over k and its rc_phi neighbours. The diagonal q_k 1 is the phi*1 part of this term
  (sum_b phi (h_k - h_b) 1 = q_k 1); it does NOT have to be added separately (adding it would double count).
- Derivation of B: dV_a/dr_a = sum_b M_ab, dV_a/dr_b = -M_ab, M symmetric, so sum_a h_a dV_a/dr_k = sum_b M_kb (h_k - h_b). The self-image terms (b = a, s != 0) cancel (h_a - h_a = 0).
- Needs nothing beyond DES-0005 section 3: the phi type-pair symmetry gives no extra derivative (phi depends on the types, not on positions); no mean-subtraction/neutralization term;
  the only addition to the DES text is the explicit statement that the q_k 1 diagonal is contained in Term B.

## Results (`python ct_reference.py`, about 25 s; FD: central, delta = 1e-5, float64)
| case (N, L, rc_desc, rc_phi) | max abs sum q | max abs (Z_an - Z_FD) | max abs sum_k Z | RMS Z | RMS(H0 - Z) | RMS(T2-only - Z) | gate |
|---|---|---|---|---|---|---|---|
| 8, 10, 4, 4 | 6e-17 | 1.3e-10 | 2e-16 | 0.236 | 0.113 | 0.136 | PASS |
| 45, 12, 4.5, 4 | 2e-15 | 1.3e-09 | 3e-15 | 0.899 | 0.847 | 0.100 | PASS |
| 59, 14, 5, 5 | 2e-15 | 2.3e-09 | 4e-15 | 1.676 | 1.211 | 0.175 | PASS |
| 12, 7.5, 3.6, 3.7 (thin, rc < L/2) | 9e-16 | 1.5e-09 | 1e-15 | 2.140 | 0.922 | 0.911 | PASS |
| 59, 22, 3.5, 3.5 (rc_d + rc_p < L/2) | 0 | 2.0e-10 | 6e-16 | 0.190 | 0.029 | 2.2e-11 | PASS |
| 10, 6, 4, 4.5 (rc > L/2, images) | 4e-16 | 1.3e-09 | 2e-15 | 2.415 | 3.405 | 3.40 | info |

G2 (20-atom cluster, L = 60, rc = 4/4): sum_k Z 7e-16; rotation covariance Z -> R Z R^T 9.5e-15; shift of one atom by L 1.1e-14; atom permutation 8e-16; analytic vs FD 7e-10 (limits 1e-10 / 1e-9 each / 1e-8): PASS.
The FD residual (1e-10 to 2e-9) is the O(delta^2) truncation error of the central difference, not an analytic discrepancy; the gate is 1e-8.

## Findings for the kernels
- Z is not symmetric (RMS(Z - Z^T) 0.05 to 1.5): the BEC array must store the full 3x3 with the beta/alpha convention fixed (index [3 beta + alpha] = dP_beta/dr_alpha).
- G3 (informational): the old kernel formula (qn 1 + 0.5 T1 + 0.5 T2, raw Jacobian, MIC) is NOT the gradient of P for this model (RMS error 0.03 to 1.2 e, comparable to Z itself even though sum q = 0 and qn = q).
  The T2-only form q_k 1 + sum_a (r_a - r_k)_MIC (x) dq_a/dr_k equals the exact Z (2e-11) only when the full dependence range rc_desc + rc_phi of q_a is below L/2 (case 59/22);
  with rc_desc + rc_phi > L/2 (all other cases) the MIC of the reference-k form wraps and it fails (0.1 to 0.9 e). It is therefore not a usable kernel form for boxes of ordinary size.
  The a-centred decomposition A + B has no such condition. This corrects the wording of DES-0005 section 1(ii): "T2-only" is a gradient only for the two-hop-safe case; the kernels should implement A + B (DES section 3), as planned.
- Validity rule for the kernels: each a-centred sum needs only distances within rc_desc (Term A: dh_a/dr_b, b within rc_desc of a) or rc_phi (V_a, Term B: pairs within rc_phi of a/k); no distance beyond rc_phi or rc_desc and no two-hop MIC.
  So the same box checks as the present kernels apply separately to rc_desc and rc_phi (minimum image unique for rc < L/2, else the small-box image path, shown to work in the informational case);
  rc_phi + rc_desc < L/2 is NOT required by A + B. The charges q_a themselves (energy) need h_b of the neighbours, each computed with its own rc_desc neighbourhood: also no extra condition.
  (The statement in DES section 2 that q_a depends on atoms within rc_phi + rc_desc is correct physically, but irrelevant for the kernels.)

## Command
```
/home/katouda/miniforge3/envs/py312-gpumd/bin/python specs/verification/VER-0029-qnep-charge-transfer-reference/scripts/ct_reference.py   # add --quick for 2 cases
```

## Not verified
GPU kernels (stage 2b), float vs double, forces/virial and the D-weight of DES section 3 (stage 2c), realistic trained weights, charge_mode 2 vs 3 network details, mixed images with rc_phi > L/2 in the real small-box path, training-side kernels.
