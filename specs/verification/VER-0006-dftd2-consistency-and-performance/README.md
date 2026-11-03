# VER-0006: DFT-D2 consistency (finite differences), NVE, and cost (DES-0002 Phase 5a; DR-002 AC-4, AC-7)

- Date: 2026-11-03. Status: verification (self-consistency of the GPU code, timing). NOT validation.
- Binary: `src/gpumd` built from worktree HEAD 27ca8d72 (`make CUDA_ARCH="-arch=sm_86" -j4`), RTX A6000, CUDA 12.3 from /usr/local/cuda. Model: NEP89 (`potentials/nep/nep89_20250409/nep89_20250409.txt`).
- Section 1 was fixed BEFORE the first GPU measurement and has not been changed.

## 1. Pre-registered criteria
Quantity: D2 part = (run with `dftd2`) - (identical run without), so the NEP part cancels.
Derivatives are 5-point central differences of the double-precision header energy of `dump_xyz ... precision double`.
Float noise: the pair terms are single precision (relative ~6e-8 per pair, not smooth in the positions), so the
D2 energy has a noise floor of ~1e-7 relative; a finite difference divides it by h. To lift the signal,
s6 = 75 (100 x the usual value; the code is linear in s6) with rc = 12 A (tests: s6 = 75, rc = 12).
Steps chosen a priori from the noise/truncation balance (damping length ~ R0/20 ~ 0.1 A): h = 0.01 A for positions and
1e-3 for the deformation x -> (1 + eps) x. A scan of h is reported but the gate is evaluated at the nominal step only.

| ID | Criterion |
|---|---|
| G1 | max abs(F_fd - F_gpu) <= 2e-4 * max abs(F_D2) at h = 0.01 A (3 atoms x 3 components per system: O, H, and O from the cell middle) |
| G2 | max abs(W_fd - W_gpu) <= 2e-4 * max abs(W_D2) at h = 1e-3, components xx, yy, zz, xy, yz, xz; W = -dE/d(eps) (isotropic part from xx + yy + zz) |
| N1 | NVE: total energy drift and std with D2 not worse than without D2 (same start; 648 water, dt 0.5 fs, 10 ps, default rc = 30 A) |
AC-7 (D2 <= 0.5 x D3 time) is informative, not a gate.
Systems: water27 (81 atoms, 9.3 A cube, rc = 12 A > box so images are exercised), the same cell with pbc F F F, with pbc T T F, and the triclinic `water27_tri`.

## 2. Finite-difference results (`results/fd_s6_75_rc12.txt`, `scripts/fd_check.py`)
The on-minus-off difference is bit-reproducible (repeat of the same configuration: dE = dF = dW = 0).
Error at the nominal step (relative to max abs of the D2 quantity):

| system | max abs F_D2 [eV/A] | force err [eV/A] (rel) | max abs W_D2 [eV] | virial err [eV] (rel) | G1 | G2 |
|---|---|---|---|---|---|---|
| cubic, periodic | 3.694 | 8.0e-5 (2.2e-5) | 401.6 | 3.1e-3 (7.6e-6) | ok | ok |
| triclinic, periodic | 3.365 | 8.3e-5 (2.5e-5) | 462.1 | 2.3e-3 (5.0e-6) | ok | ok |
| open (non-periodic) | 3.355 | 6.2e-5 (1.9e-5) | 222.8 | 2.3e-3 (1.0e-5) | ok | ok |
| slab (T T F) | 3.642 | 7.1e-5 (2.0e-5) | 368.0 | 2.7e-3 (7.3e-6) | ok | ok |
Strain deformations tested: all six components xx, yy, zz, xy, yz, xz (shear and isotropic parts). Over the whole h scan (force 0.0025-0.04 A, strain 2.5e-4 to 4e-3) the relative errors are 1.2e-5 to 3.9e-4 (force; the largest at h = 0.04 and 0.0025, truncation and noise respectively) and 3.8e-6 to 3.8e-5 (virial). Gates G1, G2 are met on all four systems.

## 3. Tests (`tests_pytest/test_dftd2_consistency.py`, marked `fast`, 127 s for the four tests)
- force vs FD on cubic and open systems, virial vs FD on cubic (xx, yz) and triclinic (yy, xy), same tolerances as G1, G2.
- Mutation sanity (scratch copies of `src/` outside the repository, `DFTD2_TEST_GPUMD` selects the binary; `results/mutation_checks.txt`):
  force sign flipped in `dftd2_pair_kernel` (`fx -= fpx` etc.): both force tests fail (errors 5.1 and 6.6 eV/A against 7e-4), virial tests pass;
  virial sign flipped (`v_xx += 0.5 ...`): both virial tests fail (errors 8.0e2 and 8.8e2 eV against 0.08-0.09), force tests pass.

## 4. NVE (`scripts/nve.py`, `results/nve.txt`)
648 water atoms (water27 repeated 2 x 2 x 2, 18.6 A), NEP89, equilibrated 10 ps NVT-BDP 300 K without D2, then 20000 steps NVE, dt 0.5 fs (10 ps), `velocity 300 seed 20261103`, identical start, thermo every 10 steps.

| run | E_tot std [eV] | max - min [eV] | linear drift [meV/ps] | drift [meV/atom/ps] | mean T [K] |
|---|---|---|---|---|---|
| NEP89 only | 0.0886 | 0.339 | +29.8 | +0.0460 | 302.2 |
| NEP89 + dftd2 0.75 (rc = 30) | 0.0324 | 0.174 | +9.6 | +0.0148 | 298.5 |
N1 met (not worse). One trajectory per run, no statistics: the difference between the two runs is of the size of the trajectory-to-trajectory scatter of float32 NEP noise in a chaotic system and must not be read as D2 improving conservation (the shifted-force D2 itself is smooth and continuous at rc).

## 5. Cost (slope method, `scripts/timing.py`, `results/timing.txt`, `results/timing.json`)
Water, NVE dt 0.5 fs, per-step time = (T(n2) - T(n1)) / (n2 - n1) of the whole process, min of 2 runs per point (n1 = N_steps/5; 100/500 steps for small cases; fewer for the expensive ones so that the long run is about 20 s). D3 = `dftd3 pbe 12 6` (the documented example; `dftd3` has no default cutoffs). D2: `dftd2 0.75 12` and the default rc = 30 A. "Extra" = (run with the term) - (run without).

| N | NEP only [ms] | D3 extra [ms] | D2 rc12 extra [ms] | D2 rc30 extra [ms] | D2(12)/D3 | D2(30)/D3 |
|---|---|---|---|---|---|---|
| 81 | 1.42 | 1.35 | 4.27 | 30.6 | 3.2 | 22.6 |
| 648 | 1.54 | 2.00 | 3.04 | 54.3 | 1.5 | 27.1 |
| 5184 | 1.79 | 1.03 | 3.15 | 55.9 | 3.0 | 54.1 |
| 41472 | 6.73 | 7.52 | 12.03 | 211.4 | 1.6 | 28.1 |

- The AC-7 target (D2 <= 0.5 x D3) is NOT met at either cutoff: D2 is 1.5-3 x D3 at rc = 12 A and 23-54 x D3 at the default rc = 30 A. (Informative, not a gate.)
- Cost does not scale linearly with N for N <= 5184 at rc = 30: 648 and 5184 atoms cost the same (54 and 56 ms), 41472 atoms 211 ms. Presumably the one-thread-per-atom kernel with a long serial neighbor loop does not fill the GPU for small N (not investigated). The 81-atom cell (smaller than rc) costs 31 ms.
- N = 648, extra time vs rc: 16 A 7.8 ms, 18 A 9.1 ms, 20 A 38.9 ms (VER-0005: 8.1 / - / 39.0 ms for 16 / 20 A). The jump between 18 and 20 A reproduces; its cause was not investigated.
- D3 numbers use the current optimized D3 (VER-0004); the D3 extra for N = 5184 (1.0 ms) is below that at 648 atoms (2.0 ms), that is, the slope-method noise on these small differences is ~0.5-1 ms.

## 6. Regression case and example
- `tests_regression/manifest.json`: `dftd2_basic` (NEP89 + `dftd2 0.75 12`, 5 NVE steps, fixture `nep89_carbon`) and `dftd2_dftd3_exclusive` (expected failure, stderr contains "dftd2 and dftd3 cannot be used together"); inputs in `tests_regression/cases/dftd2/`.
  These are valid only when the baseline binary also implements `dftd2`: a `master` baseline that predates the feature reports an unknown keyword. Checked with baseline = candidate = this binary, `--suite full --case 'dftd2_*'`, three times: 2 passed each time (byte-exact restart.xyz, thermo.out and neighbor.out, no tolerance used).
- `examples/gpumd_dftd2/` (run.in, model.xyz, README): ran in a scratch copy (potential path made absolute there); 100 steps finished.

## 7. Not verified / limitations
- FD checks use rc = 12 A and s6 = 75 only; rc = 30 A and s6 = 0.75 are covered by AC-1 (comparison with the NumPy reference), not by FD on the GPU.
- Float noise sets the FD floor (relative ~1e-5); the tolerances 2e-4 are therefore ~10 x looser than the AC-1 values and test consistency only, not accuracy.
- NVE: a single trajectory per case; no statistics. No GPU other than RTX A6000 sm_86.

## Addendum (2026-11-03): velocity seed in the NVE protocol
VER-0011 found that when `model.xyz` carries a velocity column (as `restart.xyz` from the NVT equilibration does),
`gpumd` ignores `velocity ... seed` (`Velocity::initialize_cpu`, `src/main_gpumd/velocity.cu`; not investigated further).
The NVE runs of this record (and of VER-0010) restart from such a file, so the seed did not change the initial
velocities: each system was one trajectory from one initial condition (the "identical start" above), not an ensemble
over seeds. The conclusions are unchanged in kind (a single-trajectory comparison, which was already stated as
"no evidence of improved conservation"), but the NVE comparison has no statistics across initial velocities.
