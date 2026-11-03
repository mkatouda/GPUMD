# VER-0005: DFT-D2 cutoff study (DES-0002 Phase 2, resolves DR-002 OQ-4)

- Date: 2026-11-03. Status: verification (convergence of the truncated sum with rc). NOT validation.
- Section 1 (criteria) was written BEFORE any measurement and is not changed afterwards.
  Results are in section 3 onwards (added after measurement).

## 1. Pre-registered criteria (fixed before measuring)

Quantity: the D2 contribution only, i.e. (NEP89 + dftd2 s6 rc) minus (NEP89 alone), from static
`gpumd` runs (`time_step 0`, 1 step, double-precision dump) on identical structures.
Shifted-force truncation exactly as implemented (`src/force/dftd2.cu`), s6 = 0.75.
Reference: rc = 40 A (same implementation). Compared radii: rc in {12, 16, 20, 25, 30} A.
Differences are D2(rc) - D2(40 A).

| ID | Criterion (per system, per rc) |
|---|---|
| T1 | abs(dE)/N <= 0.5 meV/atom |
| T2 | max abs(dF) per Cartesian component <= 5 meV/A AND RMS(dF) <= 1 meV/A |
| T3 | abs(dP) <= 0.02 GPa, P from the virial (isotropic: trace/3; the three diagonal components are also tabulated) |

- Systems: (a) liquid water, 648 atoms, equilibrated with NEP89 at 300 K (box about 18.6 A);
  (b) graphite, hexagonal AB supercell, >= 500 atoms, >= 2 layers; (c) MgO rock-salt supercell,
  >= 500 atoms. Structures built with ASE (no invented parameters); lattice constants are
  literature-typical values listed in `scripts/make_structures.py`.
- Proposed default rc = the SMALLEST rc in the grid that satisfies T1, T2 and T3 on ALL systems.
  If none of {12,...,30} satisfies them, the study reports that and no default is proposed from it.
- Cross-check: GPU code vs `tests_pytest/dftd2_reference.py` (NumPy) at rc = 12 and 25 on water
  (energy, forces, virial; reported, not part of the default selection).
- Cost: D2 time per step vs rc on water 648 (run-time difference of D2-on and D2-off runs, a few
  hundred steps), and the number of neighbors per atom within rc.
- NVE check (separate): water 648, dt 0.5 fs, 2 ps, fixed seed, rc = 12, 25 and the chosen default;
  report energy drift and std. Only "not worse than rc = 12 / 25" is asked; no other pass/fail.
- Thresholds are not changed after seeing data. If they turn out unsuitable (too strict or too
  loose), the study is completed with them and the opinion is stated in section 5.
- The tail estimate (uniform-density continuum, plain tail beyond rc) is reported next to the
  measured numbers; it is an estimate, not a criterion.

## 2. Deviation recorded after the first measurement (thresholds NOT changed)
The first run used ideal graphite and MgO lattices: all forces vanish by symmetry, so T2 is
empty for them (`results/static_cutoff_ideal_lattices.txt`; energy and pressure columns of that
run are valid and identical to the table below within 1e-3 meV/atom and 1e-3 GPa, forces are 0).
Rattled structures (Gaussian, sigma = 0.05 A, fixed seeds 7 and 8, `scripts/make_structures.py`) were
then added and are the ones used in section 3. T1-T3 are unchanged. The default rc is the same
for both versions.

## 3. Setup and results (measured 2026-11-03)
- Binary: `src/gpumd` built from worktree HEAD 03f6d6bc, `make CUDA_ARCH="-arch=sm_86"`, RTX A6000, CUDA from /usr/local/cuda.
- Model: `potentials/nep/nep89_20250409/nep89_20250409.txt` (NEP89; H, C, O, Mg all covered). s6 = 0.75.
- Structures: water 648 (216 molecules on a grid with random orientations, then NEP89 FIRE + 20000 steps at 0.25 fs + 20000 steps at 0.5 fs NVT-BDP 300 K, box 18.644 A, final frame used); graphite AB 9x9x3 (972 atoms, a = 2.462, c = 6.708 A, rattled); MgO rock salt 4x4x4 conventional (512 atoms, a = 4.212 A, rattled). Box sides are 18.6, 22.2 x 19.2 x 20.1 and 16.8 A, i.e. smaller than 2 rc: the code's image handling is exercised.
- Commands: `scripts/make_structures.py`, `scripts/cutoff_static.py ... --ref`, `scripts/cost.py`, `scripts/nve.py`, `scripts/analyze.py`; raw output in `results/`.

### 3.1 D2(rc) - D2(rc = 40) (energy per atom, force, pressure from the virial)
Absolute D2 energy at rc = 40 is -34.59 (water), -123.74 (graphite), -286.58 (MgO) meV/atom; D2 pressure at rc = 40 is -0.756, -2.53 and -5.90 GPa.
Pressure is isotropic-averaged (trace/3); the diagonal components agree to <=1e-4 GPa except graphite at 12 A (0.1161/0.1161/0.1158).

| system | rc [A] | dE/atom [meV] | max dF [meV/A] | RMS dF [meV/A] | dP [GPa] | T1 | T2 | T3 |
|---|---|---|---|---|---|---|---|---|
| water | 12 | 0.893 | 0.0055 | 0.0012 | 0.0143 | fail | ok | ok |
| water | 16 | 0.363 | 0.0005 | 0.0001 | 0.0058 | ok | ok | ok |
| water | 20 | 0.174 | 0.0001 | 0.00003 | 0.0028 | ok | ok | ok |
| water | 25 | 0.077 | 2e-5 | <1e-5 | 0.0012 | ok | ok | ok |
| water | 30 | 0.034 | 1e-5 | <1e-5 | 0.0006 | ok | ok | ok |
| graphite | 12 | 6.365 | 0.0036 | 0.0009 | 0.1160 | fail | ok | fail |
| graphite | 16 | 2.586 | 0.0003 | 0.0001 | 0.0471 | fail | ok | fail |
| graphite | 20 | 1.238 | 0.0001 | 0.00003 | 0.0226 | fail | ok | fail |
| graphite | 25 | 0.548 | 1e-5 | <1e-5 | 0.0100 | fail | ok | ok |
| graphite | 30 | 0.242 | 1e-5 | <1e-5 | 0.0044 | ok | ok | ok |
| MgO | 12 | 8.910 | 0.0066 | 0.0015 | 0.1530 | fail | ok | fail |
| MgO | 16 | 3.623 | 0.0006 | 0.0002 | 0.0622 | fail | ok | fail |
| MgO | 20 | 1.735 | 0.0002 | 0.00004 | 0.0298 | fail | ok | fail |
| MgO | 25 | 0.768 | 5e-5 | 1e-5 | 0.0132 | fail | ok | ok |
| MgO | 30 | 0.340 | 2e-5 | 1e-5 | 0.0058 | ok | ok | ok |

(Force columns are far below the 5 / 1 meV/A thresholds everywhere: the shifted-force scheme makes the force converge much faster than the energy. T2 never discriminates; T1 and T3 do. Force differences are the double-precision-dump differences; GPU float noise is ~5e-8 eV/A, below the printed precision.)

### 3.2 Result of the pre-registered selection
Only rc = 30 A satisfies T1-T3 on all three systems. **Proposed default rc = 30 A.**
The next smaller grid value, rc = 25 A, fails T1 for graphite (0.548 vs 0.5 meV, margin -10 %) and MgO (0.768 meV, -54 %); T3 is met there (graphite 0.0100, MgO 0.0132 GPa vs 0.02). rc = 30 passes with margin: T1 0.242 (graphite, 48 % of the limit) and 0.340 meV (MgO, 68 %); T3 0.0044 / 0.0058 GPa (22 % / 29 %).
Binding criterion: T1 (energy per atom) for graphite and MgO; for T3 alone rc = 25 would be enough.
It was not tested whether a value between 25 and 30 (e.g. 28) passes; the grid is the pre-registered one.

### 3.3 Tail estimate (beyond the reference rc = 40; model, not a measurement)
The differences scale as rc^-3 with a constant coefficient: A = dE(rc)/(rc^-3 - 40^-3) is 1589/1589/1590 (water), 11320/11324/11324 (graphite), 15862/15867/15869 (MgO) meV A^3/atom for rc = 20/25/30, so extrapolating to rc -> infinity (shifted-force sum -> plain sum limit) with the same law gives (`results/analysis.txt`):

| system | E(40)-E(inf) [meV/atom] | rc = 25: dE_inf / dP_inf | rc = 30: dE_inf / dP_inf |
|---|---|---|---|
| water | 0.025 | 0.10 meV / 1.6 MPa | 0.06 meV / 0.95 MPa |
| graphite | 0.177 | 0.73 meV / 13.2 MPa | 0.42 meV / 7.6 MPa |
| MgO | 0.248 | 1.02 meV / 17.4 MPa | 0.59 meV / 10.1 MPa |

So relative to the untruncated sum, rc = 30 A is about 0.06-0.6 meV/atom and 1-10 MPa off; MgO at 30 A is slightly above 0.5 meV/atom when compared with rc -> infinity instead of rc = 40 (T1 was pre-registered against rc = 40 and is judged against it). The tail estimate for the doc (0.64 meV at 25 A, 0.01 GPa) was a single-element continuum number; the measured values are 0.1-1.0 meV and 1.6-17 MPa at 25 A, depending on density and C6, i.e. up to ~1.6x larger in energy.

### 3.4 Cross-check, GPU vs `tests_pytest/dftd2_reference.py` (water 648)
| rc | E_gpu - E_ref [eV] | max abs dF [eV/A] (max abs F_D2 = 4.9e-2) | dP [GPa] (P = -0.742 / -0.755) |
|---|---|---|---|
| 12 | 4.9e-7 (E = -21.837 eV) | 4.9e-8 | -1.1e-7 |
| 25 | 5.1e-7 (E = -22.366 eV) | 5.0e-8 | -1.1e-7 |
(relative ~2e-8 in energy; consistent with single-precision accumulation; AC-1 tolerance of DR-002 is not restated here.)

### 3.5 Cost, water 648 (RTX A6000, 1000 NVE steps, dt 0.5 fs, min of 3 runs, NEP89 alone = 1.555 ms/step)
| rc [A] | neighbors/atom (rc sphere, full list) | total [ms/step] | D2 = on - off [ms/step] |
|---|---|---|---|
| 12 | 723 | 4.71 | 3.15 |
| 16 | 1714 | 9.66 | 8.10 |
| 20 | 3351 | 40.6 | 39.0 |
| 25 | 6542 | 47.5 | 46.0 |
| 30 | 11311 | 56.9 | 55.4 |
| 40 | 26801 | 132.9 | 131.3 |
Run-to-run spread is below 2 %. Run-time difference, not nsys. At the proposed rc = 30 the D2 term costs about 35x the NEP89 evaluation of this system, and 30x at rc = 25. The jump between 16 and 20 A (x4.8 in time for x1.95 in neighbors) looks like a change of code path rather than a smooth cost law (not investigated, see 5).

### 3.6 NVE check, water 648, dt 0.5 fs, 2 ps, `velocity 300 seed 20251103` (identical start), no thermostat, s6 = 0.75, thermo every 10 steps
| run | E_tot std [eV] | max-min [eV] | linear drift [meV/ps] | drift [meV/atom/ps] |
|---|---|---|---|---|
| NEP89 only | 0.0170 | 0.0848 | +27.1 | +0.042 |
| rc = 12 | 0.0081 | 0.0512 | -1.5 | -0.002 |
| rc = 25 | 0.0099 | 0.0564 | -4.6 | -0.007 |
| rc = 30 (proposed) | 0.0088 | 0.0561 | -8.2 | -0.013 |
No run is worse than the D2-off NEP89 baseline in std or max-min; the values with D2 are equal within noise (the earlier 0.056 / 0.058 eV figures correspond to the max-min column, not the std). The drift of rc = 30 is the largest of the D2 runs, but it is smaller in magnitude than that of the D2-off run and the differences between runs are of the size expected from float32 noise on one 2 ps trajectory of a chaotic system (one trajectory per rc; no statistics). Not a pass/fail item.

## 4. Verdict against the plan
Gate of DES-0002 Phase 2: "a default rc and the table are recorded and accepted by the human". Recorded: proposed default rc = 30 A. ACCEPTED by the human on 2026-11-03 (default rc = 30 A, results as tabulated). `src/force/dftd2.cuh` `DEFAULT_CUTOFF` (25 -> 30) and `doc/gpumd/input_parameters/dftd2.rst` (provisional note removed, tail numbers replaced by the measured values) were updated after acceptance.

## 5. Opinion on the thresholds and what was not verified
- T2 (forces) is satisfied by orders of magnitude at every rc and carries no information; the shifted-force scheme ensures that. T1 (0.5 meV/atom) and T3 (0.02 GPa) decide. They are of the same order as the NEP89 accuracy, so T1 is not unreasonable; T3 alone would have given 25 A.
- The cost at 25-30 A on 648 atoms is large (46-55 ms/step vs 1.6 ms for NEP89). Cost was not part of the pre-registered criteria, so it did not enter the choice. The human may want to weigh 25 vs 30 knowing that T1 fails at 25 by 10 % (graphite) to 54 % (MgO) and the cost is only 17 % lower. The 16->20 A jump in cost per neighbor suggests that the kernel path could be examined (outside this task).
- Not verified: only three systems, s6 = 0.75 only (all energies and pressures scale linearly with s6, so the criteria in meV would scale too), no triclinic cell, no NPT, a single structure per system (graphite/MgO at 0 K with a rattle), reference rc = 40 is itself truncated (see 3.3), tail law extrapolated from rc <= 40. Water density and structure are those of a NEP89 NVT run at 300 K/0.997 g/cc fixed volume; the NEP89 pressure of this box was not examined. Timing is from the run-time difference, no profiler.
