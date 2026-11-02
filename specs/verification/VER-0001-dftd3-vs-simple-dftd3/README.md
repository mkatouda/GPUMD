# VER-0001: GPUMD DFT-D3 vs simple-dftd3 (accuracy and timing)

- Status: Completed (verification, not validation)
- Date: 2026-11-03
- Requirement IDs: none yet (exploratory baseline for the dielectric-response work)
- Source under test: `src/force/dftd3.cu` at commit `655df6e1` (`develop`; `src/` identical to `master` `3b2c8db8`)

## Purpose
Check that the `dftd3` keyword (D3 with BJ damping, added to NEP) reproduces an
independent D3 implementation, and measure its cost.

## Setup
- Machine: RTX A6000 (sm_86), driver 595.71.05, CUDA 12.3 (`/usr/local/cuda`), 16 CPU cores.
- Build: `make CUDA_ARCH="-arch=sm_86" -j4`.
- Model: `potentials/nep/nep89_20250409/nep89_20250409.txt` (NEP89).
- GPUMD D3 contribution = (run with `dftd3 pbe <rc> <rc_cn>`) - (same run without it);
  static run, `time_step 0`, `dump_xyz ... force precision double` (energy, forces, virial).
- Reference: simple-dftd3 (conda-forge, `dftd3-python`), `RationalDampingParam(method='pbe')`,
  two-body only (ATM three-body off), double precision. Same pair / CN cutoffs
  ("same-cutoff"), and library defaults ("converged") to separate truncation from
  implementation error. Both use PBE-BJ: s6=1, s8=0.7875, a1=0.4289, a2=4.4407.
- Systems (`scripts/build_systems.py`): benzene dimer (non-periodic, 24 atoms), rattled
  graphite 3x3x1 (hexagonal cell, 36 atoms), 27 randomly rotated H2O in a 9.3 A cubic box
  (81 atoms), and a sheared (triclinic) copy of it.
- Cutoffs (pair / CN, A): 12/6 and 25/15.

## Accuracy results (same-cutoff reference; `results/accuracy.json`)
| System | N | |dE| (eV) | max |dF| (eV/A) | max |dVirial| (eV) |
|---|---|---|---|---|
| benzene dimer | 24 | 4e-7 | 3e-8 | 3e-7 |
| graphite (hexagonal) | 36 | 4e-6 | 5e-7 (2.6e-6 at 25/15) | 5e-6 (1e-4 at 25/15) |
| water27, cubic | 81 | 2e-6 | 1e-8 (1.3e-7 at 25/15) | 2e-6 |
| water27, triclinic | 81 | 2e-6 | 3e-8 (1.8e-7 at 25/15) | 2e-6 |

Forces compared with reference magnitudes of 0.01-0.08 eV/A. All differences are at the
single-precision (`float`) level.
- Truncation error vs the library-default cutoff: |dE| about 0.03-0.05 eV at 12/6 A,
  about 1e-3 eV at 25/15 A (not an implementation error).
- Virial sign: GPUMD virial equals the NEGATIVE of simple-dftd3 `virial` (sigma). Determined
  by agreement, not by an independent strain test.
- An early graphite mismatch (0.6 eV) was a reference-side error (lattice vectors passed
  transposed; simple-dftd3 expects rows). Fixed in `scripts/accuracy.py`.

## Timing results (`results/slope.json`, `results/timing.json`)
Water box, NEP89, `dftd3 pbe 12 6` unless noted. Per-step time = slope between 100- and
500-step runs (fixed start-up cost of about 1.3 s removed; D3 start-up cost is negligible).
| N | no D3 (ms/step) | D3 (ms/step) | D3 extra (ms/step) | ratio |
|---|---|---|---|---|
| 81 | 1.34 | 3.84 | 2.5 | 2.9 |
| 5,184 | 1.77 | 6.21 | 4.4 | 3.5 |
| 41,472 | 6.64 | 14.62 | 8.0 | 2.2 |
| 5,184, cutoff 25/15 | 1.66 | 42.4 | 40.7 | 25 |

- D3 cost scales with the cube of the pair cutoff: 25/15 vs 12/6 gives 9.2x extra time
  (cube of the cutoff ratio: 9.0).
- CPU reference (simple-dftd3, one thread, energy + gradient): 5 ms (81 atoms), 3.1 s
  (5,184), 196 s (41,472). Different algorithm; for orientation only.

## Relation to upstream discussion #898
The report "D3 makes GPUMD about 4x slower" (brucefan1983/GPUMD discussions/898) gives no
system size, cutoffs or GPU, so it is not reproduced exactly. For NEP89 with `dftd3 pbe 12 6`
the slowdown here is 2.2-3.5x, consistent in order of magnitude.

## Limitations / not verified
- Two-body D3(BJ) only; GPUMD has no ATM term, and none was compared.
- Only PBE, only systems with H, C, O, up to about 41k atoms; one GPU.
- No kernel-level profile: where the D3 time goes is unknown.
- Multi-GPU NEP + D3 (documented as unsupported) and non-NEP potentials were not tested.
- Timing is a single machine, repeated runs not statistically analysed.

## Reproduce
Needs `ase`, `dftd3` (conda-forge `simple-dftd3`, `dftd3-python`), `numpy`. Scripts use
absolute paths from the original session; adjust `GPUMD`, `NEP` at the top of each script.
```
python scripts/build_systems.py          # writes *.xyz in the current directory
python scripts/accuracy.py               # -> accuracy.json
OMP_NUM_THREADS=1 python scripts/timing.py
python scripts/slope.py                  # -> slope.json
```
Run in a scratch directory, not in the repository (see `AGENTS.md`, "Generated outputs").
