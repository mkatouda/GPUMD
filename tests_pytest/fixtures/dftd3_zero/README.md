# Fixtures of test_dftd3_zero.py (DFT-D3 zero damping, `dftd3 <f> <rc> <rc_cn> [zero|bj]`)

| File | Content | Produced by |
|---|---|---|
| `reference.json` | simple-dftd3 D3(0) energy/forces/virial (two-body) for PBE (8 structures), revPBE, BLYP, B3LYP, TPSS (3 structures each) | `generate_reference.py` |
| `heavy_cluster.xyz` | periodic 10 A cell with La, Hg, Tl, Bi, Pt, Os, W, Cs, Xe, I (Z <= 83; probes more R0 table entries) | `generate_reference.py` |
| `bj_frozen.json`, `tiny24.xyz` | output of the pre-change binary for `dftd3 pbe rc rcn` (BJ), bitwise reference | `generate_bj_frozen.py` |
| `zero_nep_H_C_O_Cu.txt` | all-zero NEP (H C O Cu): output of a run = D3 contribution alone, bit-reproducible | `make_zero_nep.py` |

Structures of the reference cases (`benzene_dimer`, `graphite`, `water27`, ...) are the ones of the BJ
test, `../dftd3/*.xyz` (`../dftd3/generate_reference.py`); they are not duplicated here.

## Reference values (`reference.json`)
- Oracle: simple-dftd3 (python `dftd3`, version recorded in `reference.json`), `ZeroDampingParam` with
  explicit parameters s6 = 1, sr8 = 1, alpha6 = 14 (alpha8 = 16) and (sr6, s8) of the functional,
  `s9 = 0` (no ATM). The script checks that the explicit parameters equal simple-dftd3's own
  parameters of the same functional (simple-dftd3 names `b-lyp`, `b3-lyp` = GPUMD names `blyp`, `b3lyp`).
- Parameters (def2-QZVP set, Grimme et al., J. Chem. Phys. 132, 154104 (2010)): PBE sr6 = 1.217, s8 = 0.722;
  revPBE 0.923 / 1.010; BLYP 1.094 / 1.682; B3LYP 1.261 / 1.703; TPSS 1.166 / 1.105.
- Real-space cutoffs equal to those given to GPUMD (pair = `rc_potential`, CN = `rc_cn`), as in the BJ test.
- **Length constant.** All lengths (positions, cutoffs, gradient and virial conversion) use the constant of
  `src/force/dftd3para.cuh`, `Bohr = 0.5291772575069165` A, not CODATA2018 (0.529177210903 A) of
  simple-dftd3. simple-dftd3's internal pair radii R0 follow CODATA2018, so `sr6` and `sr8` are multiplied by
  0.529177210903/0.5291772575069165 so that r/R0 equals the ratio of Angstrom quantities (R0 table in
  Angstrom). With alpha = 14/16 an inconsistent constant changes the energy by up to 1.5e-6 relative
  (VER-0023 section 2). The implementation must therefore evaluate r/R0 with R0 in Angstrom (or
  with one consistent Bohr constant, the one of `dftd3para.cuh`). Residual sub-1e-6 differences from the
  covalent-radius table of simple-dftd3 (CN) are not removable by this trick; they are part of what
  BASELINE (below) has to measure.
- Units: eV, eV/A, virial = -sigma (GPUMD convention), Hartree = 27.211386245988 eV.

## Tolerances
`min(cap, max(2 x BASELINE, floor))`, caps of ADR-0002 (relative 2e-6 of |E_D3| and of the largest virial
component, 1e-5 eV/A for forces), floors 2e-6 eV, 1e-6 eV/A, 2e-5 eV as in the BJ test.
**BASELINE for zero damping is not measured yet** (no implementation exists); the test provisionally uses
the BJ baselines of the same structure (`BJ_BASELINE` in the test, same float summation); structures without
one (`heavy_cluster`) use the floors. Measure BASELINE with the zero path in DES-0003 Phase 4 and replace the table;
do not loosen a tolerance to make the test pass.

## BJ bit-identity (`bj_frozen.json`)
- The reference is the output (dump lines with `precision double`: energy, virial, species, positions, forces) of a
  `dftd3 pbe 12 6` run with the all-zero NEP, so it is the D3 contribution alone and bit-reproducible.
  The `stress` field of the header is removed (varies from run to run on the same binary).
- Binary: built from the main-tree `src/` of 2026-11-06 (DR-001 dftd3 kernels with fixed-order reductions,
  uncommitted at that date; sha256 of `dftd3.cu` is in the JSON), `-arch=sm_86`, CUDA 12.3, RTX A6000.
  The kernels at the develop HEAD (3b2c8db8) use atomicAdd and are NOT bit-reproducible run to run (virial and
  forces differ in the last digits), so no bitwise reference exists for that binary.
- The reference is hardware dependent in principle (same instruction sequence on other GPUs may round
  differently); if the test fails on a different GPU/CUDA version only, regenerate by running the script with the
  unchanged pre-change binary on that machine, never with the new binary.
