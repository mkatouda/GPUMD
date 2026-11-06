# Fixtures of test_dftd2_qe.py (`dftd2 <s6> [<rc>] [table grimme2006|qe] [trunc sf|plain]`, DR-004 / ADR-0004)

| File | Content | Produced by |
|---|---|---|
| `d2_reference.py` | double-precision numpy reference of the D2 variants (`qe`/`grimme2006` x `plain`/`sf`), the documented oracle; UNMODIFIED copy | requester |
| `ptfe_cryst157_{vol000,volm05,volm10,volm15}.xyz` | 4 PTFE crystal structures (45 atoms, C/F, triclinic cell 5.69 x 4.92 x 19.7 A at vol000; volumes -5 / -10 / -15 %) | requester |
| `ref_cryst157_<vol>_{qe,grimme2006}_{plain,sf}_rc12.xyz` | reference energy, forces, virial (s6 = 0.75, rc = 12 A), 16 files | `d2_reference.py` (run by the requester) |
| `summary_rc12.csv` | energy per atom, max force, virial trace/3 of the 16 reference cases | requester |
| `default_frozen.json`, `generate_default_frozen.py` | output of the PRE-CHANGE binary for `dftd2 0.75` and `dftd2 0.75 12` (bitwise reference for AC-2) | `generate_default_frozen.py` |
| `water27.xyz`, `graphite.xyz`, `water27_tri.xyz`, `benzene_dimer.xyz`, `tiny24.xyz` | structures of the frozen cases (copies of `../dftd3/` and `../dftd3_zero/`, so that this directory is self-contained) | copies |
| `zero_nep_H_C_O_Cu.txt` | all-zero NEP (copy of `../dftd3_zero/`): a run gives the D2 contribution alone, bit-reproducible | copy |

## Provenance of the reference (oracle)
- Origin: request `REQUEST_dftd2_qe_compat.md` of the PTFE NEP/qNEP project, 2026-11-06 (files under
  `tmp/request_to_fork/reference/` of the requester; copied here unchanged).
- `d2_reference.py` sha256: `2c0351f03180fcb1a1b23af5ab38d7991877aff7114fb4efd4d328c9024ca4e1`
  (checked by `test_d2_reference_script_is_the_recorded_one`). Do not edit it; the stored `ref_*.xyz`
  files are the test oracle and are never regenerated (`test_stored_reference_equals_the_documented_oracle`
  recomputes them on the CPU and compares to 1e-9 relative).
- Definition: pair energy phi(r) = - s6 C6_ij f(r) / r^6, f = 1 / (1 + exp(-20 (r / R0_ij - 1))), all periodic
  images within rc, i = j images with weight 1/2; `qe`: C6 = D3 `c6ab(Zi, Zj, 1, 1, 1)`, R0 = 1.1 `r0ab`
  (QE `dftd3_version = 2`); `grimme2006`: C6 = sqrt(C6_i C6_j), R0 = R0_i + R0_j (the existing `dftd2` table);
  `plain`: hard cutoff, `sf`: shifted force. Constants: CODATA2018 (Bohr 0.529177210903 A, Hartree
  27.211386245988 eV), double precision. Virial: GPUMD/NEP convention (-dE/d(strain), 9 components).
- QE 7.2 validation claim (the requester's, NOT re-verified by the fork's tests): the numpy implementation
  reproduces the D2 energies and forces printed by QE 7.2 (`vdw_corr = 'dft-d3'`, `dftd3_version = 2`, PBE)
  to the printed digits and the stress to 0.01 kbar on PTFE structures. The fork's own comparison of the
  parameter tables with the QE 7.2 sources is VER-0025.

## Tolerances of the reference test (DR-002 AC-1, relative caps, no absolute floors)
energy 2e-6 |E_ref|; virial 2e-6 max|W_ref|; force 1e-5 max|F_ref|. The D2 part is (run with) - (run without)
on NEP89 (the NEP part cancels). `grimme2006/sf` (the unchanged default path) is the harness check of the
comparison conventions: it must pass on the pre-change binary.

## Default-path bit identity (`default_frozen.json`)
- Content: dump.xyz lines 2.. (`precision double`: energy, virial, species, positions, forces) of a static run with the all-zero
  NEP and `dftd2 0.75` / `dftd2 0.75 12`, plus the stdout lines containing "DFT-D2", for 5 structures
  (small boxes: water27 9.3 A cubic, graphite, water27_tri; non-periodic benzene_dimer and tiny24, a 40 A box,
  for the large-box path). The `stress="..."` field is removed (varies run to run with the zero potential).
- Binary: develop `af269060`, `src/force/dftd2.cu` unmodified (sha256 recorded in the JSON), `-arch=sm_86`, CUDA 12.3,
  RTX A6000. Each run was repeated 3 times (separate processes) and was bit-identical; a second complete
  generation produced a byte-identical JSON. The D2 kernels use no atomics, so the output is deterministic
  (unlike the DR-003 BJ case, no single-warp restriction was necessary).
- If the test fails only on another GPU / CUDA version, regenerate with the unchanged pre-change binary on that
  machine, never with the new binary.
- Regeneration: `python generate_default_frozen.py <pre-change gpumd> <source description> <sha256 of dftd2.cu>`.

## Finite-difference systems
Sparse C/F cells (3 atoms triclinic, periodic; 6 atoms open) built deterministically in the test; rc is placed in a gap
(>= 0.3 A) of the pair-distance spectrum, so that no pair crosses a hard cutoff during the stencils (the energy of
`trunc plain` is discontinuous at rc). s6 = 75, step 0.01 A (force) and 1e-3 (strain), 5-point stencils,
tolerance 2e-4 of the largest D2 force / virial component (as in `test_dftd2_consistency.py`).

## QE 7.2 parameter tables (added with the review fixes)
| File | Content | Produced by |
|---|---|---|
| `qe72_c6_first_point_Eh_bohr6_Z86.csv` | 86 x 86 matrix of QE `c6ab(Zi, Zj, 1, 1, 1)` (first reference point of both atoms), Hartree bohr^6 | `specs/verification/VER-0025-dftd2-qe-parameters/scripts/extract_qe_d2_params.py` (copy of `.../data/`) |
| `qe72_r0ab_angstrom_Z86.csv` | 86 x 86 matrix of QE `setr0ab` r0ab, Angstrom (decimal literals; `real(4)` in 7.2) | same |

Source: Quantum ESPRESSO tag `qe-7.2` (commit b4c5e8deeb7d1863e7553a87b45f8d22c21926a5), `dft-d3/pars.f90`
and `dft-d3/core.f90`. `dftd2_qe_oracle.py` (tests_pytest/) builds the numpy reference for any Z <= 86 from these
CSVs with the CODATA2018 constants of QE 7.2; `test_dftd2_qe_extended.py` checks `src/force/dftd2qe_table.cuh` against
them and, when the QE 7.2 sources are readable (env `DFTD2_QE72_SRC`: a directory holding `pars.f90` and `core.f90`,
or the git checkout `/home/katouda/build/q-e` with tag `qe-7.2`), against an independent re-parse of the Fortran data statements.
