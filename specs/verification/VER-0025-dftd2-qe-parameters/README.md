# VER-0025: QE `dftd3_version = 2` parameter provenance (C6 first point, r0ab, formula)

- Date: 2026-11-06. Status: verification record (not validation). Supports DR-004 D3/AC-3, DES-0004 Phase 1.
- No `src/` change. Outcome: **Stage 1 gate met (0 mismatches, Z <= 86), Stage 2 gate met, Stage 3 (QE run) done and agrees to 4.6e-8.**
  Open point for the owner: the Bohr/Hartree constant of the GPU table (section 5).

## 1. Sources
- QE tag `qe-7.2` = commit `b4c5e8deeb7d1863e7553a87b45f8d22c21926a5` (tag object `deec76b2`), read with `git show qe-7.2:dft-d3/<file>` in
  `/home/katouda/build/q-e` (repo untouched). Files: `pars.f90` (C6), `core.f90` (`setr0ab`, `setfuncpar`, `pbcedisp`, `pbcgdisp`, `copyc6`),
  `common.f90` (`autoang`, `limit`), `api.f90`; sha256 in `data/SOURCE.txt`. Also `PW/src/electrons.f90` (energy x 2 Eh->Ry) and
  `Modules/constants.f90` (BOHR_RADIUS_SI, AUTOEV).
- QE development HEAD `c7587f9f` (7.6-515) extracted the same way: `pars.f90` is byte-identical to 7.2 (`cmp`), first-point C6 identical
  (max abs diff 0), r0ab decimals identical (0 of 4465 differ); 7.2 literals are `real(4)`, HEAD literals carry `_wp` (double).
- Local binaries `/home/katouda/bin/qe/bin` are built from HEAD (7.6-515), NOT 7.2.
- GPUMD: `src/force/dftd3para.cuh` (`c6_ref_cpu`, `cn_ref`), `src/force/dftd3zero_table.cuh` (`d3zero_r0ab`), VER-0023 `r0ab_angstrom.csv`.
- Requester data: `tmp/request_to_fork/` (`d2_reference.py`, copy in `scripts/`, sha256 in `data/SOURCE.txt`).

## 2. Definition read from QE 7.2 (version == 2)
- `setfuncpar`: `rs6 = 1.1d0`, `s18 = 0`, `alp = 20.0d0`; `s6` per functional (`pbe`: 0.75, exact in binary). Other `s6` values are real(4) literals (1.2, 1.05, ...), not used here.
- `pbcedisp` (energy): for `iat < jat`: `c6 = c6ab(iz(jat),iz(iat),1,1,1)`, images tau with `r2 <= rthr` (`rthr = cutoff**2`, `cutoff = sqrt(9000)` bohr = 50.2 A,
  `api.f90`; hard cutoff, no shift); `damp6 = 1/(1+exp(-alp6*(r/(rs6*r0ab)-1)))`; `e6 += c6*damp6/r^6`. For `iat == jat`, images
  with tau != 0 and `r2 <= rthr` contribute with factor 0.5. API: `disp = -e6*s6 - e8*s18 - e6abc` with `s18 = 0`, three-body off (QE `input.f90`: `dftd3_version==2 -> dftd3_threebody=.false.`);
  `PW/src/electrons.f90` multiplies by 2 (Hartree -> Ry). No CN, no `sqrt(C6 C6)`, no per-atom combination, no C8 (s18 = 0), no r2r4 / rcov involved.
- `pbcgdisp` (forces/stress, `version.eq.2` branch): `R0 = r0ab*rs6`, `c6 = c6ab(...,1,1,1)*s6`, same damping, same cutoff; analytical gradient of the same energy.
  Finite-difference consistency is tested in `d2_reference.py --selftest`.
- Units: C6 in Hartree bohr^6 (as stored in `pars.f90`, 5 numbers per line: C6, Zi(+100*(iadr-1)), Zj(+100*(jadr-1)), CNi, CNj; `limit()` decodes the reference indices).
  r0ab is stored in **Angstrom** (decimal values) and `setr0ab` divides by `autoang = 0.52917720859` (`dft-d3/common.f90`, NOT the QE constant 0.529177210903) to get bohr; positions are in bohr.
  Hence effective R0 [A] = r0ab * (0.529177210903/0.52917720859) = r0ab * (1 + 4.37e-9).
- Conclusion for DR-004 D3: the INPUT_PW text "Original Grimme-D2 parametrization" is misleading; the source computes "D2 form with D3 tables" (C6 = first stored reference point,
  R0 = 1.1 * D3 r0ab), as the requester found numerically.

## 3. Method (scripts in `scripts/`; numpy only; run with `python -I`)
- `extract_qe_d2_params.py <dir> <out> <label>`: parses `pars.f90` (all `pars(a:b)=(/ .. /)` blocks, asserts 161925 values = 32385 lines x 5) and `setr0ab` (4465 values),
  emulates `copyc6` (non-min/max branch) and `limit()`; first point = the line with iadr = jadr = 1; asserts exactly one such line per pair and symmetry.
- `compare_qe_gpumd.py`: parses `c6_ref_cpu`, `cn_ref`, `d3zero_r0ab` from the `.cuh` files and VER-0023 csv; compares entry by entry; writes `data/stage1_compare_report.txt`.
- `stage2_pairs_constants.py`, `compare_xyz_reference.py`, `stage3_pw_vs_numpy.py` (Stage 2 / 3 below). Scratch was outside the repo.
- GPUMD layout: `c6_ref_cpu[z12*25 + a*5 + b]`, `z12 = zs*94 - zs(zs-1)/2 + (zl - zs)` (0-based Z, zs <= zl), `a` runs over the smaller-Z element's references.
  First point = `c6_ref_cpu[z12*25]`. First reference CN (`cn_ref[5*Z]`) is 0 for every element with Z <= 86 except **H (0.9118) and Ce, Z = 58 (2.7991)**; QE's
  `c6ab(i,j,1,1,1)` is literally the first stored point and uses these too (QE `c6ab(Z,Z,1,1,2)` nonzero for exactly Z = 1, 58). The two tables agree on this (CN of first points: 0 mismatches
  over all 4465 pairs).

## 4. Results
| Check | Result |
|---|---|
| Stage 1a: first-point C6, Z <= 86 (3741 pairs), QE 7.2 vs `c6_ref_cpu[z12*25]` | **0 decimal mismatches**; the float32 table value differs from the QE double by <= 5.9e-8 relative (float32 rounding only) |
| Stage 1a info: pairs with Z >= 87 (724 pairs) | 0 mismatches (same dftd3-lib data) |
| Stage 1b: r0ab, Z <= 86 (3741 pairs), QE 7.2 vs VER-0023 csv and vs `d3zero_r0ab` | **0 mismatches** (decimal and as float32) |
| Stage 1b info: r0ab Z = 87..94 QE 7.2 vs VER-0023 | 0 differences (the VER-0023 "disagreement" for Z >= 87 is between dftd3-lib/QE and simple-dftd3; QE = dftd3-lib) |
| QE 7.2 vs HEAD | C6 identical; r0ab decimals identical; only real(4) vs `_wp`: rel difference of r0ab in bohr <= 5.9e-8 (= float32 rounding), up to 1.2e-6 in the exponent argument alp*(r/R0) near r = R0 |
| Stage 2: pair values from the extraction | C-C 49.113 / 2.9103 / 3.20133, C-F 20.2827 / 2.4770 / 2.7247, F-F 9.6916 / 2.2996 / 2.52956: equal to the request table to its printed digits |
| Stage 2: `d2_reference.py --selftest` | 4 of 4 PASS (force and virial vs finite differences, tolerance 1e-7 in the script) |
| Stage 2: reference reproduction, `ptfe_cryst157_vol000.xyz`, `--table qe --trunc plain/sf --rc 12` | energy, forces, virial: **identical** (0.0 difference; output text identical to `ref_cryst157_vol000_qe_{plain,sf}_rc12.xyz`); gate 1e-10 met |
| Stage 3: pw.x 7.6-dev, CF4 in a 9 A triclinic cell, `dftd3_version=2`, rc = sqrt(9000) bohr | energy rel 4.6e-8 (diff 1.6e-9 eV of 0.0350 eV), forces rel 4.6e-8 (max diff 1.5e-9 eV/A), stress rel 2.7e-8 (diff 2.6e-12 eV/A^3, 0 at the 0.01 kbar level) |

Stage 3 details (`data/stage3_result.txt`, inputs `data/stage3_pw_*.in`): ONCV PBE pseudos (sg15) for C and F, ecutwfc 40 Ry, gamma, conv_thr 1e-12, `input_dft='pbe'` (s6 = 0.75).
`DFT-D3 Dispersion` energy = difference of XML `etot` with and without `vdw_corr` (identical SCF density because D3/D2 does not enter the KS potential); forces and stress likewise
(with minus without). Full-precision XML values are used because the text output has only 8 decimals in Ry. The printed value is -0.00257072 Ry = -0.03497643 eV
(also consistent). The remaining 4.6e-8 is the expected `autoang` effect (20 * 4.37e-9 = 8.7e-8 bound), because the reference uses decimal r0ab without the factor 1.000000004371.
Reproduce: `source /opt/intel/oneapi/setvars.sh; LD_LIBRARY_PATH=/opt/AMD/aocc-compiler-4.1.0/lib:$LD_LIBRARY_PATH python -I scripts/stage3_pw_vs_numpy.py <work> --run --np 4` (about 5 s).

## 5. Bohr / Hartree constant (for the GPU table)
| quantity | value | relative to CODATA 2018 (`d2_reference.py`, QE 7.2 `constants.f90`) |
|---|---|---|
| Hartree*Bohr^6, CODATA 2018 (27.211386245988 * 0.529177210903^6) | 0.597526904282 eV A^6 | 0 |
| GPUMD `HartreeBohr6` (`dftd3para.cuh`) | 0.597527426644 | **+8.74e-7** |
| GPUMD `Bohr` 0.5291772575069165 | - | +8.8e-8 (length), +5.3e-7 (sixth power) |
| QE 7.2 AUTOEV vs 27.211386245988 | - | 1.3e-15 |
- Using `c6_ref * HartreeBohr6` would shift every C6, hence E, F and virial of `table qe`, by +8.74e-7 relative against the requester's labels (an energy error of this size,
  systematic, not random). Computing C6 [eV A^6] on the host in double with CODATA 2018 gives agreement with `d2_reference.py` and QE 7.2 constants to rounding. `r0ab` is in Angstrom and needs no Bohr constant.
  Which constant to use is a decision for the owner/ADR-0004 (not made here).

## 6. Host-side availability (read-only analysis)
- `c6_ref_cpu` is a plain host `const float[111625]` in an anonymous namespace of `dftd3para.cuh`, readable on the host (`dftd3.cu` copies it to the GPU, `dftd2.cu` does not include it).
  Values are float32 of the decimals (<= 5.9e-8). `dftd2.cu` currently includes only `dftd2para.cuh` (double tables, Grimme 2006).
- `d3zero_r0ab` is `__constant__ float[4465]` in `dftd3zero_table.cuh` (included only by `dftd3.cu`); a `__constant__` array is NOT readable from host code. The host-side dftd2 code therefore needs a
  host copy of r0ab: the VER-0023 candidate fragment `data/d3zero_tables.inc` (packed lower triangle, double, Angstrom, `index = i(i+1)/2 + j`) or a generated `static const` array; or a lookup by the
  existing generator. No code change was made. Z <= 86 is the same in all three (this record: 0 mismatches).
- Precision note: QE 7.2 itself uses float32-rounded r0ab (real(4) literals); a double host table of the decimals equals QE HEAD, a float32 table equals QE 7.2 (both within 5.9e-8 relative of each other).

## 7. Limitations
- QE 7.2 was NOT run; only its sources were read. The runtime comparison (Stage 3) used the 7.6-dev binary, which has `_wp` (double) r0ab literals, so the 7.2 float32 effect (<= 1.2e-6 in the exponent argument) is not exercised by it.
  The requester's own QE 7.2 comparison (energies and forces to printed digits, stress to 0.01 kbar, PTFE) remains the evidence for 7.2 and was not reproduced.
- Stage 3 used one small molecule-in-box system (5 atoms, stress only 0.15 kbar); it does not cover a dense crystal or the unit test of many images beyond what `d2_reference.py` enumerates.
- The Z >= 87 columns are only compared, not decided (DR-004 P3 restricts `table qe` to Z <= 86).
- Parameters verified against QE source and software; the primary literature (Grimme 2006/2010) was not read.
