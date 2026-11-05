# VER-0023: D3(0) parameter provenance (r0ab, functional parameters)

- Date: 2026-11-06. Status: verification record (not validation). Supports DR-003 AC-7 / DES-0003 Phase 1.
- No `src/` change. Outcome: **gate met for Z <= 86 and for all functionals; NOT met for Z = 87-94 (sources disagree). See section 5.**

## 1. Sources
- Primary table source: `/home/katouda/work/dielectric-response_pj/dftd3-lib`, commit `3c99ee981266548a37588abe3a6f5cb56092d51a`
  (DFTD3 V3.1 Rev 1 repackaging), `lib/core.f90` sha256 `d08285de...f082ef4` (`data/SOURCE.txt`).
  r0ab: `subroutine setr0ab` (4465 packed lower-triangle values, 64 chunks). Parameters: `setfuncpar`, `version.eq.3` block
  (version 3 = D3 zero damping; version 5 = D3M(zero), NOT used).
- Independent oracle: simple-dftd3 1.7.0 (`dftd3` Python bindings, env `/home/katouda/miniforge3/envs/py312-teacher-orb`; `s-dftd3` binary also exists in `py312-xtb`). No env was modified, nothing installed.
- The primary literature (Grimme et al., J. Chem. Phys. 132, 154104 (2010)) was NOT read (no network). Parameter values are verified
  against two independent software implementations, not against the paper table.

## 2. Units and conventions
- r0ab is stored in **Angstrom** in the source; the library divides by `autoang = 0.52917726` (`lib/common.f90`) -> bohr.
  Sanity: r0ab(H,H) = 2.1823 A = 4.1239 bohr.
- Damping (lib `core.f90`, version 3 and DR-003): `f_n = 1/(1 + 6 (sr_n R0_ij / r)^alpha_n)`, `alpha6 = 14` (lib `alp`),
  `alpha8 = alpha6 + 2 = 16` (`prg/main.f90`, `lib/api.f90`), `sr6 = lib rs6`, `s8 = lib s18`, `sr8 = lib rs18` (default 1.0;
  only `slater-dirac-exchange` has 0.697). This equals the DR-003 formula, i.e. the "to be confirmed" item is confirmed.
- **Bohr constant:** dftd3-lib uses 0.52917726; simple-dftd3 effectively uses CODATA2018 0.529177210903 (recovered R0 ratio 9.278e-8, exact).
  Because of alpha = 14/16 this produces up to 1.5e-6 relative difference of the damped energy in the transition region
  (r ~ R0), larger than the 2e-6/ADR-0002-scale margin budget. Phase 3/4 must use ONE constant and compare against
  simple-dftd3 with the matching constant (or accept ~1.5e-6).
- dftd3-lib literals (`rs6=1.217`, `r0ab` values, `s6=0.64`) are Fortran **single precision**; the library uses
  float32(x). Generated decimals are the published values; the lib deviates from them by <= 6e-8 per input, <= 2.2e-6 in damped energy.
- Parameter variants: lib default (`TZ=false`, def2-QZVP, 48 functionals; the published set) and `TZVPP` (11 functionals). The
  data file holds both; the header fragment only the def2-QZVP set.

## 3. Method (scripts in `scripts/`)
- `gen_d3zero_tables.py <dftd3-lib> <outdir>`: parses `core.f90` (r0ab chunks, `case` blocks of the version-3 select, comments stripped),
  asserts counts (4465 = 94*95/2, symmetric, no duplicate functional, `dftb3` commented out is excluded). Writes `data/r0ab_angstrom.csv`
  (94x94, Z=1..94), `data/d3zero_params.csv`, `data/d3zero_tables.inc` (candidate C++ fragment: packed lower-triangle double array
  `index = i(i+1)/2 + j`, Angstrom; `D3ZeroParam` table), `data/SOURCE.txt`. The `.inc` was syntax-checked with `g++ -Wall -Wextra`.
- `probe.f90`: driver on the lib API (full double output) to dump the runtime r0ab / parameters and dimer energies. It needs
  `private` removed from `type dftd3_calc` in a SCRATCH copy of `lib/api.f90` (line 59; no numerical change); built in the scratchpad.
- `crosscheck_d3zero.py <probe> <data> [results]` (run with the python of `py312-teacher-orb`):
  A generated r0ab vs lib runtime (float32 emulation, exact); B parameters vs lib runtime `setfuncpar` (exact);
  C simple-dftd3 `ZeroDampingParam(method=name)` vs explicit generated (s6,s8,sr6,sr8,alpha6) for 48 functionals x 8 dimers x 5 r;
  D R0 of every pair recovered from dimer energies at two sr6 values (C6 cancels): from simple-dftd3 and from the lib itself;
  E DR-003 formula with generated tables, C6/C8 from simple-dftd3 (damping switched off), vs simple-dftd3 and vs lib energies
  (11 pairs x 6 r = 66 points; r = 0.5..5 R0 incl. r << R0).
- Reproduce: copy dftd3-lib to scratch, `make dftd3`, patch line 59 of `lib/api.f90`, `make` in `lib/`, build `probe.f90`.

## 4. Results (`data/crosscheck_results.txt`)
| Check | Result |
|---|---|
| A r0ab vs lib runtime, 8836 entries | 0 mismatches (max deviation from decimal 5.9e-8 = float32 rounding) |
| B parameters vs lib runtime, 59 functional/variant x 5 | 0 mismatches |
| C simple-dftd3 name vs generated parameters, 48 functionals | 0 disagree, max rel diff 0 (bit-identical) |
| D1 R0 from simple-dftd3 dimers, Z<=86 (7396 entries) | max rel 2.8e-12 (with CODATA2018 constant) |
| D3 R0 from the lib's own energies, all 8836 entries | max rel 2.6e-16 (vs float32 table) |
| E1 formula vs simple-dftd3 energy, Z<=86 | max rel 4.5e-11 |
| E2 formula (float32 tables) vs lib energy, Z<=86 | max rel 2.1e-7 (lib C6/C8 not bit-identical to oracle) |
| E3 decimal tables vs lib energy / lib bohr constant vs oracle | 2.2e-6 / 1.5e-6 (informative, see section 2) |
| D2 simple-dftd3 vs generated, any pair with Z>=87 | **1440 of 1440 entries differ (up to 29%)** |

Stage A (PBE, revPBE, BLYP, B3LYP, TPSS) values (def2-QZVP; s6 = 1, sr8 = 1, alpha6 = 14):
PBE sr6 = 1.217, s8 = 0.722 (**claim verified**: lib and simple-dftd3 agree); revPBE 0.923 / 1.010; BLYP (`b-lyp`) 1.094 / 1.682;
B3LYP (`b3-lyp`) 1.261 / 1.703; TPSS 1.166 / 1.105. Stage B (all 48 functionals) is in `data/d3zero_params.csv`, all covered by simple-dftd3 and agreeing.
Name mapping to simple-dftd3 is identity for the 48 names in the file.

## 5. Open issue: elements Z = 87-94
For every pair containing Z >= 87 (Fr-Pu) simple-dftd3's R0 differs from dftd3-lib's table (e.g. Fr-Fr 5.0796 A in the lib vs
5.7428 A recovered from simple-dftd3; Pu-Pu 4.4341 vs 4.4286 A; H-Fr 3.4155 vs 3.3286 A). The C6 also differs there (undamped
Fr-H dimer at 8 bohr: lib -6.473e-4 Eh, simple-dftd3 -6.833e-4 Eh), so the reference data of the two programs differ for Z >= 87
(not examined further). The generated table is the dftd3-lib one; which is right is NOT decided. NEP89 contains Ac, Th, Pa, U, Np, Pu
(Z = 89-94). Decision needed (owner): (a) restrict D3(0) to Z<=86 with an error for Z>=87, (b) adopt simple-dftd3 values
(would need extracting simple-dftd3's table; not done), or (c) check against the original DFTD3 V3.1 `pars.f`/paper supplement. The
existing BJ tables were presumably verified for Z<=86 only; not checked here.

## 6. Limitations
- Not read from the primary paper; two software sources only. TZVPP variants checked against the lib only.
- Two-body only; no three-body/ABC. Periodic and gradient paths not examined (dimer energies only).
- The lib's single-precision literals mean lib energies differ from the decimal published values by up to 2.2e-6; do not use the lib
  as a tight oracle (use simple-dftd3, bit-matching in C).
