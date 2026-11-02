# VER-0003: dftd3-lib as DFT-D2 parameter source and numerical oracle

- Date: 2026-11-03. Status: verification record (not validation). Supports DR-002 / DES-0002.
- Subject: `/home/katouda/work/dielectric-response_pj/dftd3-lib` (commit `3c99ee9`, "DFTD3
  (deprecated)", a repackaging of Grimme's DFTD3 V3.1 Rev 1; license GPL v1 or later).
  The sources were copied to a scratch directory and built there with gfortran 11.4;
  the repository copy was not modified.

## 1. Where the D2 parameters are
- The D2 tables are `r0(1:86)` and `c6(1:86)` in `subroutine loadoldpar` of **`prg/extras.f90`**
  (the program), not in the library `lib/`. The library API (`api.f90`) copies the D3 tables
  (`copyc6`, `setr0ab`), so calling the API with D2 settings would NOT give D2.
- Conventions: C6 in J nm^6 mol^-1, converted by `c6conv = 1e-3/2625.4999/(0.52917726/10)^6`
  to hartree bohr^6; stored `r0` is in A and the published R0 = 1.1 * r0 (`rs6 = 1.1`);
  R0_ij = r0_i + r0_j; C6_ij = sqrt(C6_i C6_j); d = 20 (`alp`); E = -s6 sum C6 f / r^6.
  `s6` is a functional-dependent input (b-lyp 1.2, ...); the user will set it directly.
- Range: Z = 1..86 (H-Rn). `max_elem = 94` in the library, but no D2 values for Z > 86.

## 2. Cross-check of the values
- H-Xe (54 elements): C6 and 1.1*r0 compared with Table 6.3 of the Turbomole manual
  (https://extras.csc.fi/chem/progs/turbomole/DOC/DOC_HTML/node153.html), which reproduces
  Grimme, J. Comput. Chem. 27, 1787 (2006): **0 mismatches** (C6 exact, R0 within 6e-4 A,
  Sc-Zn and Y-Cd taken as the grouped values in that table).
- Z = 55-86 (Cs-Rn, includes La-Lu): single source only (dftd3-lib). Values are group fills
  (e.g. Ce-Lu share C6 = 140.68, r0 = 1.594; Ta-Au share 81.24 / 1.611). Not cross-checked.
  The VASP wiki lists no per-element table (only the H-Xe range, d = 20).
- NEP89 (89 elements, H..Pu): **83 covered** (Z <= 86 present in NEP89); **6 not covered**:
  Ac, Th, Pa, U, Np, Pu (Z = 89-94). Fr and Ra are not in NEP89.

## 3. Numerical oracle (D2 mode of `prg/dftd3 -old -grad [-pbc]`)
Independent NumPy implementation (`scripts/d2ref.py`, `scripts/pbc_cmp.py`) written from the
formulas above, using the same tabulated values (so the table itself is checked in section 2,
not here). Results with `-func b-lyp` (s6 = 1.2), default cutoff 94.8683 bohr:
| Case | Quantity | Max abs difference | Scale |
|---|---|---|---|
| 5-atom cluster (C,H,O) | energy | agrees at the printed 8 digits (-1.11847e-3 Eh) | |
| | gradient | 3.1e-11 Eh/bohr | max 7.8e-4 |
| triclinic cell, 6 atoms | energy | agrees at the printed 8 digits (-0.0264379 eV) | |
| | gradient | 6.6e-12 Eh/bohr | max 1.7e-4 |
| | `dftd3_cellgradient` vs finite difference of dE/d(lattice) | 1.6e-7 | max 1.9e-4 (**8e-4 relative**) |
- Energy and forces match to the precision the oracle prints (energy: 8 digits; gradient: 14).
- `dftd3_cellgradient` is dE/d(lattice vectors) (rows), not the strain derivative. It differs
  from the finite-difference value by 8e-4 relative; the cause is not identified (a cutoff
  or threshold handling is suspected). **Do not use the oracle's cell gradient as a virial
  reference for AC-1 (2e-6) until this is explained;** use a numerical strain derivative of
  the NumPy reference instead.
- The oracle cutoff is a sharp, unshifted truncation (r^2 > rthr skipped). The oracle cannot
  produce shifted-force results; use it at the same rc and apply the shift analytically
  (DES-0002 Phase 1).

## 4. Reproduce
Copy `dftd3-lib` to a scratch directory, `make dftd3` (re-run `make dftd3` once if the first
parallel build races), place `scripts/*.py` next to `src/`, run `pbc_cmp.py` with a Python that has numpy.
The two inputs are in section 3 (5-atom xyz and the 6-atom POSCAR are reproduced in the scripts' history; see
`scripts/pbc_cmp.py` for the cell and positions).
