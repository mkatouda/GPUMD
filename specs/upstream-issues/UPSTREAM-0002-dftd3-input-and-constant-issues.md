# UPSTREAM-0002: DFT-D3 code (upstream behaviour): unvalidated input, uninitialized element lookup, constant provenance, int overflow

- Status: recorded 2026-11-04; NOT yet reported. Candidates for a later upstream bug report / fix PR (human decision 2026-11-04: document as known limitations in the Japanese manual, do not change code in this fork now).
- Fork-only note (do not include in an upstream issue/PR). Found while extracting facts for the manual (`docs_ja/dftd/`). All items are in upstream `master` as well unless noted; line numbers are for the fork's `develop` (cd0242e0) and master where stated. "Read" = by code reading only, NOT executed.

## A. find_atomic_number: unknown element symbol leaves atomic_number uninitialized (verified by reading, master and develop)
- `src/force/dftd3.cu` `find_atomic_number` (develop 384-415; master 413): the loop sets `atomic_number[n] = m` only on a match; the following check
  `if (atomic_number[n] >= max_elem)` reads the value even when no symbol matched. `DFTD3_Para::atomic_number` is `int atomic_number[94];` with no initializer (`dftd3.cuh:30`).
  The advertised error "DFTD3 only supports elements from H to Pu." therefore does not fire reliably for an unknown symbol (e.g. a typo), and the behaviour depends on indeterminate memory.
- Fix direction: initialize to -1, check after the inner loop, error naming the symbol.
- Check before filing: reproduce with a NEP file whose first line has an invalid symbol.

## B. HartreeBohr6 constant: provenance and a systematic offset (verified numerically; present in master)
- `src/force/dftd3para.cuh:24`: `#define HartreeBohr6 0.597527426643772f` (same in master).
- With CODATA 2018 (E_h = 4.3597447222071e-18 J, a0 = 0.529177210903e-10 m, e = 1.602176634e-19 C) E_h * a0^6 in eV A^6 is 0.5975269043; the code value is larger by about +5.2e-7 absolute, +8.7e-7 relative
  (so the D3 energies are scaled by this factor; the fork's AC-1 limit is 2e-6 relative, i.e. this alone uses about 44%). The source of the constant is not stated in code or comments.
- Fix direction: derive from documented CODATA constants (state the year) or document the origin. Changing the value changes results at the 1e-6 level (needs an ADR in this fork and a regression result).

## C. int overflow of N * MN in the small-box path (read, not executed; master and develop)
- `const int MN = 10000;` (develop line 46) and `const int size_x12 = N * MN;` (develop 1417, master 865); `size_x12 * 6` is used for `r12.resize(...)` and offsets. 2^31 / (MN * 6) is about 35,800 atoms: above that,
  `size_x12 * 6` overflows int (signed overflow, undefined). In practice the small-box path also needs about 320 kB per atom (N*MN*32 B), so it is not usable at that size anyway; the overflow is a latent problem for thin boxes with many atoms.
- Fix direction: size_t arithmetic; error when N * MN exceeds a limit.

## D. No check of the neighbor count against MN (read; master and develop)
- The small-box neighbor-list kernel writes up to MN entries per atom; a neighbor count above MN (large rc_pot, dense systems) is not detected (no error). Out-of-bounds writes are possible (not reproduced).
- Fix direction: count and report/abort when a count exceeds MN.

## E. No range validation of `dftd3 <functional> <rc_pot> <rc_cn>` cutoffs (read)
- Negative values, 0, and rc_cn > rc_pot are accepted. The small-box decision and the cell sizes use only rc_radial (= rc_pot); with rc_cn > rc_pot the CN counting may be incomplete (inference from the code, not executed).
- Fix direction: validate (> 0, rc_cn <= rc_pot or document the behaviour).

## F. NPT: possible change from the small-box to the large-box path leaves cell-list arrays unallocated (read only, master structure)
- When the box change moves a run from the small-box path to the large-box path, the cell-list arrays are guarded by `cn.size() != 0`-type checks that may skip their allocation. Not reproduced; may be a false alarm. Needs a run with strong compression/expansion across the 2.5*rc threshold.

## G. Other notes for the manual (behaviour, not bugs)
- `dftd3` is silently ignored with non-NEP potentials; `dftd2` (fork) raises an error. The element list is taken from the first potential file only.
- The per-atom virial of the CN term is assigned asymmetrically (partner's D_j only); the sum is correct.
- Large-box path: bit reproducibility is not guaranteed (shared `find_cell_list` uses atomicAdd).
- Sources of `dftd3para.cuh` tables (r2r4, covalent radii with the 4/3 factor, c6_ref units, 54 functional parameter sets) are not stated in the code.
