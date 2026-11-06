# ADR-0003: Optional damping selector for the `dftd3` keyword

- Status: Accepted (approved by the human on 2026-11-06)
- Requirement: `DR-003` (design `DES-0003`)
- Needed because: `AGENTS.md` makes changes to existing keywords an upstream contract
  change, and `ADR-0002` item 4 puts damping functions and keyword changes out of the D3
  optimization scope.

## Context
`dftd3 <functional> <rc_potential> <rc_cn>` currently means PBE-style BJ damping and requires
exactly 4 tokens (`NEP::initialize_dftd3`, `NEP_Charge::initialize_dftd3`). Zero damping
(D3(0)) must be added (DR-003) without changing any existing input or result.

## Decision
1. Syntax: `dftd3 <functional> <rc_potential> <rc_cn> [zero|bj]`. The token is case
   sensitive and lower case, like the functional names. Absent means `bj`.
2. Parsing: 4 tokens (BJ) or 5 tokens with a valid selector. A 5th token other than `zero`
   or `bj`, or more than 5 tokens, is an input error with a message that lists the valid
   values. The existing message for fewer than 3 parameters stays compatible in meaning.
3. Functional tables are per damping. A functional not tabulated for the chosen damping is an
   error that names both (`... not supported for DFT-D3 with <zero|BJ> damping`); there is no
   fallback to the other damping.
4. `bj` (explicit or default) must reproduce the previous binary's results bit for bit
   (DR-003 AC-3) AND its printed output unchanged: the header gets NO damping line for BJ.
   Only zero damping adds a line `damping = zero (D3(0))` after `functional =`; the existing
   lines keep their text and order. (Revised 2026-11-06 after the full regression run: a
   `damping = bj` line changed the stdout of `dftd3_duplicate` against the master baseline;
   decided by the human, option (a).)
5. The BJ code path is not altered numerically; the shared helper that selects the damping
   is verified for this by the bit-identity test.
6. Unchanged: `dftd3` may appear once; mutually exclusive with `dftd2`; rejected on
   `NEP_Charge_VDW` (`nep_vdw`, `qnep3_vdw`) and multi-GPU.
7. Numerical tolerances follow ADR-0002 (simple-dftd3 oracle, two-body, relative 2e-6 for
   energy and virial); the oracle for zero damping is simple-dftd3 with zero damping.
   The ATM three-body term remains out of scope (DR-003).

8. Functional names in the zero table use the same spelling as the BJ table (`blyp`,
   `b3lyp`, not `b-lyp`, `b3-lyp`), so that one name selects one functional for both dampings
   (decided 2026-11-06 on the Phase 2 test report).
9. Length constant: the Angstrom/bohr constant of `dftd3para.cuh` (`Bohr`) is used for both
   dampings; R0 is stored in Angstrom and converted with it. The simple-dftd3 oracle uses
   CODATA2018, a relative difference up to 1.5e-6 in the damping transition region, which is
   included in the Phase 4 BASELINE (decided 2026-11-06).

## Alternatives considered
- Functional names with a suffix (`pbe-zero`): rejected by the human on 2026-11-06 (decision
  D1 of DR-003); it would multiply table keys and break the one-name-per-functional rule.
- A new keyword (`dftd3_zero`): rejected; it would duplicate parsing and the mutual-exclusion
  checks.

## Consequences
Existing inputs are unaffected. Users select D3(0) by one trailing token. The manual
(`doc/gpumd/input_parameters/dftd3.rst`, `docs_ja/dftd/`) documents the token, the default,
and the two-body limitation. For an upstream PR the ADR is fork-only; the PR description
states the compatibility argument (default unchanged, bit-identical BJ).
