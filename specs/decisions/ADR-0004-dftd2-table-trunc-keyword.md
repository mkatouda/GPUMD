# ADR-0004: `table` and `trunc` options for the `dftd2` keyword

- Status: Accepted (approved by the human on 2026-11-06, including item 6: explicit cutoff required with `table qe` or `trunc plain`)
- Requirement: `DR-004` (design `DES-0004`; decisions P1-P5 of DR-004 approved 2026-11-06)
- Needed because: `AGENTS.md` treats changes to existing keywords as an upstream contract change
  (ADR-0003 handled the same question for `dftd3`).

## Context
Current syntax (`DFTD2::parse_tokens`, `src/force/dftd2.cu`): `dftd2 <s6> [<cutoff>]`; 2 or 3
tokens; `s6 > 0` finite; cutoff positive finite, default 30 Angstrom (DR-002 D1); shifted-force
truncation; Grimme 2006 C6/R0 table; once only; exclusive with `dftd3`. The parser is called by
`NEP` and `NEP_Charge`. DR-004 adds the QE `dftd3_version = 2` table (R1) and a plain cutoff (R2)
without changing existing inputs or results.

## Decision
1. Syntax: `dftd2 <s6> [<cutoff>] [table grimme2006|qe] [trunc sf|plain]`.
   - The optional options are keyword/value pairs after `s6`, in any order, each at most once.
   - `<cutoff>` stays positional and is recognised only as the 2nd token when it is a number;
     a 2nd token that is `table` or `trunc` starts the options. A number after an option keyword
     pair is an error.
   - Defaults: `table grimme2006`, `trunc sf`, cutoff 30 A: an input without options behaves
     exactly as before.
2. Parsing errors (informative messages that list the valid values): unknown `table` or `trunc`
   value, missing value after a keyword, repeated option, unknown token, more tokens than the
   grammar allows. Value matching is case sensitive, lower case (as ADR-0003). Existing messages
   for the legacy forms keep their meaning.
3. `table qe` (QE `dftd3_version = 2`, DR-004): C6_ij from the D3 table (first reference point of
   both atoms), R0_ij = `rs6 * r0ab_ij` with rs6 = 1.1, steepness 20 (as `D_DAMP`), r0ab in
   Angstrom from the table shared with DR-003; `s6` stays user-given (no functional table).
   `table grimme2006` is the existing behavior.
4. `table qe` is limited to Z <= 86; an atom type with Z >= 87 present in the structure is an
   error naming the element(s), checked at the first force evaluation (DR-002 D5, DR-003 D5).
   `table grimme2006` keeps its existing element rule.
5. `trunc plain` sets phi(rc) and phi'(rc) of the pair table to 0 (hard cutoff); `trunc sf` is the
   existing shifted-force truncation. Both apply to both tables.
6. Cutoff must be explicit with the new options: `table qe` or `trunc plain` without a numeric
   `<cutoff>` is an input error ("the cutoff must be given explicitly so that it matches the one
   used to generate the labels"). Rationale: the purpose of these options is to reproduce a
   specific label setting; the 30 A default belongs to the Grimme 2006 / shifted-force convergence
   study of DR-002 and would silently differ from QE-based labels. (Approved by the human.)
7. Header output: existing `dftd2` output lines keep their text and order. Lines for the new
   options (`table = qe`, `trunc = plain`) are printed only when the value is not the default, so
   the output of every existing input is byte-identical (same policy as ADR-0003 item 4).
8. Unchanged: once only; exclusive with `dftd3`; NEP, NEP_Charge, qNEP and `nep_vdw`/`qnep3_vdw`
   scope of DR-002 D4; multi-GPU rejection; DFT-D3 behavior.
9. Defaults are bit-identical: the default path (`grimme2006`, `sf`) must reproduce the pre-change
   binary's results bit for bit (DR-004 AC-2), verified by a frozen-output test as for DR-003.
10. Numerical tolerances for the new modes follow DR-002 AC-1 (float-level); the numpy reference
    `d2_reference.py` of the request and QE-printed D2 values are the oracles (DR-004 AC-1).

## Decision on the open point (2026-11-06)
Item 6 (explicit cutoff required with `table qe` or `trunc plain`) is approved as written. It is
stricter than the request; the alternative (default 30 A with documentation) was rejected because
of the risk of a silent mismatch with 12 A plain-cutoff labels.

## Alternatives considered
- `dftd2 <s6> <rc> table qe` with a mandatory cutoff for all forms (request wording): breaks
  existing `dftd2 <s6>` inputs; rejected.
- A new keyword (`dftd2_qe`): duplicates parsing and the exclusion checks; rejected.
- Functional-style names (`dftd2 pbe`): D2 has no functional table (DR-002 D2); rejected.

## Consequences
Existing inputs and outputs are unchanged. QE-label-consistent D2 is selected with
`dftd2 0.75 12 table qe trunc plain`. The manual (`doc/gpumd/input_parameters/dftd2.rst`,
`docs_ja/dftd/`) documents the options, the Z <= 86 limit, the explicit-cutoff rule and that QE
`dftd3_version = 2` is "D2 with D3 tables", not Grimme 2006. The ADR is fork-only; an upstream
PR states the compatibility argument (defaults unchanged, bit-identical output).
