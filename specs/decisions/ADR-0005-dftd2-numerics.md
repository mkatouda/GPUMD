# ADR-0005: Numerical tolerance for the dftd2 speed-up (DR-004 R3)

- Status: Accepted (option (a) approved by the human on 2026-11-06)
- Requirement: `DR-004` (R3, AC-2, AC-7); related `DR-002` AC-1, `ADR-0002` (same method for D3)

## Context
The `dftd2` pair kernel accumulates energy, force and 9 virial components per pair in double
precision. On GA102 (FP64 = 1/64 FP32) this dominates the run time: replacing the 13 accumulators by
float (timing-only variant, VER-0026 sec. 4) cut the kernel time by 83% at N = 41,472. A conforming
variant changes the order and precision of the sums, hence the last digits of the results of EVERY
`dftd2` input (also the default `table grimme2006 trunc sf`), which conflicts with the wording
"existing inputs give identical results" and with the bit-identity criterion DR-004 AC-2. The same
situation arose for D3 (DR-001) and was resolved by ADR-0002.

## Decision
1. `dftd2` results may change at the single-precision round-off level when the accumulation
   changes. They must stay within the tolerances of `DR-002` AC-1 against the independent
   double-precision numpy reference (`tests_pytest/dftd2_reference.py`) and against the
   requester's reference data (`tests_pytest/fixtures/dftd2_qe`): energy and virial 2e-6 relative
   (of |E_ref| and max|W_ref|), forces 1e-5 of max|F_ref|. The caps are not loosened.
2. Reference values are never regenerated to make a test pass: the numpy references and the
   requester's references stay fixed (their provenance is recorded).
3. Precision is not lowered below float partial sums that are converted to double and combined in
   a fixed order; no half precision, no fast-math flags. Compensated or segmented float
   accumulation (flush of per-thread partial sums to double after a bounded number of terms) is
   allowed and is chosen from the measured errors (margins to the caps are reported for the worst
   cases Rn-Rn, H-H, PTFE, water and replicated cells).
4. Determinism is kept: no atomics on forces, a fixed reduction order, run-to-run bit identity.
5. The frozen-output tests of the default form (bit identity with the pre-change binary, DR-004
   AC-2) are valid for the R1/R2 change (verified against the binary built from develop af269060,
   VER-0026) and are replaced, for the R3 change, by (a) a tolerance comparison of the new binary
   against the OLD frozen outputs at the AC-1 level and (b) new frozen outputs of the new binary
   (bitwise run-to-run reproducibility). Regenerating the frozen outputs is approved ONCE for this
   purpose (human decision 2026-11-06; reason: the accumulation precision changes the last digits);
   it is not a precedent.
6. Acceptance of the change: the accurate variant is kept only if (i) all accuracy criteria above
   are met and (ii) it makes the D2 kernel at least 20% faster at N = 41,472 (sm_86, nsys kernel
   time) and the D2 extra time per step at least 20% smaller; otherwise it is dropped, not
   relaxed. The target "dftd2 not slower than dftd3 at N >= 40,000" (DR-004 R3) stays informative
   (AC-7); A100 results are measured by the requester.
7. The kernel only changes; keyword, parser, tables and output stay as in ADR-0004.

## Consequences
Speed-ups of `dftd2` are checked against a fixed numerical oracle; a faster variant that exceeds the
tolerances is rejected. Bit identity with the old binary is no longer required after the R3 commit;
the pre-R3 state is the reference for the tolerance comparison.
