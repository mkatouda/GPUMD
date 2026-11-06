# NEP89 with the DFT-D2 dispersion correction

This example runs 100 NVE steps (0.5 fs) of 27 water molecules (81 atoms, 9.3 A cubic box) with the NEP89 model
(`../../potentials/nep/nep89_20250409/nep89_20250409.txt`) plus the DFT-D2 pairwise dispersion correction
(`dftd2 0.75 15`: scaling factor s6 = 0.75, cutoff 15 Angstrom; the default cutoff is 30 Angstrom).

`dftd2` is an external correction: use it only with a model whose training data do not already contain dispersion
(see the manual page of `dftd2`). The s6 value and the cutoff in this example are illustrative, not recommendations.

Run (preferably in a copy of this directory, because the run writes `thermo.out`, `dump.xyz` and other output files):

```bash
cp -r examples/gpumd_dftd2 /tmp/gpumd_dftd2 && cd /tmp/gpumd_dftd2
/path/to/gpumd
```

To obtain the contribution of the correction alone, run once with and once without the `dftd2` line
and subtract the energies and forces (use `time_step 0` and `run 1` for a single-point comparison).

## QE-compatible variant (`run_qe.in`)

`run_qe.in` uses `dftd2 0.75 12 table qe trunc plain`: the D2 functional form with the DFT-D3 tables of
Quantum ESPRESSO 7.2 (`dftd3_version = 2`: C6 of the first D3 reference point, R0 = 1.1 r0ab; elements up to Rn) and a
hard cutoff of 12 Angstrom, i.e. the setting for reproducing the D2 term of such calculations. With `trunc plain`
the energy jumps by |phi(rc)| (about 7e-6 eV for C-C at 12 A) and the force by about 6|phi(rc)|/rc (about 3.7e-6 eV/A) when a pair
crosses the cutoff, so the total energy of an NVE run shows small steps (no systematic drift is expected in equilibrium; a short NVE measurement found no visible drift: in a 20 ps NVE run of a 180-atom PTFE crystal at 300 K (NEP89 + ``dftd2 0.75 12 table qe``, 0.5 fs, 3 seeds per variant) the fitted slope of the total energy is |slope| <= 2e-4 meV/atom/ps for ``trunc plain`` and <= 1e-4 meV/atom/ps for ``trunc sf``, and the standard deviation of the total energy is 5.1-5.7e-4 eV for ``plain`` and 4.9-5.1e-4 eV for ``sf``; the single steps of 7e-6 eV are far below the step-to-step noise of the integration (median 9e-5 eV), so they are not resolved). Pairs at exactly rc are excluded (r < rc). `table qe` is consistent with QE 7.2
`dftd3_version = 2` to about 1e-6 relative (QE 7.2 stores r0ab as single-precision literals), verified against the requester's
QE 7.2 data and QE 7.6-dev (not run with a 7.2 binary); it is not the QE `vdw_corr = 'grimme-d2'` D2, and `dsd-blyp` (steepness 60)
cannot be reproduced. The cutoff must be given
explicitly with these options and should equal the one used to generate the reference data. Run it with
`gpumd` after copying the directory and replacing `run.in` by `run_qe.in` (the program reads `run.in`).
