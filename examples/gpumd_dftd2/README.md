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
the energy jumps when a pair crosses the cutoff, so the total energy of an NVE run shows small steps. The cutoff must be given
explicitly with these options and should equal the one used to generate the reference data. Run it with
`gpumd` after copying the directory and replacing `run.in` by `run_qe.in` (the program reads `run.in`).
