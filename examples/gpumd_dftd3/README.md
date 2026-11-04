# NEP89 with the DFT-D3(BJ) dispersion correction

This example runs 100 NVE steps (0.5 fs) of 27 water molecules (81 atoms, 9.3 A cubic box) with the NEP89 model
(`../../potentials/nep/nep89_20250409/nep89_20250409.txt`) plus the DFT-D3 pairwise dispersion correction with
Becke-Johnson damping (`dftd3 pbe 12 6`: PBE parameters, potential cutoff 12 Angstrom, coordination-number cutoff 6 Angstrom).

`dftd3` is an external correction: use it only with a model whose training data do not already contain dispersion
(see the manual page of `dftd3`). The functional and the cutoffs in this example are illustrative, not recommendations.
The cost grows roughly with the cube of the cutoff; see the manual page for measured timings.

Run (preferably in a copy of this directory, because the run writes `thermo.out`, `dump.xyz` and other output files):

```bash
cp -r examples/gpumd_dftd3 /tmp/gpumd_dftd3 && cd /tmp/gpumd_dftd3
/path/to/gpumd
```

Note on the temperature: `velocity 300` sets the initial temperature to 300 K, but the starting structure in `model.xyz` is not
relaxed for this model, so the potential energy is converted into kinetic energy and the temperature of this 81-atom box rises to
about 600-700 K within the first 100 steps (50 fs). This is a short demonstration of the input, not an equilibrated simulation.
The total energy (kinetic + potential, columns 2 and 3 of `thermo.out`) stays constant within about 0.06 eV in the NVE run.
For production use, equilibrate first (for example with an NVT or NPT run) and then use the equilibrated structure.

To obtain the contribution of the correction alone, run once with and once without the `dftd3` line
and subtract the energies and forces (use `time_step 0` and `run 1` for a single-point comparison).
