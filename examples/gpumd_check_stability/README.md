# Run-time stability guard (`check_stability`)

Two carbon atoms with a deliberately deep Lennard-Jones well (5 eV, `lj.txt`) start 5.2 A apart and fall together.
Without the guard, the thermostat (`nvt_bdp`, 300 K) removes the gained kinetic energy within a few hundred steps, so the
temperature in `thermo.out` looks normal again although the potential energy has dropped by about 2.5 eV/atom. The
`check_stability` line stops the run at the first check (every 10 steps; here step 30) at which `|U - U0|/N` exceeds 1 eV/atom or a force component
exceeds 5 eV/A, prints the step, criterion, observed value and threshold, writes the last three evaluated frames to
`stability_frames.xyz` and exits with a non-zero status. This is the expected result of this example.

Run in a scratch copy (the run writes `thermo.out` and `stability_frames.xyz`):

```bash
cp -r examples/gpumd_check_stability /tmp/gpumd_check_stability && cd /tmp/gpumd_check_stability
/path/to/gpumd
```

Remove the `check_stability` line to see the unguarded run: it ends normally.
See the manual page of `check_stability` for all criteria.
