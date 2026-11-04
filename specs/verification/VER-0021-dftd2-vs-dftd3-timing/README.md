# VER-0021: DR-002 AC-7 re-measurement (D2 extra <= 0.5 x D3 extra) against the Phase-7 optimized D3

- Date: 2026-11-04. Status: verification (timing, informative; NOT validation). Worktree `d2d3-ratio`, HEAD 8fe8ab99 (final D2 and D3 code), one binary
  (`make CUDA_ARCH="-arch=sm_86" -j4`), RTX A6000, CUDA 12.3, NEP89, one GPU job at a time. VER-0006/0008/0010 used the pre-Phase-7 D3.
- Method (as VER-0010/0018): slope = (t500 - t100)/400 from `Time used`; NVE, dt 0.5 fs, `velocity 300 seed 1`. Per repetition all variants are run in sequence
  (off, D3, D2-12, D2-30); extra = variant slope - off slope of the same repetition; 3 repetitions, median; scatter = [min, max].
  Systems: water27 (`tests_pytest/fixtures/dftd3/water27.xyz`) repeated (1,2,4,8)^3 = 81/648/5184/41472 atoms (as VER-0010), Cu fcc a = 3.61 A 10x10x10 (as VER-0018).
  D3 = `dftd3 pbe 12 6`; D2 = `dftd2 0.75 12` / `0.75 30`.
- Scripts: `scripts/ratio.py` (+ `scripts/run_all2.sh` for the re-run and cutoff scan). Results: `results/ratio_main.{json,log}`, `ratio_rerun.*`, `ratio_cut.*`, `tables.txt`.

## Main table (ms/step; extra = with correction minus without; [min,max] of 3)
| system | NEP only | D3 extra | D2(12) extra | D2(30) extra | D2(12)/D3 | D2(30)/D3 | AC-7 rc12 | AC-7 rc30 |
|---|---|---|---|---|---|---|---|---|
| water 81 | 1.395 | 0.061 [0.044,0.074] | 0.442 [0.332,0.596] | 0.950 [0.820,1.047] | 7.2 | 15.6 | not met (D3 at noise) | not met |
| water 648 | 1.554 | 0.543 [0.539,0.587] | 0.207 [0.127,0.292] | 1.865 [1.833,1.945] | 0.38 | 3.4 | met (D2 near noise) | not met |
| water 5184 | 1.784 | 0.841 [0.752,0.845] | 1.006 [0.980,1.019] | 9.894 [9.754,9.925] | 1.20 | 11.8 | not met | not met |
| water 41472 | 6.729 | 6.242 [6.199,6.303] | 7.023 [7.011,7.110] | 70.787 [70.388,70.832] | 1.13 | 11.3 | not met | not met |
| Cu 4000 | 1.296 | 0.498 [0.497,0.567] | 0.947 [0.913,0.965] | 8.852 [8.762,8.907] | 1.90 | 17.8 | not met | not met |

AC-7 as written (<= 0.5) is met only at water 648 with rc = 12 (and there the D2 extra, 0.13-0.29 ms, is within about 3x the scatter). After the D3 optimization D2/D3 at rc = 12 is
>= 1.1 for every large-box case, and D2 with the default rc = 30 is 3-18x D3 everywhere.

Noise: slope scatter is about 0.03-0.1 ms for most points (water 81: 0.03, water 648 D3 0.05, D2(12) 0.17). Extras below about 0.3 ms (3x scatter) are at noise level: water 81 D3 (0.06-0.13 ms), water 648 D2(12)
(0.13-0.29 ms); the ratios at N = 81 and N = 648 rc 12 are therefore not meaningful to better than a factor of ~2.

## Gate: consistency with earlier records and re-run
| point | this run | earlier | within 25 % |
|---|---|---|---|
| D2(30) 81/648/5184/41472 | 0.95 / 1.87 / 9.89 / 70.8 | VER-0010 0.90 / 1.81 / 10.01 / 68.6 | yes (<= 3.2 %) |
| D2(12) 81/648/5184/41472 | 0.44 / 0.21 / 1.01 / 7.02 | VER-0010 0.34 / 0.19 / 1.12 / 7.06 | 81 no (noise level, 0.1 ms), others yes |
| D3 648/5184/41472/Cu | 0.543 / 0.841 / 6.242 / 0.498 | VER-0018 0.459 / 0.846 / 6.193 / 0.554 | 648: 18 %, others <= 10 %: yes |
| D3 81 | 0.061 | VER-0018 0.152 | NO (but absolute difference 0.09 ms, noise) -> re-run |
Re-run (`results/ratio_rerun.*`, water 81, 648, Cu): D3 81 = 0.133 [0.064,0.163], D2(12) 81 = 0.421, D2(30) 81 = 0.993; D3 648 = 0.531 [0.441,0.645], D2(12) 648 = 0.234, D2(30) 648 = 1.901;
Cu: D3 0.521, D2(12) 0.927, D2(30) 8.753. Re-run medians agree with the first run within the scatter; the D3 81 value (0.06 / 0.13 ms; VER-0018 0.15) is at noise level, so no conclusion is drawn from that point.
Note the D2(12) extra at N = 81 (0.44) exceeds that at N = 648 (0.21); both are small and the 81-atom case is dominated by launch/host overhead; not investigated further.

## D2 cutoff dependence after optimization (D2 extra, ms/step, median of 3, [min,max]; `results/ratio_cut.*`)
| system | rc 12 (main) | rc 16 | rc 20 | rc 25 | rc 30 (main) |
|---|---|---|---|---|---|
| water 648 | 0.207 | 0.434 [0.402,0.542] | 1.284 [1.223,1.343] | 1.496 [1.425,1.530] | 1.865 |
| water 5184 | 1.006 | 2.394 [2.336,2.578] | 3.854 [3.823,3.901] | 6.846 [6.802,6.873] | 9.894 |
The scan is a single series of 3 repetitions (NEP-only baselines 1.540 / 1.752 ms).

## Limits
Single machine/GPU, medians of 3, scatter ~0.03-0.17 ms; the rc scan was not alternated with D3; slope method includes non-kernel overhead (host, array addition). No code change. Not run: other GPUs, HIP.
