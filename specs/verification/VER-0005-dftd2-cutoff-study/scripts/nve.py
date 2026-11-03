"""NVE energy conservation, water 648, dt 0.5 fs, 2 ps. usage: nve.py GPUMD_BIN NEP_TXT STRUCT_DIR WORKDIR rc [rc ...] ('0' = no D2)"""
import subprocess, sys
from pathlib import Path
import numpy as np
gpumd, nep, sdir, work = sys.argv[1:5]
for rc in sys.argv[5:]:
    d = Path(work) / f'nve_{rc}'; d.mkdir(parents=True, exist_ok=True)
    (d / 'model.xyz').write_text((Path(sdir) / 'water.xyz').read_text())
    L = [f'potential {nep}'] + ([f'dftd2 0.75 {rc}'] if rc != '0' else []) + \
        ['velocity 300 seed 20251103', 'ensemble nve', 'time_step 0.5', 'dump_thermo 10', 'run 4000']
    (d / 'run.in').write_text('\n'.join(L) + '\n')
    subprocess.run([gpumd], cwd=d, stdout=subprocess.DEVNULL, check=True)
    th = np.loadtxt(d / 'thermo.out'); e = th[:, 1] + th[:, 2]; t = np.arange(len(e)) * 10 * 0.5e-3  # ps
    slope = np.polyfit(t, e, 1)[0]
    print(f'rc={rc:>3s}: Etot mean {e.mean():.4f} eV, std {e.std():.5f} eV, drift {slope*1e3:.3f} meV/ps '
          f'(= {slope/648*1e3:.4f} meV/atom/ps), max-min {np.ptp(e):.5f} eV, T_end {th[-1,0]:.1f} K')
