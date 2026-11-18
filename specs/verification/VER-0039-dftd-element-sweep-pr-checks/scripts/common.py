import itertools, re, subprocess, numpy as np
from pathlib import Path
from ase import Atoms
from ase.data import chemical_symbols
import os
R=os.environ.get('GPUMD_REPO') or str(Path(__file__).resolve().parents[4])   # repository root
import os
GPUMD=os.environ.get('GPUMD_BIN',f'{R}/src/gpumd')
HERE=Path(__file__).resolve().parent
ELEMS=list(range(1,87))
def zero_nep(path, mn=300):
    T=len(ELEMS); NEURONS=2; DIM=(1+1)+(1+1)*2
    N_ANN=T*(NEURONS*DIM+NEURONS+NEURONS)+1; N_DESC=(1+1)*(1+1)*T*T*2; N=N_ANN+N_DESC
    names=' '.join(chemical_symbols[z] for z in ELEMS)
    txt=f"""nep {T} {names}
format_version 1
model nep
cutoff 3 3 {mn} {mn}
n_max 1 1
basis_size 1 1
l_max 2 0 0
ann {NEURONS} 0
energy_reference {' '.join(['0']*T)}
@ parameters {N}
{chr(10).join(['0']*N)}
@ q_scaler {DIM}
{chr(10).join(['1']*DIM)}
"""
    Path(path).write_text(txt)
def cluster(seed, radius, dmin, cell=None):
    rng=np.random.default_rng(seed); pts=[]
    while len(pts)<len(ELEMS):
        x=rng.uniform(-radius,radius,3)
        if np.linalg.norm(x)<radius and all(np.linalg.norm(x-y)>dmin for y in pts): pts.append(x)
    order=rng.permutation(len(ELEMS))
    at=Atoms(numbers=[ELEMS[i] for i in order],positions=np.array(pts))
    if cell is None: at.center(vacuum=20.0); at.pbc=False
    else: at.set_cell([cell]*3); at.positions+=cell/2; at.pbc=True
    return at
def structures():
    s={}
    for k in range(10): s[f'iso_r6_{k}']=cluster(100+k,6.0,1.3)
    for k in range(5): s[f'dense_r4.5_{k}']=cluster(200+k,4.5,1.0)
    for k in range(3): s[f'pbc25_{k}']=cluster(300+k,7.0,1.3,cell=25.0)   # small-box path (thickness 25 <= 30)
    for k in range(3): s[f'pbc40_{k}']=cluster(400+k,7.0,1.3,cell=40.0)   # large-box path (thickness 40 > 30)
    return s
_cnt=itertools.count()
def run_gpumd(atoms, line, model, tag):
    from ase.io import write, read
    import tempfile
    d=Path(tempfile.mkdtemp(prefix=f'{tag}_',dir=os.environ.get('RUNS_DIR') or None))
    write(d/'model.xyz',atoms,format='extxyz')
    (d/'run.in').write_text(f'potential {model}\n{line}\nensemble nve\ntime_step 0\ndump_xyz 1 d.xyz force precision double\nrun 1\n')
    r=subprocess.run([GPUMD],cwd=d,capture_output=True,text=True)
    if r.returncode: return None, r.stdout[-400:]
    h=(d/'d.xyz').read_text().splitlines()[1]
    e=float(re.search(r'energy=([-0-9.eE+]+)',h).group(1))
    v=np.array(re.search(r'virial="([^"]+)"',h).group(1).split(),float).reshape(3,3)
    return (e,read(d/'d.xyz',format='extxyz').get_forces(),v),''
