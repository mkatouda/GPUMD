"""Fit GPUMD BEC - q*1 = a*T1 + b*T2 (T1 = sum_j rel_jk x dq_k/dr_j, T2 = sum_a rel_ak x dq_a/dr_k) using J.npy from bec_code_formula.py. Usage: bec_formula_fit.py MODEL.xyz NEP.txt J.npy WORKDIR (WORKDIR is created; runs gpumd 1 step). Code formula = (0.5, 0.5)."""
import os, sys, numpy as np, subprocess, pathlib, shutil
from ase.io import read
model, nep, Jp, work = sys.argv[1:5]
w = pathlib.Path(work); w.mkdir(exist_ok=True); shutil.copy(nep, w/'nep.txt')
a = read(model); 
txt = open(model).read().split('\n')
import re
# keep original model.xyz as is
shutil.copy(model, w/'model.xyz')
(w/'run.in').write_text("potential nep.txt\nkspace pppm\ntime_step 0.000001\nensemble nve\ndump_xyz 1 f.xyz precision double charge bec\nrun 1\n")
subprocess.run([os.environ.get('GPUMD_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'src', 'gpumd'))], cwd=w, capture_output=True)
N=len(a); L=(w/'f.xyz').read_text().split('\n')[2:2+N]
v=np.array([[float(t) for t in l.split()[1:]] for l in L]); q0=v[:,3]; b0=v[:,4:13]
J=np.load(Jp); cell=a.cell.array; inv=np.linalg.inv(cell); pos=a.get_positions()
d=pos[:,None,:]-pos[None,:,:]; f=d@inv; f-=np.round(f); rel=f@cell
T1=np.zeros((N,9)); T2=np.zeros((N,9)); D=np.zeros((N,9))
for k in range(N):
    T2[k]=np.einsum("ab,aA->bA",rel[:,k,:],J[:,k,:]).reshape(9)
    T1[k]=np.einsum("jb,jA->bA",rel[:,k,:],J[k,:,:]).reshape(9)
    D[k]=(q0[k]*np.eye(3)).reshape(9)
y=(b0-D).ravel(); X=np.stack([T1.ravel(),T2.ravel()],1)
c,res,_,_=np.linalg.lstsq(X,y,rcond=None); print('fit b0-D = a*T1 + b*T2: a,b =',c,' resid RMS',np.sqrt(((y-X@c)**2).mean()),' y RMS',np.sqrt((y**2).mean()))
print('RMS T1',np.sqrt((T1**2).mean()),'RMS T2',np.sqrt((T2**2).mean()))
for name,(aa,bb) in {'0.5/0.5':(.5,.5),'0.5/0':(.5,0),'0/0.5':(0,.5),'1/0':(1,0),'0/1':(0,1)}.items():
    print(name, np.sqrt(((y-aa*T1.ravel()-bb*T2.ravel())**2).mean()))
