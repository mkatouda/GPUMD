import json,sys
import numpy as np
sys.path.insert(0,sys.argv[2])   # tests_pytest directory (dftd2_reference.py)
from ase import Atoms
from ase.io import read
import dftd2_reference as R
B=0.52917726; EV=27.21138505
T2=sys.argv[1].rstrip('/')+'/'     # directory with the oracle outputs dftd3_gradient, dftd3_cellgradient of the 6-atom POSCAR case
def load(p): return np.array([[float(x.replace('E','e')) for x in l.split()] for l in open(p) if l.strip()])
# VER-0003 geometry: C C O H H H, s6 = float32(1.2), rc = 94.8683 bohr
cart=np.array([[0,0,0],[1.3,.2,.1],[2.6,3.1,2.2],[5,1,4],[6.1,5.2,7.7],[2,6,3.3]])
lat=np.array([[8,0,0],[.5,7.5,0],[.3,.4,9]])
a=Atoms('CCOHHH',positions=cart,cell=lat,pbc=True)
s6=float(np.float32(1.2)); rc=94.8683*B
cg=load(T2+'dftd3_cellgradient')*EV/B        # eV/A, dE/d lat[i][a]
E,F,W=R.energy_forces_virial(a,s6,rc,shifted=False)
Fo=-load(T2+'dftd3_gradient')*EV/B
print('E',E,'F rel',abs(F-Fo).max()/abs(Fo).max())
S_or=cg.T@lat    # S[a,b]=sum_i cg[i,a] lat[i,b]
print('oracle strain deriv S=cg^T lat\n',S_or,'\nanalytic -W\n',-W)
print('max|S_or+W|/max|W|',abs(S_or+W).max()/abs(W).max(),' transposed',abs(S_or.T+W).max()/abs(W).max())
# G_analytic at fixed frac: dE/dlat = (S) lat^-T  -> cg_an[i,a]=sum_b S[a,b] (lat^-1)[b,i]
cg_an=(-W)@np.linalg.inv(lat)           # [a,i]
print('cg direct compare', abs(cg_an.T-cg).max()/abs(cg).max())
# FD like VER-0003: h=1e-4 bohr on unshifted sharp-cutoff energy
h=1e-4*B
num=np.zeros((3,3))
for p in range(3):
    for q in range(3):
        def e(s):
            l=lat.copy(); l[p,q]+=s
            at=a.copy(); at.set_cell(l,scale_atoms=False); at.set_positions(a.get_scaled_positions(wrap=False)@l)
            return R.energy_forces_virial(at,s6,rc,shifted=False)[0]
        num[p,q]=(e(h)-e(-h))/(2*h)
print('FD (sharp) vs oracle rel', abs(num-cg).max()/abs(cg).max(), ' FD vs analytic', abs(num-cg_an.T).max()/abs(cg).max())
