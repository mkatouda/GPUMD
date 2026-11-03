import sys, numpy as np
from ase.io import read
from ase.build import bulk
W='/home/katouda/work/dielectric-response_pj/GPUMD/.claude/worktrees/d2-phase6/tests_pytest/fixtures/dftd2/water27.xyz'
def dump(name, at, zs):
    L=np.diag(at.cell.array)
    t=[zs.index(z) for z in at.get_atomic_numbers()]
    with open(f'sys/{name}.txt','w') as f:
        f.write(f"{len(at)}\n{' '.join('%.10f'%l for l in L)}\n{' '.join(map(str,zs))}\n")
        for ti,p in zip(t,at.positions): f.write("%d %.8f %.8f %.8f\n"%(ti,*p))
w=read(W)
for name,rep in [('w81',(1,1,1)),('w324',(2,2,1)),('w648',(2,2,2)),('w2025',(5,5,1)),('w5184',(4,4,4)),('w20736',(8,8,4))]:
    a=w.repeat(rep); dump(name,a,[8,1]); print(name,len(a))
cu=bulk('Cu','fcc',a=3.61,cubic=True).repeat((10,10,10)); dump('cu4000',cu,[29]); print(len(cu))
