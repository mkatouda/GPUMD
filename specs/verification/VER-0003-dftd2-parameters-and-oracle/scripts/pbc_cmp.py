import numpy as np
from d2ref import *
sym=["C","C","O","H","H","H"]
cart=np.array([[0,0,0],[1.3,.2,.1],[2.6,3.1,2.2],[5,1,4],[6.1,5.2,7.7],[2,6,3.3]])/BOHR
lat=np.array([[8,0,0],[.5,7.5,0],[.3,.4,9]])/BOHR
frac=cart@np.linalg.inv(lat)
rc=94.8683; s6=1.2
z=[SYM.index(s) for s in sym]; n=6
def E_g(lat):
    x=frac@lat
    m=int(np.ceil(rc/ min(np.linalg.norm(lat,axis=1)) ))+2  # safe for this mildly skewed cell
    r_=np.arange(-m,m+1); T=np.array(np.meshgrid(r_,r_,r_)).reshape(3,-1).T@lat
    E=0.;g=np.zeros((n,3))
    for a in range(n):
        for b in range(a,n):
            d=x[a]-x[b]+T
            r=np.linalg.norm(d,axis=1)
            ok=(r<=rc)&(r>1e-9)
            d=d[ok];r=r[ok]
            c6=np.sqrt(C6[z[a]]*C6[z[b]])*c6conv; R=1.1*(R0[z[a]]+R0[z[b]])/BOHR
            e=np.exp(-20*(r/R-1)); f=1/(1+e)
            w=0.5 if a==b else 1.0
            E+=w*np.sum(-s6*c6*f/r**6)
            dE=-s6*c6*(20*e/R*f*f/r**6-6*f/r**7)
            gv=(w*dE/r)[:,None]*d
            if a!=b: g[a]+=gv.sum(0); g[b]-=gv.sum(0)
    return E,g
E,g=E_g(lat)
print("E",E,"eV",E*27.21138505)
gr=np.array([[float(v.replace('E','e')) for v in l.split()] for l in open('t2/dftd3_gradient')])
print("force max|d|",abs(g-gr).max(),"of",abs(gr).max())
cg=np.array([[float(v.replace('E','e')) for v in l.split()] for l in open('t2/dftd3_cellgradient')])
h=1e-4;num=np.zeros((3,3))
for i in range(3):
    for j in range(3):
        lp=lat.copy();lp[i,j]+=h;lm=lat.copy();lm[i,j]-=h
        num[i,j]=(E_g(lp)[0]-E_g(lm)[0])/(2*h)
print("cellgrad max|d| (rows=lat vecs)",abs(num-cg).max(),"of",abs(cg).max())
print("cellgrad max|d| transposed",abs(num.T-cg).max())
S=num.T@lat   # S[a,b]=sum_i num[i,a]*lat[i,b]
print("cg\n",cg); print("strain-deriv\n",S)
print("max|S-cg|",abs(S-cg).max()," max|S^T-cg|",abs(S.T-cg).max())
