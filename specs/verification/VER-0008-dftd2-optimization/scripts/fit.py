import numpy as np
N={'w648':648,'w5184':5184,'cu4000':4000}
rows=[l.split() for l in open('tab_fit.txt') if l.split()[0] in N]
S=[r[0] for r in rows]; n=np.array([int(r[2]) for r in rows]); k=np.array([int(r[3]) for r in rows]); t=np.array([float(r[5]) for r in rows])
Nn=np.array([N[s] for s in S]); cells=(2*k+1)**3.0; vis=Nn*cells/n**3
best=None
for C0 in [0,1,2,3,5,8,12,20,30,50,80]:
    for mode in ['all','sys']:
        cost=vis+C0*cells
        res=[]
        if mode=='all':
            A=np.exp(np.mean(np.log(t/cost))); err=np.log(t/(A*cost)); print(C0,'all  rms log err %.3f'%np.sqrt(np.mean(err**2)))
        else:
            e=[]
            for s in N:
                m=np.array([x==s for x in S]); A=np.exp(np.mean(np.log(t[m]/cost[m]))); e+=list(np.log(t[m]/(A*cost[m])))
            print(C0,'sys  rms log err %.3f'%np.sqrt(np.mean(np.array(e)**2)))
print('--- with accepted term')
RC=np.array([float(r[1]) for r in rows]); Vb={'w648':18.6**3,'w5184':37.2**3,'cu4000':36.1**3}
acc=np.array([N[s]*(4/3*np.pi*c**3)/Vb[s] for s,c in zip(S,RC)])
res=[]
for C0 in [0,1,2,3,5,8,12]:
  for Ca in [0,0.5,1,2,3,5,8]:
    cost=vis+C0*cells+Ca*acc; e=[]
    for s in N:
        m=np.array([x==s for x in S]); A=np.exp(np.mean(np.log(t[m]/cost[m]))); e+=list(np.log(t[m]/(A*cost[m])))
    res.append((np.sqrt(np.mean(np.array(e)**2)),C0,Ca))
for r in sorted(res)[:6]: print('%.3f C0=%g Ca=%g'%r)
