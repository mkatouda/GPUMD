import pickle, numpy as np, common
S=common.structures(); ref=pickle.load(open('ref_zero.pkl','rb')); M=str(common.HERE/'zero86.txt')
CAP_REL,CAP_F=2e-6,1e-5; bad=0; rows=[]
for (n,f),(e0,F0,V0) in ref.items():
    r,err=common.run_gpumd(S[n],f'dftd3 {f} 12 6 zero',M,'zr')
    if r is None: print('FAIL RUN',n,f,err); bad+=1; continue
    e,F,V=r
    de=abs(e-e0)/abs(e0); dF=np.abs(F-F0).max(); dV=np.abs(V-V0).max()/np.abs(V0).max()
    ok=de<=CAP_REL and dF<=CAP_F and dV<=CAP_REL; bad+=not ok
    rows.append((n,f,de,dF,dV,ok))
    print('%-14s %-7s dE_rel %.1e dF %.1e (|F|max %.1e) dV_rel %.1e %s'%(n,f,de,dF,np.abs(F0).max(),dV,'ok' if ok else 'EXCEEDS'),flush=True)
print('worst: dE_rel %.1e dF %.1e dV_rel %.1e ; exceeding: %d of %d'%(max(r[2] for r in rows),max(r[3] for r in rows),max(r[4] for r in rows),bad,len(ref)))
