import pickle, numpy as np, common
from ase.data import chemical_symbols as cs
S=common.structures(); ref=pickle.load(open('ref_bj.pkl','rb')); M=str(common.HERE/'zero86.txt'); bad=0
for n,(e0,F0,V0) in ref.items():
    r,err=common.run_gpumd(S[n],'dftd3 pbe 12 6',M,'bjm'); e,F,V=r
    dF=np.abs(F-F0).max(axis=1); i=int(np.argmax(dF)); ok=abs(e-e0)/abs(e0)<=2e-6 and dF.max()<=1e-5 and np.abs(V-V0).max()/np.abs(V0).max()<=2e-6; bad+=not ok
    print('%-13s dE_rel %.1e dF %.1e (%s) dV_rel %.1e %s'%(n,abs(e-e0)/abs(e0),dF.max(),cs[S[n].numbers[i]],np.abs(V-V0).max()/np.abs(V0).max(),'ok' if ok else 'EXCEEDS'),flush=True)
print('exceeding',bad,'of',len(ref))
