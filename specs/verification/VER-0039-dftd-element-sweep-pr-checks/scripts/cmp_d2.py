import sys, numpy as np, common
sys.path.insert(0,common.R+'/tests_pytest')
import dftd2_reference as D2
S=common.structures(); M=str(common.HERE/'zero86.txt'); bad=0; rows=[]
for n,at in S.items():
    for s6,rc in ([(0.75,12)] if not n.startswith('iso_r6_0') else [(0.75,12),(0.75,30),(1.0,12)]):
        e0,F0,V0=D2.energy_forces_virial(at,s6,rc)
        r,err=common.run_gpumd(at,f'dftd2 {s6} {rc}',M,'d2')
        if r is None: print('FAIL RUN',n,s6,rc,err); bad+=1; continue
        e,F,V=r; de=abs(e-e0)/abs(e0); dF=np.abs(F-F0).max(); dV=np.abs(V-V0).max()/np.abs(V0).max()
        ok=de<=2e-6 and dF<=1e-5 and dV<=2e-6; bad+=not ok; rows.append((de,dF,dV))
        print('%-14s s6 %.2f rc %2d dE_rel %.1e dF %.1e (|F|max %.1e) dV_rel %.1e %s'%(n,s6,rc,de,dF,np.abs(F0).max(),dV,'ok' if ok else 'EXCEEDS'),flush=True)
print('worst: dE_rel %.1e dF %.1e dV_rel %.1e ; exceeding %d of %d'%(max(r[0] for r in rows),max(r[1] for r in rows),max(r[2] for r in rows),bad,len(rows)))
