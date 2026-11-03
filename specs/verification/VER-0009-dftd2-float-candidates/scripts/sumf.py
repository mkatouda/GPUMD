import sys,pathlib,numpy as np
sys.path.insert(0,'/home/katouda/work/dielectric-response_pj/GPUMD/.claude/worktrees/d2-phase6e/tests_pytest')
import test_dftd2_reference as t
tag=sys.argv[1]
SP=pathlib.Path(sys.argv[2])
def sumf(name,at,rc):
    e0,f0,v0=t._run(SP/f'{tag}_{name}_0',at,None)
    e1,f1,v1=t._run(SP/f'{tag}_{name}_1',at,f'dftd2 {t.S6_GPU} {rc}')
    f=f1-f0
    print(tag,name,rc,'sumF/Fmax = %.2e'%(np.abs(f.sum(axis=0)).max()/np.abs(f).max()),flush=True)
for rc in (12.0,30.0):
    a=t._water((2,2,2),tri=True)
    sumf('tri',a,rc)
    b=a.copy(); b.positions+=np.array([13.3,-41.7,7.9])+0.5*a.cell.array.sum(axis=0)
    sumf('trishift',b,rc)
    sumf('far100',t._open_far_from_origin(),rc)
    sumf('far300',t._open_far_300(),rc)
