import os,sys
from lib import *
# time vs forced n (cubic), visited cells per atom
import math
L={'w648':18.6,'w5184':37.2,'cu4000':36.1}
print('sys rc n k cells_per_atom time_ms')
for s in ['w648','w5184','cu4000']:
  for rc in [12,20,30]:
    for n in [1,2,3,4,5,6,8]:
        W=L[s]; k=math.ceil(rc/(W/n)*(1+1e-12))
        if n**3>max(1000,2*natoms(s)): continue
        os.environ['NF']=str(n)
        t=kern('fit',s,rc)[0]
        print(s,rc,n,k,(2*k+1)**3,'%.4f'%t,flush=True)
