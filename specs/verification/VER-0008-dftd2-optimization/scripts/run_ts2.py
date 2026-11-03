import os
from lib import *
print('sys rc TS time_ms')
for s in ['w81','w324']:
  for rc in [12,30]:
    for T in [1,2,4,8,16,32]:
        os.environ['TS']=str(T)
        print(s,rc,T,'%.4f'%kern('ts',s,rc)[0],flush=True)
