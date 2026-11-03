import os,math
from lib import *
L={'w5184':37.2,'cu4000':36.1}
for s in ['w5184','cu4000']:
    print(s,'auto',['%.2f'%kern('it2',s,30)[0] for _ in range(2)],flush=True)
    for n in [3,4,5,6,7,8,10]:
        os.environ['NF']=str(n)
        print(s,'n',n,'%.2f'%kern('fit2',s,30)[0],flush=True)
