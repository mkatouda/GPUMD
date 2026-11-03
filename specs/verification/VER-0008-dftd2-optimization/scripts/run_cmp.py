import sys
from lib import *
cand=sys.argv[1]; reps=int(sys.argv[2]) if len(sys.argv)>2 else 3
systems=sys.argv[3].split(',') if len(sys.argv)>3 else ['w648','w5184','cu4000']
rcs=[float(x) for x in sys.argv[4].split(',')] if len(sys.argv)>4 else [12,14,16,18,20,25,30]
print('sys rc base_ms cand_ms speedup',flush=True)
for s in systems:
  for rc in rcs:
    r=compare(['base2',cand],s,rc,reps)
    print(s,rc,'%.3f %.3f %.2fx'%(r['base2'][0],r[cand][0],r['base2'][0]/r[cand][0]),flush=True)
