import sys, os, subprocess, csv, io, statistics, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from acc_lib import *
NSYS='/usr/local/cuda/bin/nsys'
def mk(name):
    if name=='cu4000': return cases()['cu4000']()
    rep={'w81':(1,1,1),'w324':(2,2,1),'w648':(2,2,2),'w2025':(5,5,1),'w5184':(4,4,4),'w20736':(8,8,4)}[name]
    return water(rep)
def kern(binary,p,rc,it):
    out=f'{SC}/out/t_{binary}'
    subprocess.run([NSYS,'profile','-o',out,'--force-overwrite','true','-t','cuda',f'{SC}/bin/{binary}',p,f'{SC}/out/x.bin',str(rc),str(it)],capture_output=True,check=True)
    r=subprocess.run([NSYS,'stats','--force-export=true','-r','cuda_gpu_kern_sum','--format','csv',out+'.nsys-rep'],capture_output=True,text=True).stdout
    r=r[r.index('Time (%)'):]
    for row in csv.DictReader(io.StringIO(r)):
        if 'dftd2_pair_kernel' in row['Name']: return float(row['Med (ns)'])/1e6
systems=sys.argv[1].split(','); rcs=[float(x) for x in sys.argv[2].split(',')]; reps=int(sys.argv[3])
base,cand=sys.argv[4],sys.argv[5]
print('sys rc base_ms cand_ms speedup',flush=True)
for s in systems:
    at=mk(s); p=write_case('t_'+s,at)
    for rc in rcs:
        est=49*(len(at)/648)*(rc/30)**3+0.05; it=int(max(5,min(40,600/est)))
        r={base:[],cand:[]}
        for _ in range(reps):
            for b in (base,cand): r[b].append(kern(b,p,rc,it))
        mb,mc=statistics.median(r[base]),statistics.median(r[cand])
        print(s,rc,'%.3f %.3f %.2fx'%(mb,mc,mb/mc),flush=True)
