import subprocess, csv, io, statistics, re, os
P=os.path.dirname(os.path.abspath(__file__))
NSYS='/usr/local/cuda/bin/nsys'
def natoms(sysn): return int(open(f'{P}/sys/{sysn}.txt').readline())
def niter(sysn, rc):
    est=49*(natoms(sysn)/648)*(rc/30)**3+0.05
    return int(max(5,min(40,600/est)))
def kern(binary, sysn, rc, it=None):
    it=it or niter(sysn,rc)
    out=f'{P}/nsys/t_{binary}'
    subprocess.run([NSYS,'profile','-o',out,'--force-overwrite','true','-t','cuda',f'{P}/bin/{binary}',f'{P}/sys/{sysn}.txt',str(rc),str(it)],capture_output=True,check=True,cwd=P)
    r=subprocess.run([NSYS,'stats','--force-export=true','-r','cuda_gpu_kern_sum','--format','csv',out+'.nsys-rep'],capture_output=True,text=True,cwd=P).stdout
    r=r[r.index('Time (%)'):]
    for row in csv.DictReader(io.StringIO(r)):
        if 'dftd2_pair_kernel' in row['Name']:
            return float(row['Med (ns)'])/1e6, int(row['Instances'])
def compare(bins, sysn, rc, reps=7):
    res={b:[] for b in bins}
    for _ in range(reps):
        for b in bins: res[b].append(kern(b,sysn,rc)[0])
    return {b:(statistics.median(v),min(v),max(v)) for b,v in res.items()}
