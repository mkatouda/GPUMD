import os, re, shutil, subprocess, csv, io, json
from ase.io import read, write
from ase.build import bulk
GPUMD='/home/katouda/work/dielectric-response_pj/GPUMD/src/gpumd'
NEP='/home/katouda/work/dielectric-response_pj/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt'
NSYS='/usr/local/cuda/bin/nsys'
water=read('../water27.xyz'); gr=read('../graphite.xyz')
cases=[('water81_12_6',water,(12,6)),('water648_12_6',water.repeat((2,2,2)),(12,6)),
 ('water5184_12_6',water.repeat((4,4,4)),(12,6)),('water41472_12_6',water.repeat((8,8,8)),(12,6)),
 ('water5184_25_15',water.repeat((4,4,4)),(25,15)),
 ('graphite4320_12_6',gr.repeat((5,5,8)) if False else gr.repeat((4,4,8)),(12,6)),
 ('Cu4000_12_6',bulk('Cu','fcc',a=3.61,cubic=True).repeat((10,10,10)),(12,6))]
STEPS=100; out={}
for name,a,cs in cases:
    d=f'p_{name}'; shutil.rmtree(d,ignore_errors=True); os.makedirs(d); write(f'{d}/model.xyz',a)
    open(f'{d}/run.in','w').write(f'potential {NEP}\ndftd3 pbe {cs[0]} {cs[1]}\nvelocity 300\nensemble nve\ntime_step 0.5\nrun {STEPS}\n')
    subprocess.run([NSYS,'profile','-o','prof','--force-overwrite','true','--stats=false',GPUMD],cwd=d,capture_output=True,text=True)
    r=subprocess.run([NSYS,'stats','-r','cuda_gpu_kern_sum','--format','csv','prof.nsys-rep'],cwd=d,capture_output=True,text=True).stdout
    txt=r[r.index('Time (%)'):]
    rows=list(csv.DictReader(io.StringIO(txt)))
    tot=sum(float(x['Total Time (ns)']) for x in rows)
    d3=[x for x in rows if 'dftd3' in x['Name'] or 'find_neighbor_list_small_box' in x['Name'] and False]
    # classify D3 kernels by name; small-box neighbor kernel name is shared -> detect by instance count and name
    def isd3(n): return 'dftd3' in n
    kd=[(x['Name'].split('(')[0].replace('<unnamed>::',''),float(x['Total Time (ns)'])/STEPS/1e6,int(x['Instances'])) for x in rows if isd3(x['Name'])]
    d3t=sum(k[1] for k in kd); allt=tot/STEPS/1e6
    out[name]=dict(natoms=len(a),total_gpu_ms_per_step=allt,d3_ms_per_step=d3t,d3_share=d3t/allt,kernels=sorted(kd,key=lambda k:-k[1]))
    print(name,len(a),f'GPU {allt:.2f} ms/step, D3 {d3t:.2f} ms ({100*d3t/allt:.0f}%)',[(k[0],round(k[1],3)) for k in sorted(kd,key=lambda k:-k[1])[:4]],flush=True)
json.dump(out,open('prof_summary.json','w'),indent=1)
