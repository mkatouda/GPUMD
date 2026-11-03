import sys, os, subprocess, numpy as np, json, time
from ase.io import read
from ase.build import bulk
W='/home/katouda/work/dielectric-response_pj/GPUMD/.claude/worktrees/d2-phase6c'
sys.path.insert(0, W+'/tests_pytest')
import dftd2_reference as ref
SC=os.path.dirname(os.path.abspath(__file__))
FIX=W+'/tests_pytest/fixtures/dftd2/'
def water(rep, tri=False):
    return read(FIX+('water27_tri.xyz' if tri else 'water27.xyz')).repeat(rep)
def cases():
    c={}
    c['w81']=lambda: water((1,1,1))
    c['w648']=lambda: water((2,2,2))
    c['cu4000']=lambda: bulk('Cu','fcc',a=3.61,cubic=True).repeat((10,10,10))
    def big(f):
        a=water((2,2,2)); a.set_cell(a.cell.array*f, scale_atoms=True); return a
    c['w648_x5']=lambda: big(5)
    c['w648_x3']=lambda: big(3)
    c['w648_tri']=lambda: water((2,2,2),True)
    def trix(f):
        a=water((2,2,2),True); a.set_cell(a.cell.array*f, scale_atoms=True); return a
    c['w648_tri_x4']=lambda: trix(4)
    def opn(sh):
        a=water((2,2,2)); a.pbc=False; a.positions+=sh; return a
    c['w648_open']=lambda: opn(0.0)
    c['w648_open_s100']=lambda: opn(100.0)
    def cu_s():
        a=bulk('Cu','fcc',a=3.61,cubic=True).repeat((10,10,10)); a.positions+=np.array([500.3,-720.1,333.3]); return a
    c['cu4000_shift']=cu_s
    def cu_r():
        a=bulk('Cu','fcc',a=3.61,cubic=True).repeat((10,10,10)); a.rattle(0.15,seed=1); return a
    c['cu4000_rat']=cu_r
    return c
def write_case(name, at):
    zs=sorted(set(at.get_atomic_numbers()))
    p=f'{SC}/cases/{name}.txt'
    H=at.cell.array.T  # columns are lattice vectors
    with open(p,'w') as f:
        f.write(f"{len(at)}\n"+' '.join('%.17g'%x for x in H.reshape(-1))+"\n"+' '.join('1' if x else '0' for x in at.pbc)+"\n")
        f.write(f"{len(zs)} "+' '.join(map(str,zs))+"\n")
        for z,pp in zip(at.get_atomic_numbers(), at.positions):
            f.write("%d %.17g %.17g %.17g\n"%(zs.index(z),*pp))
    return p
def run(binary, p, rc, N, tag):
    out=f'{SC}/out/{tag}.bin'
    subprocess.run([f'{SC}/bin/{binary}',p,out,str(rc),'1'],check=True)
    a=np.fromfile(out)
    pot=a[:N]; F=a[N:4*N].reshape(3,N).T; V=a[4*N:].reshape(9,N).sum(1).reshape(3,3)
    # order xx yy zz xy xz yz yx zx zy
    V=np.array([[a_ for a_ in r] for r in V]) if False else None
    v=a[4*N:].reshape(9,N).sum(1)
    W3=np.array([[v[0],v[3],v[4]],[v[6],v[1],v[5]],[v[7],v[8],v[2]]])
    return pot.sum(), F, W3
def main():
    names=sys.argv[1].split(','); rcs=[float(x) for x in sys.argv[2].split(',')]
    bins=sys.argv[3].split(',') if len(sys.argv)>3 else ['fa','fb']
    useref = os.environ.get('NOREF','0')!='1'
    cs=cases(); res=[]
    print('case rc | variant: dE/E  dF/Fmax  dW/Wmax (vs dbl GPU) | same vs NumPy ref | sumF/Fmax | ratio to AC-1 (E2e-6,F1e-5,W2e-6)')
    for n in names:
        at=cs[n](); N=len(at); p=write_case(n,at)
        for rc in rcs:
            Ed,Fd,Wd=run('dbl',p,rc,N,f'{n}_{rc}_dbl')
            if useref:
                t=time.time(); Er,Fr,Wr=ref.energy_forces_virial(at,0.75,rc); Wr=np.asarray(Wr)
            else: Er,Fr,Wr=Ed,Fd,Wd
            def met(E,F,W,E0,F0,W0):
                return (abs(E-E0)/abs(E0), np.abs(F-F0).max()/np.abs(F0).max(), np.abs(W-W0).max()/np.abs(W0).max())
            rows={'dbl':met(Ed,Fd,Wd,Er,Fr,Wr)}
            row={'case':n,'rc':rc,'N':N,'dbl_vs_ref':rows['dbl'],'sumF_dbl':float(np.abs(Fd.sum(0)).max()/np.abs(Fd).max()),'sumF_ref':float(np.abs(Fr.sum(0)).max()/np.abs(Fr).max())}
            print(f'{n} rc={rc} N={N}: dbl-vs-ref %.2e %.2e %.2e  sumF/Fmax dbl %.1e ref %.1e'%(*rows['dbl'],row['sumF_dbl'],row['sumF_ref']),flush=True)
            for b in bins:
                E,F,W_=run(b,p,rc,N,f'{n}_{rc}_{b}')
                vd=met(E,F,W_,Ed,Fd,Wd); vr=met(E,F,W_,Er,Fr,Wr)
                sf=float(np.abs(F.sum(0)).max()/np.abs(F).max())
                ratio=max(vr[0]/2e-6,vr[1]/1e-5,vr[2]/2e-6)
                print(f'   {b}: vs dbl %.2e %.2e %.2e | vs ref %.2e %.2e %.2e | sumF %.1e | AC1 frac %.3f'%(*vd,*vr,sf,ratio),flush=True)
                row[b]={'vs_dbl':vd,'vs_ref':vr,'sumF':sf,'ac1_frac':ratio}
            res.append(row)
    json.dump(res,open(f'{SC}/out/acc_{sys.argv[1].replace(",","_")}_{sys.argv[2].replace(",","_")}.json','w'))
main()
