import re, numpy as np
BOHR=0.52917726; KJ2EH=2625.4999
def load_tables(path):
    t=open(path).read()
    def arr(name):
        m=re.search(name+r"\(1:86\) = \(/(.*?)/\)",t,re.S)
        s=re.sub(r"&|\n|!.*?\n","",re.sub(r"![^\n]*","",m.group(1)))
        return np.array([float(x.replace('d','e')) for x in s.split(',') if x.strip()])
    return arr('r0'),arr('c6')
R0,C6=load_tables('src/prg/extras.f90')
assert len(R0)==86 and len(C6)==86
c6conv=1e-3/KJ2EH/((BOHR/10)**6)
SYM="H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
def d2(sym,xyz_bohr,s6,rc=94.8683,lat=None):
    z=[SYM.index(s) for s in sym]; n=len(z); x=np.asarray(xyz_bohr)
    E=0.;g=np.zeros((n,3))
    shifts=[np.zeros(3)]
    if lat is not None:
        m=int(np.ceil(rc/min(np.linalg.norm(lat,axis=1))))+1  # crude, cubic tests only
        shifts=[i*lat[0]+j*lat[1]+k*lat[2] for i in range(-m,m+1) for j in range(-m,m+1) for k in range(-m,m+1)]
    for a in range(n):
        for b in range(n):
            for t in shifts:
                if lat is None and b<=a: continue
                if lat is not None and (b<a or (a==b and not t.any())): continue
                d=x[a]-x[b]+t if lat is not None else x[a]-x[b]
                r=np.linalg.norm(d)
                if r>rc or r==0: continue
                c6=np.sqrt(C6[z[a]]*C6[z[b]])*c6conv
                R=1.1*(R0[z[a]]+R0[z[b]])/BOHR
                e=np.exp(-20*(r/R-1)); f=1/(1+e)
                E+=-s6*c6*f/r**6
                dE=-s6*c6*(20*e/R*f*f/r**6-6*f/r**7)  # dE/dr
                gv=dE*d/r
                g[a]+=gv; g[b]-=gv
    return E,g
if __name__=="__main__":
    sym=["C","H","H","O","H"]
    xyz=np.array([[0,0,0],[.63,.63,.63],[-.63,-.63,.63],[3.2,.3,-.2],[3.9,.9,.1]])/BOHR
    E,g=d2(sym,xyz,1.2)
    print("E",E, "kcal",E*627.509541)
    ref=np.array([[float(v.replace('E','e')) for v in l.split()] for l in open('t1/dftd3_gradient')])
    print("max|dg|",abs(g-ref).max(),"max|g|",abs(ref).max())
