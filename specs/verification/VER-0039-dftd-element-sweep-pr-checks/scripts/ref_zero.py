import sys, pickle, importlib.util
sys.path.append('/home/katouda/miniforge3/envs/py312-gpumd/lib/python3.12/site-packages')
import common
R=common.R
spec=importlib.util.spec_from_file_location('gen',R+'/tests_pytest/fixtures/dftd3_zero/generate_reference.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
S=common.structures(); out={}
jobs=[(n,'pbe') for n in S]+[(n,f) for n in ['iso_r6_0','iso_r6_1','dense_r4.5_0'] for f in ['revpbe','blyp','b3lyp','tpss']]
for n,f in jobs:
    e,F,V=gen.dispersion(S[n],(12,6),f); out[(n,f)]=(e,F,V); print(n,f,e,flush=True)
pickle.dump(out,open('ref_zero.pkl','wb'))
