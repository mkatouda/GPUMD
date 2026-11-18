import sys, pickle, importlib.util
sys.path.append('/home/katouda/miniforge3/envs/py312-gpumd/lib/python3.12/site-packages')
import common
spec=importlib.util.spec_from_file_location('gen',common.R+'/tests_pytest/fixtures/dftd3/generate_reference.py')
gen=importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
S=common.structures(); out={}
for n in S: out[n]=gen.reference(S[n],(12,6))[:3]; print(n,flush=True)
pickle.dump(out,open('ref_bj.pkl','wb'))
