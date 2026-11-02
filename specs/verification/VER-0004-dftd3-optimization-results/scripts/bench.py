import sys, os, shutil, subprocess, csv, io
S = '/tmp/claude-1001/-home-katouda-work-dielectric-response-pj-GPUMD/e8515210-3788-4777-872c-6503ac9338a6/scratchpad/p5/'
NEP = '/home/katouda/work/dielectric-response_pj/GPUMD/potentials/nep/nep89_20250409/nep89_20250409.txt'
NSYS = '/usr/local/cuda/bin/nsys'
binary, tag, xyz = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = 100
d = S + 'run_' + tag + '_' + xyz[:-4]
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
shutil.copy(S + xyz, d + '/model.xyz')
open(d + '/run.in', 'w').write(
    f'potential {NEP}\ndftd3 pbe 12 6\nvelocity 300\nensemble nve\ntime_step 0.5\nrun {STEPS}\n')
subprocess.run([NSYS, 'profile', '-o', 'prof', '--force-overwrite', 'true', '--stats=false', S + binary],
               cwd=d, capture_output=True, text=True)
r = subprocess.run([NSYS, 'stats', '-r', 'cuda_gpu_kern_sum', '--format', 'csv', 'prof.nsys-rep'],
                   cwd=d, capture_output=True, text=True).stdout
rows = list(csv.DictReader(io.StringIO(r[r.index('Time (%)'):])))
tot = 0
for x in rows:
    n = x['Name'].split('(')[0].replace('<unnamed>::', '')
    if 'dftd3' in n or 'cell' in n:
        t = float(x['Total Time (ns)']) / STEPS / 1e6
        tot += t
        print(f"{tag:6s} {xyz:12s} {n[:60]:60s} ms/step={t:.4f}")
print(f"{tag:6s} {xyz:12s} SUM(dftd3*+cell) ms/step={tot:.4f}")
shutil.rmtree(d, ignore_errors=True)
