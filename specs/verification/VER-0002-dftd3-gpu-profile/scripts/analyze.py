import csv, io, json, os, subprocess
NSYS = '/usr/local/cuda/bin/nsys'
STEPS = 100
cases = ['water81_12_6','water648_12_6','water5184_12_6','water41472_12_6','water5184_25_15','graphite4320_12_6','Cu4000_12_6']
natoms = {}
out = {}
for c in cases:
    d = f'p_{c}'
    natoms[c] = int(open(f'{d}/model.xyz').readline())
    t = subprocess.run([NSYS,'stats','--force-export=true','-r','cuda_gpu_kern_sum','--format','csv','prof.nsys-rep'],
                       cwd=d, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(t[t.index('Time (%)'):])))
    ms = lambda x: float(x['Total Time (ns)']) / STEPS / 1e6
    total = sum(ms(x) for x in rows)
    def is_d3(n):   # D3 kernels live in an anonymous namespace of dftd3.cu
        return 'dftd3' in n or n.startswith('<unnamed>::find_neighbor_list_small_box')
    k = {x['Name'].split('(')[0].replace('<unnamed>::',''): ms(x) for x in rows if is_d3(x['Name'])}
    d3 = sum(k.values())
    out[c] = dict(natoms=natoms[c], gpu_ms_per_step=total, d3_ms_per_step=d3, d3_share=d3/total,
                  kernels={a: round(b, 4) for a, b in sorted(k.items(), key=lambda kv: -kv[1])})
    top = list(out[c]['kernels'].items())[:3]
    print(f"{c:20s} N={natoms[c]:6d} GPU {total:6.2f} ms/step  D3 {d3:6.2f} ms ({100*d3/total:3.0f}%)  top: " + ', '.join(f'{a}={b:.2f}' for a, b in top))
json.dump(out, open('prof_summary.json','w'), indent=1)
