"""Apply pre-registered T1-T3 to results/static_cutoff.txt and estimate the tail beyond rc=40 (A*(rc^-3) model fitted at rc=30)."""
import sys
rows = {}
for l in open(sys.argv[1]):
    p = l.split()
    if len(p) == 11 and p[2].isdigit() and p[0] in ('water', 'graphite', 'mgo'):
        rows.setdefault(p[0], {})[int(p[2])] = [float(x) for x in p[3:]]
ok_all = {}
print('system rc  T1(dE<=0.5) T2(max<=5e-3 eV/A..., rms<=1) T3(dP<=0.02)')
for s, d in rows.items():
    for rc, v in d.items():
        if rc == 40: continue
        t1 = abs(v[1]) <= 0.5; t2 = v[2] <= 5.0 and v[3] <= 1.0; t3 = abs(v[4]) <= 0.02
        ok_all.setdefault(rc, []).append(t1 and t2 and t3)
        print(f'{s:9s}{rc:3d}  {t1!s:5}  {t2!s:5}  {t3!s:5}')
print('rc satisfying all systems:', [rc for rc, v in ok_all.items() if all(v)])
print('\nTail model: E(rc)-E(inf) ~ A rc^-3 (and P likewise), A from the rc=30 vs 40 difference')
for s, d in rows.items():
    k = 1 / 30**3 - 1 / 40**3
    A_e = d[30][1] / k; A_p = d[30][4] / k
    print(f'{s:9s} E(40)-E(inf)= {A_e/40**3:.3f} meV/atom, P(40)-P(inf)= {A_p/40**3*1e3:.3f} MPa;  '
          f'rc=25: dE_inf={A_e/25**3:.3f} meV, dP_inf={A_p/25**3*1e3:.2f} MPa; '
          f'rc=30: dE_inf={A_e/30**3:.3f} meV, dP_inf={A_p/30**3*1e3:.2f} MPa')
    chk = [d[rc][1] / (1/rc**3 - 1/40**3) for rc in (20, 25, 30)]
    print(f'          scaling check A(rc=20,25,30) = {chk[0]:.0f}, {chk[1]:.0f}, {chk[2]:.0f} meV A^3/atom')
