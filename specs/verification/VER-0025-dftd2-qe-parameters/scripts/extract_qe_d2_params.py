#!/usr/bin/env python3
"""Extract the QE DFT-D3 'version 2' parameters from Fortran sources (never by hand).

Usage: extract_qe_d2_params.py <dft-d3 source dir> <outdir> [label]
  <dft-d3 source dir> holds pars.f90, core.f90 (e.g. `git show qe-7.2:dft-d3/<f>` copies).
Writes <outdir>/qe_<label>_c6_first_point.csv  (94x94, Eh bohr^6, c6ab(i,j,1,1,1))
       <outdir>/qe_<label>_c6_first_point_cn.csv (CN of atom i of that line, 94x94)
       <outdir>/qe_<label>_r0ab_decimal.csv     (94x94, Angstrom, decimal as printed in setr0ab)
Emulates copyc6 (non-min/max branch) and limit(): line nn has iat=int(pars(kk+1)), jat=int(pars(kk+2));
iadr = 1 + number of 100s subtracted from iat, jadr likewise; c6ab(iat,jat,iadr,jadr,1)=pars(kk).
"""
import re, sys, os
import numpy as np

def strip_comments(txt):
    out = []
    for ln in txt.splitlines():
        i = ln.find('!')
        out.append(ln[:i] if i >= 0 else ln)
    return "\n".join(out)

def blocks(txt, name):
    """yield (a, b, [tokens]) for every `name(a:b)=(/ ... /)` block"""
    txt = strip_comments(txt).replace('&', ' ')
    pat = re.compile(name + r'\(\s*(\d+)\s*:\s*(\d+)\s*\)\s*=\s*\(/(.*?)/\)', re.S)
    for m in pat.finditer(txt):
        toks = [t.strip() for t in m.group(3).replace('\n', ' ').split(',') if t.strip()]
        yield int(m.group(1)), int(m.group(2)), toks

def fnum(tok):
    tok = tok.replace('_wp', '').replace('D', 'e').replace('d', 'e')
    return tok

def load_pars(path):
    txt = open(path, encoding='latin-1').read()
    npars = 161925
    vals = [None] * npars
    for a, b, toks in blocks(txt, 'pars'):
        assert len(toks) == b - a + 1, (a, b, len(toks))
        for k, t in enumerate(toks):
            assert vals[a - 1 + k] is None
            vals[a - 1 + k] = float(fnum(t))
    assert all(v is not None for v in vals)
    return np.array(vals)

def load_r0ab(path):
    txt = open(path, encoding='latin-1').read()
    i = txt.index('subroutine setr0ab')
    j = txt.index('end subroutine setr0ab')
    body = txt[i:j]
    toks_all = [None] * 4465
    for a, b, toks in blocks(body, 'r0ab'):
        assert len(toks) == b - a + 1, (a, b, len(toks))
        for k, t in enumerate(toks):
            assert toks_all[a - 1 + k] is None
            toks_all[a - 1 + k] = t
    assert all(t is not None for t in toks_all)
    return toks_all

def limit(iat, jat):
    iadr = jadr = 1
    while iat > 100:
        iat -= 100; iadr += 1
    while jat > 100:
        jat -= 100; jadr += 1
    return iat, jat, iadr, jadr

def main():
    src, out = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else 'qe'
    os.makedirs(out, exist_ok=True)
    pars = load_pars(os.path.join(src, 'pars.f90'))
    nlines = len(pars) // 5
    assert nlines == 32385
    c6 = np.full((94, 94), -1.0); cn = np.full((94, 94), -1.0)
    nhit = np.zeros((94, 94), int)
    for nn in range(nlines):
        kk = nn * 5
        iat, jat, iadr, jadr = limit(int(pars[kk + 1]), int(pars[kk + 2]))
        if iadr == 1 and jadr == 1:
            for (a, b, ci) in ((iat, jat, pars[kk + 3]), (jat, iat, pars[kk + 4])):
                c6[a - 1, b - 1] = pars[kk]; cn[a - 1, b - 1] = ci; nhit[a - 1, b - 1] += 1
    # every unordered pair appears exactly once in the file; diagonal once (symmetric store, same cell twice)
    for i in range(94):
        for j in range(94):
            exp = 2 if i == j else 1
            assert nhit[i, j] == exp, (i + 1, j + 1, nhit[i, j])
    assert np.all(c6 == c6.T)
    np.savetxt(os.path.join(out, f'qe_{label}_c6_first_point.csv'), c6, fmt='%.8e', delimiter=',',
               header='QE c6ab(i,j,1,1,1) in Hartree bohr^6 (as in pars.f90, double); rows/cols Z=1..94', comments='# ')
    np.savetxt(os.path.join(out, f'qe_{label}_c6_first_point_cn.csv'), cn, fmt='%.6f', delimiter=',',
               header='QE c6ab(i,j,1,1,2): CN of atom i (row) of the first reference point; Z=1..94', comments='# ')
    toks = load_r0ab(os.path.join(src, 'core.f90'))
    lit = [fnum(t) for t in toks]
    r = np.zeros((94, 94)); k = 0
    for i in range(94):
        for j in range(i + 1):
            r[i, j] = r[j, i] = float(lit[k]); k += 1
    np.savetxt(os.path.join(out, f'qe_{label}_r0ab_decimal.csv'), r, fmt='%.4f', delimiter=',',
               header='QE setr0ab r0ab in Angstrom (decimal as printed; 7.2: real(4) literal -> float32 at run time); Z=1..94', comments='# ')
    print(f'{label}: pars lines {nlines}, r0ab entries {len(toks)}, '
          f'literals with _wp: {sum("_wp" in t for t in toks)}')

if __name__ == '__main__':
    main()
