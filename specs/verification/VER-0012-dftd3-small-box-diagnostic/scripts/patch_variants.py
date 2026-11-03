"""Create timing-only variants of src/force/dftd3.cu (VER-0012). Never touches the repository.

usage: patch_variants.py SRC_DFTD3_CU VARIANT OUT_CU
Variants (all change results; for timing only):
  base        unchanged copy
  noc6        (d) C6 interpolation removed from the small-box force kernel: c6 = c6_ref[z12*max_cn2], dc6 = 0
  f32atomic   (e) the 15 double atomicAdd per pair in the small-box force kernel become float atomicAdd
  nlf32       (e) neighbor-list build: coordinate differences in float (position loaded as double, cast first)
  n1only      (e) force kernel: only the 3 atomics on atom n1 kept (n2 reaction force and virial atomics removed)
  regacc      (e) force kernel: no pair-loop atomics at all; n1 force summed in registers, written once
              (upper bound of an atomic-free pair loop)
  noc6L       (d) as noc6 for the LARGE-box force kernel (find_dftd3_force_large_box)
  xf32L       (e) large-box force kernel: coordinate differences g_x[n2] - x1 formed in float
  empty       (f) all four small-box kernels return immediately (launch/latency floor of one D3 step)
  count       prints N, candidates per atom and NN_radial/NN_angular statistics once (small-box path)
"""
import re
import sys

src, variant, out = sys.argv[1:4]
t = open(src).read()
F0 = t.index('void __global__ add_dftd3_force_small_box(')
F1 = t.index('void __global__ add_dftd3_force_extra_small_box(')
force = t[F0:F1]


def must(cond, msg):
    if not cond:
        sys.exit('patch failed: ' + msg)


if variant == 'base':
    pass
elif variant == 'noc6':
    old = 'if (num_cn_1 == 1 && num_cn_2 == 1) {'
    must(force.count(old) == 1, 'noc6')
    force = force.replace(old, 'if (true) {')
elif variant == 'f32atomic':
    force, n = re.subn(r'atomicAdd\(&(g_\w+)\[(.+?)\], double\((.+?)\)\);',
                       r'atomicAdd(reinterpret_cast<float*>(\1) + (\2), float(\3));', force)
    must(n == 15, 'f32atomic count %d' % n)
elif variant == 'n1only':
    lines = [ln for ln in force.split('\n')
             if not ('atomicAdd(' in ln and 'n1]' not in ln)]
    force = '\n'.join(lines)
    must(force.count('atomicAdd(') == 3, 'n1only')
elif variant == 'regacc':
    res = []
    for ln in force.split('\n'):
        m = re.match(r'(\s*)atomicAdd\(&g_f([xyz])\[n1\], double\((f12\[\d\])\)\);', ln)
        if m:
            res.append('%sacc_%s += %s;' % (m.group(1), m.group(2), m.group(3)))
        elif 'atomicAdd(' in ln:
            continue
        else:
            res.append(ln)
    force = '\n'.join(res)
    force = force.replace('    float s_potential = 0.0f;\n',
                          '    float s_potential = 0.0f;\n    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;\n', 1)
    force = force.replace('    g_potential[n1] += s_potential;\n',
                          '    g_fx[n1] += acc_x;\n    g_fy[n1] += acc_y;\n    g_fz[n1] += acc_z;\n'
                          '    g_potential[n1] += s_potential;\n', 1)
    must('acc_x += ' in force and 'g_fx[n1] += acc_x' in force and 'atomicAdd(' not in force, 'regacc')
elif variant == 'nlf32':
    for c in 'xyz':
        old = '%s12 = g_%s[n2] + delta[%d] - %s1;' % (c, c, 'xyz'.index(c), c)
        must(t.count(old) == 1, 'nlf32 ' + c)
        t = t.replace(old, '%s12 = float(g_%s[n2]) + delta[%d] - %s1;' % (c, c, 'xyz'.index(c), c))
elif variant in ('noc6L', 'xf32L'):
    L0 = t.index('find_dftd3_force_large_box(')
    L1 = t.index('find_dftd3_force_extra_large_box(')
    must(L0 < L1, 'large region')
    reg = t[L0:L1]
    if variant == 'noc6L':
        must(reg.count('if (num_cn_1 == 1 && num_cn_2 == 1) {') == 1, 'noc6L')
        reg = reg.replace('if (num_cn_1 == 1 && num_cn_2 == 1) {', 'if (true) {')
    else:
        for c in 'xyz':
            old = 'float %s12 = g_%s[n2] - %s1;' % (c, c, c)
            must(reg.count(old) == 1, 'xf32L ' + c)
            reg = reg.replace(old, 'float %s12 = float(g_%s[n2]) - float(%s1);' % (c, c, c))
    t = t[:L0] + reg + t[L1:]
elif variant == 'empty':
    for name in ['void __global__ find_dftd3_coordination_number_small_box(',
                 'void __global__ add_dftd3_force_extra_small_box(',
                 'static __global__ void find_neighbor_list_small_box(']:
        i = t.index(name)
        j = t.index('{\n', t.index(')', t.index('g_', i)))  # first '{' after the parameter list
        # parameter lists contain no '{', so the first '{\n' after the name is the body
        j = t.index('{\n', i)
        t = t[:j + 2] + '  if (N > 0) return;\n' + t[j + 2:]
    F0 = t.index('void __global__ add_dftd3_force_small_box(')
    F1 = t.index('void __global__ add_dftd3_force_extra_small_box(')
    force = t[F0:F1]
    j = force.index('{\n')
    force = force[:j + 2] + '  if (N > 0) return;\n' + force[j + 2:]
elif variant == 'count':
    old = '''    GPU_CHECK_KERNEL
  }

  find_dftd3_coordination_number_small_box<<<'''
    must(t.count(old) == 1, 'count')
    new = '''    GPU_CHECK_KERNEL
    static int once = 0;
    if (!once) {
      once = 1;
      std::vector<int> a(N), b(N);
      NN_radial.copy_to_host(a.data());
      NN_angular.copy_to_host(b.data());
      long sa = 0, sb = 0;
      int ma = 0, mb = 0;
      for (int i = 0; i < N; ++i) {
        sa += a[i]; sb += b[i];
        ma = std::max(ma, a[i]); mb = std::max(mb, b[i]);
      }
      printf("D3COUNT N=%d cells=%dx%dx%d cand_per_atom=%ld rc=%g/%g mean_rad=%.2f max_rad=%d mean_ang=%.2f max_ang=%d MN=%d\\n",
             N, ebox.num_cells[0], ebox.num_cells[1], ebox.num_cells[2],
             (long)N * ebox.num_cells[0] * ebox.num_cells[1] * ebox.num_cells[2],
             rc_radial, rc_angular, double(sa) / N, ma, double(sb) / N, mb, MN);
    }
  }

  find_dftd3_coordination_number_small_box<<<'''
    t = t.replace(old, new)
else:
    sys.exit('unknown variant')

t = t[:F0] + force + t[F1:]
open(out, 'w').write(t)
