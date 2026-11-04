"""Timing-only variants of the LARGE-box CN and extra kernels of src/force/dftd3.cu (VER-0015 stage 6a).
Never touches the repository: usage  patch_variants.py SRC_DFTD3_CU VARIANT OUT_CU
All variants except `base` and `nsenv` change the results; they are used for per-kernel timing only
(only the timing of the kernel named in the description is read).
  base         unchanged copy
  nsenv        environment variables D3_NS_CN / D3_NS_EX fix the threads per atom of the CN / extra kernel
               (1,2,4,8,16; block 64 for 1 else 128); unset = unchanged behaviour
  cn_noexp     CN kernel: exp(arg) -> (arg) (removes the exponential only)
  cn_const     CN kernel: accepted pair adds a constant (removes exp, sqrt, division and the g_type[n2] load;
               the traversal, minimum image and cutoff test stay) = traversal + acceptance only
  ex_noexp     extra kernel: exp(arg) -> (arg)
  ex_const     extra kernel: f12/f21 factor = dc * 0.01 (removes exp, sqrt, divisions, the type load;
               the g_dc6_sum/g_dc8_sum[n2] load, force and virial accumulation stay)
  cn_nomic     CN kernel: apply_mic call removed (non-periodic arithmetic cost; positions then wrong)
  cn_xf32      CN kernel: coordinate differences formed in float (float(g_x[n2]) - float(x1))
  ex_xf32      same for the extra kernel
  fused        CN kernel also does the pair work of the extra kernel (using the dc6/dc8 sums that are in memory,
               i.e. STALE: timing only); the extra kernel launches but returns at once
  cache        the CN kernel also stores the accepted partner indices of its (atom, sub) slot (48 per slot, clamped);
               the extra kernel loops over the stored list instead of the cell traversal (positions in double,
               as now).  Timing of CN + extra = cost of a realizable shared traversal (index cache)
  cachef       as cache but also stores the float r12 vector (minimum image applied) so that the extra kernel needs
               no position loads and no FP64 (more memory traffic)
  count        prints visited/accepted candidates per atom of the CN traversal once (first call)
"""
import sys

src, variant, out = sys.argv[1:4]
t = open(src).read()


def must(c, m):
    if not c:
        sys.exit('patch failed: ' + m)


def region(t, a, b):
    i, j = t.index(a), t.index(b)
    must(i < j, 'region ' + a)
    return i, j


CN0, CN1 = region(t, 'find_dftd3_coordination_number_large_box(', 'find_dftd3_force_large_box(')
# step back to the 'template' line / forward to next template
CN0 = t.rindex('template <int num_sub>', 0, CN0)
CN1 = t.rindex('template <int num_sub>', 0, CN1)
EX0 = t.index('find_dftd3_force_extra_large_box(')
EX0 = t.rindex('template <int num_sub>', 0, EX0)
EX1 = t.index('bool set_para(')
cn, ex = t[CN0:CN1], t[EX0:EX1]
EXP_CN = 'exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f))'
CN_STMT = 'cn_temp += 1.0f / (' + EXP_CN + ' + 1.0f);'
EXP_EX = 'exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f));'
LAUNCH_CN = 'find_dftd3_coordination_number_large_box<NS><<<num_blocks, block_size>>>('
LAUNCH_EX = 'find_dftd3_force_extra_large_box<NS><<<num_blocks, block_size>>>('


def put():
    global t
    t = t[:CN0] + cn + t[CN1:EX0] + ex + t[EX1:]


if variant == 'base':
    pass
elif variant == 'nsenv':
    must(t.count('switch (num_sub) {') == 3, 'switch count')
    parts = t.split('switch (num_sub) {')
    t = parts[0] + 'switch (ns_cn) {' + parts[1] + 'switch (num_sub) {' + parts[2] + 'switch (ns_ex) {' + parts[3]
    anchor = '  const int num_blocks = (N * num_sub + block_size - 1) / block_size;\n'
    must(t.count(anchor) >= 1, 'anchor')
    k = t.index('void DFTD3::compute_large_box(')
    k = t.index(anchor, k)
    add = '''  const int ns_cn = getenv("D3_NS_CN") ? atoi(getenv("D3_NS_CN")) : num_sub;
  const int ns_ex = getenv("D3_NS_EX") ? atoi(getenv("D3_NS_EX")) : num_sub;
  const int bs_cn = (ns_cn == 1) ? 64 : BLOCK_SIZE_LARGE_BOX;
  const int bs_ex = (ns_ex == 1) ? 64 : BLOCK_SIZE_LARGE_BOX;
  const int nb_cn = (N * ns_cn + bs_cn - 1) / bs_cn;
  const int nb_ex = (N * ns_ex + bs_ex - 1) / bs_ex;
'''
    t = t[:k + len(anchor)] + add + t[k + len(anchor):]
    must(t.count(LAUNCH_CN) == 1 and t.count(LAUNCH_EX) == 1, 'launch')
    t = t.replace(LAUNCH_CN, 'find_dftd3_coordination_number_large_box<NS><<<nb_cn, bs_cn>>>(')
    t = t.replace(LAUNCH_EX, 'find_dftd3_force_extra_large_box<NS><<<nb_ex, bs_ex>>>(')
elif variant == 'cn_noexp':
    must(cn.count(EXP_CN) == 1, 'cn_noexp')
    cn = cn.replace(EXP_CN, '(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f))'); put()
elif variant == 'cn_const':
    must(cn.count(CN_STMT) == 1, 'cn_const')
    cn = cn.replace(CN_STMT, 'cn_temp += 0.37f;'); put()
elif variant == 'ex_noexp':
    must(ex.count(EXP_EX) == 1, 'ex_noexp')
    ex = ex.replace(EXP_EX, '(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f));'); put()
elif variant == 'ex_const':
    a = ex.index('int atomic_number_2 = dftd3_para.atomic_number[g_type[n2]];')
    b = ex.index('float f12[3] =')
    ex = ex[:a] + '''float f12_factor = 0.01f * dc6_sum_plus_dc8_sum_1;
            float f21_factor = -0.01f * (g_dc6_sum[n2] + g_dc8_sum[n2]);
            ''' + ex[b:]
    put()
elif variant == 'cn_nomic':
    must(cn.count('apply_mic(box, x12, y12, z12);') == 1, 'cn_nomic')
    cn = cn.replace('apply_mic(box, x12, y12, z12);', ''); put()
elif variant in ('cn_xf32', 'ex_xf32'):
    r = cn if variant == 'cn_xf32' else ex
    for c in 'xyz':
        o = 'float %s12 = g_%s[n2] - %s1;' % (c, c, c)
        must(r.count(o) == 1, variant + c)
        r = r.replace(o, 'float %s12 = float(g_%s[n2]) - float(%s1);' % (c, c, c))
    if variant == 'cn_xf32':
        cn = r
    else:
        ex = r
    put()
elif variant == 'fused':
    # --- CN kernel gets the extra pair work
    must(cn.count('float* g_cn_weight)') == 1, 'fused params')
    cn = cn.replace('float* g_cn_weight)', '''float* g_cn_weight,
  const float* g_dc6_sum,
  const float* g_dc8_sum,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)''')
    cn = cn.replace('__shared__ float s_red[1][BLOCK_SIZE_LARGE_BOX];', '__shared__ float s_red[13][BLOCK_SIZE_LARGE_BOX];')
    cn = cn.replace('  float cn_temp = 0.0f;\n', '''  float cn_temp = 0.0f;
  float dc6_sum_plus_dc8_sum_1 = g_dc6_sum[n1_safe] + g_dc8_sum[n1_safe];
  float s_fx = 0.0f, s_fy = 0.0f, s_fz = 0.0f, s_sxx = 0.0f, s_sxy = 0.0f, s_sxz = 0.0f, s_syx = 0.0f,
        s_syy = 0.0f, s_syz = 0.0f, s_szx = 0.0f, s_szy = 0.0f, s_szz = 0.0f;
''')
    must(cn.count(CN_STMT) == 1, 'fused stmt')
    cn = cn.replace(CN_STMT, '''float cn_exp_factor = ''' + EXP_CN + ''';
            cn_temp += 1.0f / (cn_exp_factor + 1.0f);
            float f12_factor = cn_exp_factor * 16.0f * (R_cov_1 + R_cov_2);
            f12_factor /= (cn_exp_factor + 1.0f) * (cn_exp_factor + 1.0f) * d12 * d12_2;
            float f21_factor = f12_factor;
            f12_factor *= dc6_sum_plus_dc8_sum_1;
            f21_factor *= -(g_dc6_sum[n2] + g_dc8_sum[n2]);
            float f12[3] = {r12[0] * f12_factor, r12[1] * f12_factor, r12[2] * f12_factor};
            float f21[3] = {r12[0] * f21_factor, r12[1] * f21_factor, r12[2] * f21_factor};
            s_fx += f12[0] - f21[0];
            s_fy += f12[1] - f21[1];
            s_fz += f12[2] - f21[2];
            s_sxx += r12[0] * f21[0];
            s_sxy += r12[0] * f21[1];
            s_sxz += r12[0] * f21[2];
            s_syx += r12[1] * f21[0];
            s_syy += r12[1] * f21[1];
            s_syz += r12[1] * f21[2];
            s_szx += r12[2] * f21[0];
            s_szy += r12[2] * f21[1];
            s_szz += r12[2] * f21[2];''')
    old_tail = cn[cn.index('  if (num_sub > 1) {\n    s_red[0][tid] = cn_temp;'):]
    new_tail = '''  if (num_sub > 1) {
    s_red[0][tid] = cn_temp;
    s_red[1][tid] = s_fx; s_red[2][tid] = s_fy; s_red[3][tid] = s_fz;
    s_red[4][tid] = s_sxx; s_red[5][tid] = s_sxy; s_red[6][tid] = s_sxz;
    s_red[7][tid] = s_syx; s_red[8][tid] = s_syy; s_red[9][tid] = s_syz;
    s_red[10][tid] = s_szx; s_red[11][tid] = s_szy; s_red[12][tid] = s_szz;
    reduce_sub_large_box<num_sub, 13>(s_red, tid, sub);
  }
  if (!valid || sub != 0) {
    return;
  }
  if (num_sub > 1) {
    cn_temp = s_red[0][tid];
    s_fx = s_red[1][tid]; s_fy = s_red[2][tid]; s_fz = s_red[3][tid];
    s_sxx = s_red[4][tid]; s_sxy = s_red[5][tid]; s_sxz = s_red[6][tid];
    s_syx = s_red[7][tid]; s_syy = s_red[8][tid]; s_syz = s_red[9][tid];
    s_szx = s_red[10][tid]; s_szy = s_red[11][tid]; s_szz = s_red[12][tid];
  }
  g_cn[n1] = cn_temp;
  store_cn_weights(atomic_number_1, cn_temp, n1, g_cn_weight);
  g_fx[n1] += s_fx; g_fy[n1] += s_fy; g_fz[n1] += s_fz;
  g_virial[n1 + 0 * N] += s_sxx; g_virial[n1 + 1 * N] += s_syy; g_virial[n1 + 2 * N] += s_szz;
  g_virial[n1 + 3 * N] += s_sxy; g_virial[n1 + 4 * N] += s_sxz; g_virial[n1 + 5 * N] += s_syz;
  g_virial[n1 + 6 * N] += s_syx; g_virial[n1 + 7 * N] += s_szx; g_virial[n1 + 8 * N] += s_szy;
}

'''
    cn = cn.replace(old_tail, new_tail)
    # extra kernel returns at once
    j = ex.index('  static_assert(')
    ex = ex[:j] + '  if (N > 0) return;\n' + ex[j:]
    put()
    o = 'cn_weight.data());\n  switch'
    must(t.count(o) == 1, 'fused launch')
    t = t.replace(o, 'cn_weight.data(), dc6_sum.data(), dc8_sum.data(), force_per_atom.data(), '
                  'force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data());\n  switch')
elif variant in ('cache', 'cachef'):
    full = variant == 'cachef'
    pre = '''constexpr int D3_CAPS = 48;
constexpr size_t D3_SLOTS = 340000;
__device__ int d_list[D3_SLOTS * D3_CAPS];
__device__ int d_listn[D3_SLOTS];
__device__ float d_listr[D3_SLOTS * D3_CAPS * 3];
'''
    must(cn.count(CN_STMT) == 1, 'cache stmt')
    store = '''
            if (kk < D3_CAPS) {
              const size_t slot = (size_t(n1_safe) * num_sub + sub) * D3_CAPS + kk;
              d_list[slot] = n2;
''' + ('''              d_listr[slot * 3] = r12[0];
              d_listr[slot * 3 + 1] = r12[1];
              d_listr[slot * 3 + 2] = r12[2];
''' if full else '') + '''            }
            ++kk;'''
    cn = cn.replace(CN_STMT, CN_STMT + store)
    cn = cn.replace('  float cn_temp = 0.0f;\n', '  float cn_temp = 0.0f;\n  int kk = 0;\n', 1)
    cn = cn.replace('  if (num_sub > 1) {\n    s_red[0][tid] = cn_temp;',
                    '  if (valid) d_listn[n1 * num_sub + sub] = kk < D3_CAPS ? kk : D3_CAPS;\n'
                    '  if (num_sub > 1) {\n    s_red[0][tid] = cn_temp;', 1)
    a = ex.index('  for (int c = valid ? sub')
    skip = '          if (n1 == n2) {\n            continue;\n          }\n'
    b = ex.index(skip) + len(skip)
    hdr = '''  const int cnt = valid ? d_listn[n1 * num_sub + sub] : 0;
  for (int kk = 0; kk < cnt; ++kk) {
    {
      {
        for (int once = 0; once < 1; ++once) {
          const size_t slot = (size_t(n1) * num_sub + sub) * D3_CAPS + kk;
          const int n2 = d_list[slot];
'''
    ex = ex[:a] + hdr + ex[b:]
    if full:
        for c, i in (('x', 0), ('y', 1), ('z', 2)):
            o = 'float %s12 = g_%s[n2] - %s1;' % (c, c, c)
            must(ex.count(o) == 1, 'cachef ' + c)
            ex = ex.replace(o, 'float %s12 = d_listr[slot * 3 + %d];' % (c, i))
        ex = ex.replace('          apply_mic(box, x12, y12, z12);\n', '', 1)
    put()
    t = t.replace(cn, pre + cn, 1)
elif variant == 'count':
    must(cn.count('  const int num_z = 2 * z_lim + 1;\n') == 1, 'count a')
    cn = cn.replace('          if (n1 == n2) {\n            continue;\n          }\n',
                    '          ++vis;\n          if (n1 == n2) {\n            continue;\n          }\n', 1)
    cn = cn.replace('          if (d12_2 < rc * rc) {\n', '          if (d12_2 < rc * rc) {\n            ++acc;\n', 1)
    cn = cn.replace('  float cn_temp = 0.0f;\n', '  float cn_temp = 0.0f;\n  unsigned long long vis = 0, acc = 0;\n', 1)
    cn = cn.replace('  if (num_sub > 1) {\n    s_red[0][tid] = cn_temp;',
                    '  if (valid) { atomicAdd(&d_cnt[0], vis); atomicAdd(&d_cnt[1], acc); }\n  if (num_sub > 1) {\n    s_red[0][tid] = cn_temp;', 1)
    cn = '__device__ unsigned long long d_cnt[2];\n' + cn
    put()
    o = '#undef DFTD3_LAUNCH_CN_LARGE_BOX\n  GPU_CHECK_KERNEL\n'
    must(t.count(o) == 1, 'count launch')
    t = t.replace(o, o + '''  {
    static int once = 0;
    if (!once) {
      once = 1;
      cudaDeviceSynchronize();
      unsigned long long h[2];
      cudaMemcpyFromSymbol(h, d_cnt, sizeof(h));
      printf("D3COUNT N=%d num_sub=%d rc_ang=%g cells=%dx%dx%d visited/atom=%.1f accepted/atom=%.1f ratio=%.3f\\n", N,
             num_sub, rc_angular, num_bins_angular[0], num_bins_angular[1], num_bins_angular[2], double(h[0]) / N,
             double(h[1]) / N, double(h[1]) / double(h[0]));
    }
  }
''')
else:
    sys.exit('unknown variant')
open(out, 'w').write(t)
