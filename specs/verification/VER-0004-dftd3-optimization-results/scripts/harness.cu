#include "harness_kernels.inc"
#include <random>
#include <cstring>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(2);} } while (0)

struct Out {
  std::vector<int> nnr, nna, nlr, nla;
  std::vector<float> r12;
};

template <typename F>
static Out run(F launch, int N, const Box& box, const DFTD3::ExpandedBox& ebox, float rcr, float rca,
               const std::vector<double>& pos)
{
  size_t sz = (size_t)N * MN;
  double* dpos; int *nnr, *nna, *nlr, *nla; float* r12;
  CK(cudaMalloc(&dpos, sizeof(double) * 3 * N));
  CK(cudaMemcpy(dpos, pos.data(), sizeof(double) * 3 * N, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&nnr, sizeof(int) * N)); CK(cudaMalloc(&nna, sizeof(int) * N));
  CK(cudaMalloc(&nlr, sizeof(int) * sz)); CK(cudaMalloc(&nla, sizeof(int) * sz));
  CK(cudaMalloc(&r12, sizeof(float) * sz * 6));
  CK(cudaMemset(nnr, 0, sizeof(int) * N)); CK(cudaMemset(nna, 0, sizeof(int) * N));
  CK(cudaMemset(nlr, 0, sizeof(int) * sz)); CK(cudaMemset(nla, 0, sizeof(int) * sz));
  CK(cudaMemset(r12, 0, sizeof(float) * sz * 6));
  launch(rcr * rcr, rca * rca, N, box, ebox, dpos, dpos + N, dpos + 2 * N, nnr, nlr, nna, nla, r12, sz);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Out o;
  o.nnr.resize(N); o.nna.resize(N); o.nlr.resize(sz); o.nla.resize(sz); o.r12.resize(sz * 6);
  CK(cudaMemcpy(o.nnr.data(), nnr, sizeof(int) * N, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.nna.data(), nna, sizeof(int) * N, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.nlr.data(), nlr, sizeof(int) * sz, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.nla.data(), nla, sizeof(int) * sz, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.r12.data(), r12, sizeof(float) * sz * 6, cudaMemcpyDeviceToHost));
  cudaFree(dpos); cudaFree(nnr); cudaFree(nna); cudaFree(nlr); cudaFree(nla); cudaFree(r12);
  return o;
}

static bool test(const char* name, int N, const double h[9], int px, int py, int pz, float rcr, float rca,
                 unsigned seed)
{
  Box box;
  for (int i = 0; i < 18; ++i) box.cpu_h[i] = 0;
  for (int i = 0; i < 9; ++i) box.cpu_h[i] = h[i];
  box.pbc_x = px; box.pbc_y = py; box.pbc_z = pz;
  DFTD3::ExpandedBox ebox;
  bool small = get_expanded_box(rcr, box, ebox);
  if (!small) { printf("%s: not small box, skipped (harness bug)\n", name); return false; }
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> pos(3 * N);
  for (int n = 0; n < N; ++n) {
    double s[3] = {u(gen), u(gen), u(gen)};
    for (int d = 0; d < 3; ++d)
      pos[n + d * N] = h[d * 3 + 0] * s[0] + h[d * 3 + 1] * s[1] + h[d * 3 + 2] * s[2];
  }
  auto lo = [&](float a, float b, int NN, const Box& bx, const DFTD3::ExpandedBox& eb, const double* x,
                const double* y, const double* z, int* nnr, int* nlr, int* nna, int* nla, float* r,
                size_t sz) {
    oldk::find_neighbor_list_small_box<<<(NN - 1) / 64 + 1, 64>>>(a, b, NN, bx, eb, x, y, z, nnr, nlr, nna,
      nla, r, r + sz, r + 2 * sz, r + 3 * sz, r + 4 * sz, r + 5 * sz);
  };
  auto ln = [&](float a, float b, int NN, const Box& bx, const DFTD3::ExpandedBox& eb, const double* x,
                const double* y, const double* z, int* nnr, int* nlr, int* nna, int* nla, float* r,
                size_t sz) {
    newk::find_neighbor_list_small_box<<<NN, NL_BLOCK_SIZE>>>(a, b, NN, bx, eb, x, y, z, nnr, nlr, nna,
      nla, r, r + sz, r + 2 * sz, r + 3 * sz, r + 4 * sz, r + 5 * sz);
  };
  Out a = run(lo, N, box, ebox, rcr, rca, pos);
  Out b = run(ln, N, box, ebox, rcr, rca, pos);
  long cand = (long)N * ebox.num_cells[0] * ebox.num_cells[1] * ebox.num_cells[2];
  int maxnr = 0, maxna = 0;
  for (int n = 0; n < N; ++n) { maxnr = std::max(maxnr, a.nnr[n]); maxna = std::max(maxna, a.nna[n]); }
  bool ok = a.nnr == b.nnr && a.nna == b.nna && a.nlr == b.nlr && a.nla == b.nla &&
            std::memcmp(a.r12.data(), b.r12.data(), a.r12.size() * sizeof(float)) == 0;
  printf("%-28s N=%4d cells=%dx%dx%d candidates=%ld (mod128=%ld) max NN_rad=%d max NN_ang=%d : %s\n", name, N,
         ebox.num_cells[0], ebox.num_cells[1], ebox.num_cells[2], cand, cand % 128, maxnr, maxna,
         ok ? "BITWISE IDENTICAL (all 6 arrays, full buffers)" : "MISMATCH");
  return ok;
}

int main()
{
  bool ok = true;
  const double cubic10[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double cubic14[9] = {14, 0, 0, 0, 14, 0, 0, 0, 14};
  const double cubic8[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  const double tri[9] = {10, 3, 2, 0, 9, 4, 0, 0, 11}; // columns are the lattice vectors
  const double tri2[9] = {12, 5, -3, 0, 11, 4, 0, 0, 9};
  ok &= test("(a) multiple of 128", 64, cubic10, 1, 1, 1, 6.0f, 4.0f, 1);
  ok &= test("(a2) multiple of 128", 16, cubic10, 1, 1, 1, 6.0f, 4.0f, 11);
  ok &= test("(b) non-multiple", 50, cubic10, 1, 1, 1, 6.0f, 4.0f, 2);
  ok &= test("(b2) non-multiple", 129, cubic10, 1, 1, 1, 6.0f, 4.0f, 12);
  ok &= test("(c) NL>=128 radial", 200, cubic10, 1, 1, 1, 6.0f, 4.0f, 3);
  ok &= test("(c2) NL>>128 rc=12/6", 300, cubic14, 1, 1, 1, 12.0f, 6.0f, 4);
  ok &= test("(d) N=1", 1, cubic8, 1, 1, 1, 6.0f, 4.0f, 5);
  ok &= test("(d2) N=1 rc=12/6", 1, cubic14, 1, 1, 1, 12.0f, 6.0f, 15);
  ok &= test("(e) triclinic", 120, tri, 1, 1, 1, 6.0f, 4.0f, 6);
  ok &= test("(e2) triclinic rc=9/5", 150, tri2, 1, 1, 1, 9.0f, 5.0f, 16);
  ok &= test("(f) pbc x,y only", 100, cubic10, 1, 1, 0, 6.0f, 4.0f, 7);
  ok &= test("(f2) pbc x only", 100, cubic10, 1, 0, 0, 6.0f, 4.0f, 17);
  ok &= test("(f3) triclinic pbc y,z", 100, tri, 0, 1, 1, 6.0f, 4.0f, 18);
  printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
  return ok ? 0 : 1;
}
