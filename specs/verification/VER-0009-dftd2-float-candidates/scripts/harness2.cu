#include "force/dftd2.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
// harness2 INFILE OUTFILE RC NITER
// INFILE: N / 9 H (row-major, cell vectors are columns) / 3 pbc / nz zs... / N lines: type x y z
int main(int argc, char** argv)
{
  FILE* f = fopen(argv[1], "r");
  const double rc = atof(argv[3]);
  const int niter = atoi(argv[4]);
  int N, nz;
  Box box;
  for (int a = 0; a < 18; ++a) box.cpu_h[a] = 0;
  fscanf(f, "%d", &N);
  for (int a = 0; a < 9; ++a) fscanf(f, "%lf", &box.cpu_h[a]);
  fscanf(f, "%d %d %d", &box.pbc_x, &box.pbc_y, &box.pbc_z);
  fscanf(f, "%d", &nz);
  std::vector<int> zs(nz);
  for (int k = 0; k < nz; ++k) fscanf(f, "%d", &zs[k]);
  std::vector<int> type(N);
  std::vector<double> pos(3 * N);
  for (int i = 0; i < N; ++i) {
    double x, y, z;
    fscanf(f, "%d %lf %lf %lf", &type[i], &x, &y, &z);
    pos[i] = x; pos[i + N] = y; pos[i + 2 * N] = z;
  }
  box.get_inverse();
  GPU_Vector<int> d_type(N);
  GPU_Vector<double> d_pos(3 * N), d_pot(N, 0.0), d_f(3 * N, 0.0), d_v(9 * N, 0.0);
  d_type.copy_from_host(type.data());
  d_pos.copy_from_host(pos.data());
  DFTD2 d2;
  d2.initialize(0.75, rc, zs);
  for (int it = 0; it < niter; ++it) {
    d2.compute(box, d_type, d_pos, d_pot, d_f, d_v);
  }
  cudaDeviceSynchronize();
  std::vector<double> pot(N), F(3 * N), V(9 * N);
  d_pot.copy_to_host(pot.data());
  d_f.copy_to_host(F.data());
  d_v.copy_to_host(V.data());
  FILE* o = fopen(argv[2], "wb");
  const double s = 1.0 / niter;
  for (auto& p : pot) p *= s;
  for (auto& p : F) p *= s;
  for (auto& p : V) p *= s;
  fwrite(pot.data(), 8, N, o); fwrite(F.data(), 8, 3 * N, o); fwrite(V.data(), 8, 9 * N, o);
  fclose(o);
  return 0;
}
