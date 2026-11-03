#include "force/dftd2.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv)
{
  // harness FILE RC NITER
  FILE* f = fopen(argv[1], "r");
  const double rc = atof(argv[2]);
  const int niter = atoi(argv[3]);
  int N;
  double L[3];
  fscanf(f, "%d", &N);
  fscanf(f, "%lf %lf %lf", &L[0], &L[1], &L[2]);
  std::vector<int> zs;
  {
    char line[256];
    fgets(line, 256, f);
    fgets(line, 256, f);
    char* p = line;
    int z, n;
    while (sscanf(p, "%d%n", &z, &n) == 1) { zs.push_back(z); p += n; }
  }
  std::vector<int> type(N);
  std::vector<double> pos(3 * N);
  for (int i = 0; i < N; ++i) {
    double x, y, z;
    fscanf(f, "%d %lf %lf %lf", &type[i], &x, &y, &z);
    pos[i] = x; pos[i + N] = y; pos[i + 2 * N] = z;
  }
  Box box;
  for (int a = 0; a < 18; ++a) box.cpu_h[a] = 0;
  box.cpu_h[0] = L[0]; box.cpu_h[4] = L[1]; box.cpu_h[8] = L[2];
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
  std::vector<double> pot(N);
  d_pot.copy_to_host(pot.data());
  double s = 0; for (double p : pot) s += p;
  printf("E_sum_last %.6e\n", s / niter);
  return 0;
}
