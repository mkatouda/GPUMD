/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
DFT-D2:

[1] Stefan Grimme, Semiempirical GGA-type density functional constructed with a
long-range dispersion correction, J. Comput. Chem. 27, 1787 (2006).

E = - s6 sum_{i<j} C6_ij f(r_ij) / r_ij^6,  f = 1 / (1 + exp(-d (r / R0_ij - 1))),
C6_ij = sqrt(C6_i C6_j), R0_ij = R0_i + R0_j, d = 20 (table grimme2006, the default).
The sum runs over all pairs and periodic images with r < rc, with the shifted-force
truncation phi_sf(r) = phi(r) - phi(rc) - phi'(rc) (r - rc) (trunc sf, the default).
Alternatives (ADR-0004): table qe takes C6_ij from the DFT-D3 table (first reference point of both
atoms) and R0_ij = 1.1 r0ab_ij (Quantum ESPRESSO dftd3_version = 2, Z <= 86); trunc plain uses the
hard cutoff phi(r) for r < rc without any shift.

Pair search: one cell list (counting sort) in fractional coordinates; one thread per atom
loops over the neighboring cells including periodic images (the same kernel handles cells
smaller or larger than the cutoff). Energy, forces and virials are accumulated per atom in registers
as compensated float sums (TwoSum, see CompensatedSum; ADR-0005) that are added to double
accumulators every FLUSH_CELLS neighbor cells; the sub-threads of an atom are added in double; there
is no atomicAdd on forces. Candidate positions are stored in float
relative to the center of their cell; the displacement between two cell centers is formed in double
and rounded once to float. The pair vector is then (pj - pi) + offset in float: the offset (up to
about k*width, i.e. about rc) is rounded and the sum is rounded again, so each component has an
absolute error of about 6e-8 * (pair distance + cell width), independent of the box size and of the
distance from the origin. For close pairs this is up to about (1 + width/r) times larger than
rounding the exact double vector once (about 6e-8 * r); e.g. r = 3 A and width = 12 A: about 5 times
(2e-7 -> 1e-6 A).
------------------------------------------------------------------------------*/

#include "dftd2.cuh"
#include "dftd2para.cuh"
#include "dftd2qe_table.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

static_assert(
  DFTD2QE_MAX_Z == DFTD2_MAX_Z && DFTD2QE_NUM_PAIRS == DFTD2QE_MAX_Z * (DFTD2QE_MAX_Z + 1) / 2,
  "the table qe arrays must cover Z <= DFTD2_MAX_Z as a packed lower triangle");

namespace
{
constexpr double D_DAMP = 20.0;
// Constants of the `table qe` pair parameters (Quantum ESPRESSO 7.2, dft-d3 with dftd3_version =
// 2). Modules/constants.f90 (CODATA 2018): BOHR_RADIUS_SI, HARTREE_SI, ELECTRONVOLT_SI; AUTOEV =
// HARTREE_SI / ELECTRONVOLT_SI = 27.211386245988. C6 [eV A^6] = C6 [Eh bohr^6] * AUTOEV * BOHR^6
// with BOHR in Angstrom. R0 = 1.1 * r0ab [A] (QE divides r0ab by autoang and multiplies it back by
// the Bohr radius: the relative difference is 4.4e-9 and is neglected).
constexpr double QE_BOHR_RADIUS_SI = 0.529177210903e-10;
constexpr double QE_HARTREE_SI = 4.3597447222071e-18;
constexpr double QE_ELECTRONVOLT_SI = 1.602176634e-19;
constexpr double QE_BOHR_A = QE_BOHR_RADIUS_SI * 1.0e10;
constexpr double QE_AUTOEV = QE_HARTREE_SI / QE_ELECTRONVOLT_SI;
constexpr double QE_RS6 = 1.1;
// a warning is printed for cutoffs above this value (Angstrom)
constexpr double WARN_CUTOFF = 200.0;
constexpr int BLOCK_SIZE = 128;
constexpr int REDUCE_BLOCK_SIZE = 256;
constexpr int MAX_NUM_SUB = 32;
constexpr double MIN_CANDIDATES_PER_THREAD = 32.0;
constexpr int NUM_REDUCE = 13; // E, F (3), W (9) per atom
// The compensated float sums of a thread are added to double accumulators after this many neighbor
// cells. It bounds the number of terms per float sum (and so the second-order error of the
// compensation: net force / largest force of dense water at rc = 30 A is 8e-14 (4e-12 without the
// flush; the test cap is 1e-12)); the cost is 13 double additions per flush (ADR-0005).
constexpr int FLUSH_CELLS = 4;
// The number of cells per direction is chosen by a cost estimate (see choose_cell_counts):
// candidate counts are floor(kk * width / rc) for kk = 1..MAX_CELL_RANGE (cell width >= rc / kk).
constexpr int MAX_CELL_RANGE = 6;
// cost of visiting one (empty or not) neighbor cell, in units of one candidate pair test
// (fitted to measured pair-kernel times on water and Cu, rc = 12..30 A)
constexpr double CELL_VISIT_COST = 3.0;

const char* const SYMBOL[94] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

// plain (untruncated) pair energy and its radial derivative, in double (host only)
void phi_dphi(const double r, const double s6c6, const double R0, double& phi, double& dphi)
{
  const double e = std::exp(-D_DAMP * (r / R0 - 1.0));
  const double f = 1.0 / (1.0 + e);
  const double dfdr = (D_DAMP / R0) * e * f * f;
  phi = -s6c6 * f / std::pow(r, 6);
  dphi = -s6c6 * (dfdr / std::pow(r, 6) - 6.0 * f / std::pow(r, 7));
}

struct Grid {
  int n[3]; // number of cells
  int k[3]; // neighbor-cell range in each direction
  int periodic[3];
  double s0[3];  // fractional origin of the grid
  double len[3]; // fractional length covered by the grid
  double H[9];   // lattice vectors are the columns (box.cpu_h[0..8])
  double C[9];   // cell step vectors: column d = H[:, d] * len[d] / n[d]
  double O[3];   // cartesian origin of the grid: H * s0
  double Hinv[9];
};

// min / max of the fractional coordinates over all atoms (one block)
__global__ void find_fractional_extent(
  const int N, const Grid grid, const double* x, const double* y, const double* z, double* extent)
{
  __shared__ double s_min[3][REDUCE_BLOCK_SIZE];
  __shared__ double s_max[3][REDUCE_BLOCK_SIZE];
  const int tid = threadIdx.x;
  double lo[3] = {1.0e300, 1.0e300, 1.0e300};
  double hi[3] = {-1.0e300, -1.0e300, -1.0e300};
  for (int i = tid; i < N; i += blockDim.x) {
    for (int d = 0; d < 3; ++d) {
      const double s =
        grid.Hinv[3 * d] * x[i] + grid.Hinv[3 * d + 1] * y[i] + grid.Hinv[3 * d + 2] * z[i];
      lo[d] = min(lo[d], s);
      hi[d] = max(hi[d], s);
    }
  }
  for (int d = 0; d < 3; ++d) {
    s_min[d][tid] = lo[d];
    s_max[d][tid] = hi[d];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int d = 0; d < 3; ++d) {
        s_min[d][tid] = min(s_min[d][tid], s_min[d][tid + stride]);
        s_max[d][tid] = max(s_max[d][tid], s_max[d][tid + stride]);
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int d = 0; d < 3; ++d) {
      extent[2 * d] = s_min[d][0];
      extent[2 * d + 1] = s_max[d][0];
    }
  }
}

// wrap into the cell (periodic directions), assign the cell, count atoms per cell
__global__ void bin_atoms(
  const int N,
  const Grid grid,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  double4* g_wrapped,
  int* g_cell,
  int* g_count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  double x[3] = {g_x[i], g_y[i], g_z[i]};
  double s[3];
  for (int d = 0; d < 3; ++d) {
    s[d] = grid.Hinv[3 * d] * x[0] + grid.Hinv[3 * d + 1] * x[1] + grid.Hinv[3 * d + 2] * x[2];
  }
  int c[3];
  for (int d = 0; d < 3; ++d) {
    if (grid.periodic[d]) {
      const double fl = floor(s[d]);
      s[d] -= fl;
      for (int a = 0; a < 3; ++a) {
        x[a] -= fl * grid.H[3 * a + d];
      }
    }
    c[d] = static_cast<int>((s[d] - grid.s0[d]) / grid.len[d] * grid.n[d]);
    c[d] = min(max(c[d], 0), grid.n[d] - 1);
  }
  const int cell = c[0] + grid.n[0] * (c[1] + grid.n[1] * c[2]);
  g_wrapped[i] = make_double4(x[0], x[1], x[2], static_cast<double>(g_type[i]));
  g_cell[i] = cell;
  atomicAdd(&g_count[cell], 1);
}

// exclusive prefix sum over the cells (one block, sequential chunks; no warp-size assumption)
__global__ void scan_cells(const int num_cells, const int* g_count, int* g_start)
{
  __shared__ int s_total[REDUCE_BLOCK_SIZE];
  const int tid = threadIdx.x;
  const int chunk = (num_cells + blockDim.x - 1) / blockDim.x;
  const int begin = min(tid * chunk, num_cells);
  const int end = min(begin + chunk, num_cells);
  int sum = 0;
  for (int c = begin; c < end; ++c) {
    sum += g_count[c];
  }
  s_total[tid] = sum;
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int t = 0; t < blockDim.x; ++t) {
      const int v = s_total[t];
      s_total[t] = acc;
      acc += v;
    }
    g_start[num_cells] = acc;
  }
  __syncthreads();
  int acc = s_total[tid];
  for (int c = begin; c < end; ++c) {
    g_start[c] = acc;
    acc += g_count[c];
  }
}

// first pass: list the atoms of each cell; the order inside a cell depends on thread scheduling
__global__ void
list_atoms(const int N, const int* g_cell, const int* g_start, int* g_fill, int* g_listed_index)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const int cell = g_cell[i];
  g_listed_index[g_start[cell] + atomicAdd(&g_fill[cell], 1)] = i;
}

// second pass: deterministic order inside a cell (increasing original index): the slot of atom i
// is the start of its cell plus the number of atoms of the same cell with a smaller index
__global__ void place_atoms(
  const int N,
  const Grid grid,
  const double4* g_wrapped,
  const int* g_cell,
  const int* g_start,
  const int* g_listed_index,
  float4* g_sorted,
  int* g_sorted_index)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }
  const int cell = g_cell[i];
  const int begin = g_start[cell];
  const int end = g_start[cell + 1];
  int rank = 0;
  for (int s = begin; s < end; ++s) {
    rank += (g_listed_index[s] < i) ? 1 : 0;
  }
  // position relative to the center of its own cell (small numbers: float keeps ~1e-7 of the cell
  // width as absolute accuracy)
  const double4 w = g_wrapped[i];
  const int ux = cell % grid.n[0];
  const int uy = (cell / grid.n[0]) % grid.n[1];
  const int uz = cell / (grid.n[0] * grid.n[1]);
  const double cu[3] = {ux + 0.5, uy + 0.5, uz + 0.5};
  float local[3];
  for (int a = 0; a < 3; ++a) {
    const double center =
      grid.O[a] + grid.C[3 * a] * cu[0] + grid.C[3 * a + 1] * cu[1] + grid.C[3 * a + 2] * cu[2];
    local[a] = static_cast<float>((a == 0 ? w.x : (a == 1 ? w.y : w.z)) - center);
  }
  g_sorted[begin + rank] = make_float4(local[0], local[1], local[2], static_cast<float>(w.w));
  g_sorted_index[begin + rank] = i;
}

// Float sum with a running compensation term (error-free TwoSum of each addition): value() = sum
// + err is the sum of the terms with an error of about one float rounding (relative to the sum of
// their magnitudes) instead of one per term, and no FP64 operation is needed per pair (FP64 is
// 1/64 of FP32 on GA102). TwoSum must not be reassociated; without fast-math the compiler does not
// do so. The terms must be passed as already rounded floats (no contraction into the sum).
struct CompensatedSum {
  float sum = 0.0f;
  float err = 0.0f;
  __device__ __forceinline__ void add(const float x)
  {
    const float t = sum + x;
    const float bb = t - sum;
    err += (sum - (t - bb)) + (x - bb);
    sum = t;
  }
  __device__ __forceinline__ double value() const
  {
    return static_cast<double>(sum) + static_cast<double>(err);
  }
};

// The 13 per-atom sums of one thread: compensated float partial sums and their double totals.
// Index: 0 = sum of phi_sf, 1..3 = force, 4..12 = sums of dx * fp (xx, yy, zz, xy, xz, yz, yx, zx,
// zy), the order of SubReduce.
struct PairSums {
  CompensatedSum part[NUM_REDUCE];
  double total[NUM_REDUCE];
  __device__ PairSums()
  {
#pragma unroll
    for (int c = 0; c < NUM_REDUCE; ++c) {
      total[c] = 0.0;
    }
  }
  __device__ __forceinline__ void flush()
  {
#pragma unroll
    for (int c = 0; c < NUM_REDUCE; ++c) {
      total[c] += part[c].value();
      part[c] = CompensatedSum();
    }
  }
};

// Sum over the NUM_SUB consecutive threads that share an atom, in a fixed tree order; the result is
// left in the first thread (sub == 0) of the group. The shared array exists only for NUM_SUB > 1.
template <int NUM_SUB>
struct SubReduce {
  static __device__ void run(
    const int tid,
    const int sub,
    double& e_i,
    double& fx,
    double& fy,
    double& fz,
    double& v_xx,
    double& v_yy,
    double& v_zz,
    double& v_xy,
    double& v_xz,
    double& v_yz,
    double& v_yx,
    double& v_zx,
    double& v_zy)
  {
    __shared__ double s_red[NUM_REDUCE][BLOCK_SIZE];
    s_red[0][tid] = e_i;
    s_red[1][tid] = fx;
    s_red[2][tid] = fy;
    s_red[3][tid] = fz;
    s_red[4][tid] = v_xx;
    s_red[5][tid] = v_yy;
    s_red[6][tid] = v_zz;
    s_red[7][tid] = v_xy;
    s_red[8][tid] = v_xz;
    s_red[9][tid] = v_yz;
    s_red[10][tid] = v_yx;
    s_red[11][tid] = v_zx;
    s_red[12][tid] = v_zy;
    for (int offset = NUM_SUB / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (sub < offset) {
        for (int c = 0; c < NUM_REDUCE; ++c) {
          s_red[c][tid] += s_red[c][tid + offset];
        }
      }
    }
    e_i = s_red[0][tid];
    fx = s_red[1][tid];
    fy = s_red[2][tid];
    fz = s_red[3][tid];
    v_xx = s_red[4][tid];
    v_yy = s_red[5][tid];
    v_zz = s_red[6][tid];
    v_xy = s_red[7][tid];
    v_xz = s_red[8][tid];
    v_yz = s_red[9][tid];
    v_yx = s_red[10][tid];
    v_zx = s_red[11][tid];
    v_zy = s_red[12][tid];
  }
};

// one thread per atom: nothing to add
template <>
struct SubReduce<1> {
  static __device__ void run(
    const int,
    const int,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&,
    double&)
  {
  }
};

// NUM_SUB consecutive threads (a power of two) share one atom (in cell order). Each of them takes
// every NUM_SUB-th atom of each neighbor cell (including periodic images); the partial sums are
// added in a fixed tree order in shared memory, so the result does not depend on scheduling.
template <int NUM_SUB>
__global__ void dftd2_pair_kernel(
  const int N,
  const int num_types,
  const Grid grid,
  const float rc,
  const float4* g_pair,
  const int* g_cell,
  const int* g_start,
  const float4* g_sorted,
  const int* g_sorted_index,
  double* g_potential,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  static_assert(
    NUM_SUB >= 1 && (NUM_SUB & (NUM_SUB - 1)) == 0 && BLOCK_SIZE % NUM_SUB == 0,
    "NUM_SUB must be a power of two that divides the block size");
  const int tid = threadIdx.x;
  const int sub = tid % NUM_SUB;
  const int slot = blockIdx.x * (BLOCK_SIZE / NUM_SUB) + tid / NUM_SUB;
  const bool valid = slot < N;
  // (invalid threads stay alive for the reduction; they read atom 0 and skip all loops)
  const int slot_safe = valid ? slot : 0;
  const int i = g_sorted_index[slot_safe];
  const float4 pi = g_sorted[slot_safe];
  const int type_i = static_cast<int>(pi.w);
  const int cell_i = g_cell[i];
  const int cx = cell_i % grid.n[0];
  const int cy = (cell_i / grid.n[0]) % grid.n[1];
  const int cz = cell_i / (grid.n[0] * grid.n[1]);
  const float rc2 = rc * rc;

  PairSums sums;
  int cells_since_flush = 0;

  const int oz_begin = valid ? -grid.k[2] : 1;
  const int oz_end = valid ? grid.k[2] : 0; // empty range for invalid threads
  for (int oz = oz_begin; oz <= oz_end; ++oz) {
    const double bz0 = oz * grid.C[2], bz1 = oz * grid.C[5], bz2 = oz * grid.C[8];
    int uz = cz + oz;
    int shz = 0;
    if (grid.periodic[2]) {
      shz = (uz < 0) ? -((-uz + grid.n[2] - 1) / grid.n[2]) : uz / grid.n[2];
      uz -= shz * grid.n[2];
    } else if (uz < 0 || uz >= grid.n[2]) {
      continue;
    }
    for (int oy = -grid.k[1]; oy <= grid.k[1]; ++oy) {
      int uy = cy + oy;
      int shy = 0;
      if (grid.periodic[1]) {
        shy = (uy < 0) ? -((-uy + grid.n[1] - 1) / grid.n[1]) : uy / grid.n[1];
        uy -= shy * grid.n[1];
      } else if (uy < 0 || uy >= grid.n[1]) {
        continue;
      }
      const double by0 = bz0 + oy * grid.C[1], by1 = bz1 + oy * grid.C[4],
                   by2 = bz2 + oy * grid.C[7];
      for (int ox = -grid.k[0]; ox <= grid.k[0]; ++ox) {
        int ux = cx + ox;
        int shx = 0;
        if (grid.periodic[0]) {
          shx = (ux < 0) ? -((-ux + grid.n[0] - 1) / grid.n[0]) : ux / grid.n[0];
          ux -= shx * grid.n[0];
        } else if (ux < 0 || ux >= grid.n[0]) {
          continue;
        }
        const int cell = ux + grid.n[0] * (uy + grid.n[1] * uz);
        // displacement of the (image) neighbor cell center from the center of this atom's cell, in
        // double, rounded once to float (it is the same for all candidates of the cell)
        const float ofx = static_cast<float>(by0 + ox * grid.C[0]);
        const float ofy = static_cast<float>(by1 + ox * grid.C[3]);
        const float ofz = static_cast<float>(by2 + ox * grid.C[6]);
        const bool home = (shx == 0 && shy == 0 && shz == 0);
        const int j_end = g_start[cell + 1];
        for (int js = g_start[cell] + sub; js < j_end; js += NUM_SUB) {
          if (home && js == slot) {
            continue;
          }
          const float4 pj = g_sorted[js];
          const float dx = (pj.x - pi.x) + ofx;
          const float dy = (pj.y - pi.y) + ofy;
          const float dz = (pj.z - pi.z) + ofz;
          const float r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= rc2) {
            continue;
          }
          const float4 par = g_pair[type_i * num_types + static_cast<int>(pj.w)];
          const float r = sqrtf(r2);
          const float rinv = 1.0f / r;
          const float r2inv = rinv * rinv;
          const float r6inv = r2inv * r2inv * r2inv;
          const float e = expf(-static_cast<float>(D_DAMP) * (r / par.y - 1.0f));
          const float f = 1.0f / (1.0f + e);
          const float dfdr = (static_cast<float>(D_DAMP) / par.y) * e * f * f;
          const float phi = -par.x * f * r6inv;
          const float dphi = -par.x * (dfdr * r6inv - 6.0f * f * r6inv * rinv);
          const float phi_sf = phi - par.z - par.w * (r - rc);
          const float dphi_sf = dphi - par.w;
          sums.part[0].add(phi_sf);
          const float g = dphi_sf * rinv;
          // products are rounded to float before they enter the compensated sums
          const float fpx = __fmul_rn(g, dx); // force on i from this neighbor
          const float fpy = __fmul_rn(g, dy);
          const float fpz = __fmul_rn(g, dz);
          sums.part[1].add(fpx);
          sums.part[2].add(fpy);
          sums.part[3].add(fpz);
          sums.part[4].add(__fmul_rn(dx, fpx));
          sums.part[5].add(__fmul_rn(dy, fpy));
          sums.part[6].add(__fmul_rn(dz, fpz));
          sums.part[7].add(__fmul_rn(dx, fpy));
          sums.part[8].add(__fmul_rn(dx, fpz));
          sums.part[9].add(__fmul_rn(dy, fpz));
          sums.part[10].add(__fmul_rn(dy, fpx));
          sums.part[11].add(__fmul_rn(dz, fpx));
          sums.part[12].add(__fmul_rn(dz, fpy));
        }
        if (++cells_since_flush == FLUSH_CELLS) {
          sums.flush();
          cells_since_flush = 0;
        }
      }
    }
  }

  sums.flush();
  // E = 1/2 sum phi_sf, W = -1/2 sum (dx fp) (the factors are exact)
  double e_i = 0.5 * sums.total[0];
  double fx = sums.total[1], fy = sums.total[2], fz = sums.total[3];
  double v_xx = -0.5 * sums.total[4], v_yy = -0.5 * sums.total[5], v_zz = -0.5 * sums.total[6];
  double v_xy = -0.5 * sums.total[7], v_xz = -0.5 * sums.total[8], v_yz = -0.5 * sums.total[9];
  double v_yx = -0.5 * sums.total[10], v_zx = -0.5 * sums.total[11], v_zy = -0.5 * sums.total[12];

  SubReduce<NUM_SUB>::run(
    tid, sub, e_i, fx, fy, fz, v_xx, v_yy, v_zz, v_xy, v_xz, v_yz, v_yx, v_zx, v_zy);
  if (!valid || sub != 0) {
    return;
  }

  g_potential[i] += e_i;
  g_fx[i] += fx;
  g_fy[i] += fy;
  g_fz[i] += fz;
  g_virial[i + 0 * N] += v_xx;
  g_virial[i + 1 * N] += v_yy;
  g_virial[i + 2 * N] += v_zz;
  g_virial[i + 3 * N] += v_xy;
  g_virial[i + 4 * N] += v_xz;
  g_virial[i + 5 * N] += v_yz;
  g_virial[i + 6 * N] += v_yx;
  g_virial[i + 7 * N] += v_zx;
  g_virial[i + 8 * N] += v_zy;
}

// Choose the number of cells n[d] in each direction by minimizing the estimated work per atom
//   (number of candidate atoms visited) + CELL_VISIT_COST * (number of neighbor cells visited),
// for a uniform density. Candidates per direction: n = 1 and floor(kk * width / rc), kk
// = 1..MAX_CELL_RANGE. Each count is limited to cbrt(max_cells), so the product never exceeds
// max_cells. Returns the estimated number of candidate atoms visited per atom for the chosen
// counts.
double choose_cell_counts(
  const int N,
  const double width[3],
  const int periodic[3],
  const double rc,
  const long long max_cells,
  int n[3])
{
  // per-direction cap so that (cap, cap, cap) always fits into max_cells
  const double cap =
    std::max(1.0, std::floor(std::cbrt(static_cast<double>(max_cells)) * (1.0 + 1.0e-9)));
  int cand[3][MAX_CELL_RANGE + 1];
  int num_cand[3];
  for (int d = 0; d < 3; ++d) {
    num_cand[d] = 0;
    for (int kk = 0; kk <= MAX_CELL_RANGE; ++kk) {
      const double v = (kk == 0) ? 1.0 : std::floor(kk * width[d] / rc);
      const int c = static_cast<int>(std::min(std::max(v, 1.0), cap));
      bool seen = false;
      for (int m = 0; m < num_cand[d]; ++m) {
        seen = seen || (cand[d][m] == c);
      }
      if (!seen) {
        cand[d][num_cand[d]++] = c;
      }
    }
  }
  double visited = 0.0;
  double best = -1.0; // the candidate (1, 1, 1) is always evaluated, so n is always set
  for (int a = 0; a < num_cand[0]; ++a) {
    for (int b = 0; b < num_cand[1]; ++b) {
      for (int c = 0; c < num_cand[2]; ++c) {
        const int m[3] = {cand[0][a], cand[1][b], cand[2][c]};
        double cells_visited = 1.0;
        for (int d = 0; d < 3; ++d) {
          const double k = std::ceil(rc / (width[d] / m[d]) * (1.0 + 1.0e-12));
          double count = 2.0 * k + 1.0;
          if (!periodic[d]) {
            count = std::min(count, static_cast<double>(m[d]));
          }
          cells_visited *= count;
        }
        const double num_cells = static_cast<double>(m[0]) * m[1] * m[2];
        const double cost = N * cells_visited / num_cells + CELL_VISIT_COST * cells_visited;
        if (best < 0.0 || cost < best) {
          best = cost;
          visited = N * cells_visited / num_cells;
          for (int d = 0; d < 3; ++d) {
            n[d] = m[d];
          }
        }
      }
    }
  }
  return visited;
}

// Number of threads that share one atom in the pair kernel (a power of two, at most MAX_NUM_SUB):
// the largest value for which the total thread count N * num_sub stays at or below target_threads
// (number of SMs * BLOCK_SIZE: a small system still fills the GPU, a larger one is not slowed down
// by the reduction) and every thread keeps at least MIN_CANDIDATES_PER_THREAD candidate atoms.
// One thread per atom if N * 2 > target_threads.
int get_num_sub(const int N, const double candidates_per_atom, const int target_threads)
{
  int num_sub = 1;
  while (num_sub < MAX_NUM_SUB && static_cast<long long>(N) * (2 * num_sub) <= target_threads &&
         candidates_per_atom >= MIN_CANDIDATES_PER_THREAD * (2 * num_sub)) {
    num_sub *= 2;
  }
  return num_sub;
}

template <typename T>
void ensure_size(GPU_Vector<T>& v, const size_t n)
{
  if (v.size() < n) {
    v.resize(n);
  }
}
} // namespace

namespace
{
// true if the whole token is a number (the cutoff is recognised only as such)
bool is_number(const std::string& token)
{
  char* end = nullptr;
  std::strtod(token.c_str(), &end);
  return !token.empty() && end != token.c_str() && *end == '\0';
}
} // namespace

void DFTD2::parse_tokens(
  const std::vector<std::string>& tokens, double& s6, double& rc, Options& options)
{
  if (tokens.size() < 2) {
    PRINT_INPUT_ERROR("dftd2 requires at least 1 parameter: dftd2 <s6> [<cutoff in Angstrom>] "
                      "[table grimme2006|qe] [trunc sf|plain].");
  }
  s6 = get_double_from_token(tokens[1], __FILE__, __LINE__);
  if (!(s6 > 0.0) || !std::isfinite(s6)) {
    PRINT_INPUT_ERROR("dftd2: s6 must be a positive number.");
  }
  rc = DEFAULT_CUTOFF;
  options = Options();
  size_t next = 2;
  bool has_cutoff = false;
  if (next < tokens.size() && is_number(tokens[next])) {
    rc = get_double_from_token(tokens[next], __FILE__, __LINE__);
    if (!(rc > 0.0) || !std::isfinite(rc)) {
      PRINT_INPUT_ERROR("dftd2: cutoff must be a positive number (Angstrom).");
    }
    has_cutoff = true;
    ++next;
  }
  bool has_table = false;
  bool has_trunc = false;
  while (next < tokens.size()) {
    const std::string& key = tokens[next];
    if (key == "table" || key == "trunc") {
      const bool is_table = (key == "table");
      const char* const valid = is_table ? "grimme2006 or qe" : "sf or plain";
      if (is_table ? has_table : has_trunc) {
        PRINT_INPUT_ERROR(("dftd2: the option '" + key + "' can only be given once.").c_str());
      }
      if (next + 1 >= tokens.size()) {
        PRINT_INPUT_ERROR(
          ("dftd2: the option '" + key + "' needs a value (" + valid + ").").c_str());
      }
      const std::string& value = tokens[next + 1];
      if (is_table) {
        if (value != "grimme2006" && value != "qe") {
          PRINT_INPUT_ERROR(("dftd2: unknown table '" + value +
                             "'; the valid values of 'table' are " + valid + " (lower case).")
                              .c_str());
        }
        options.table_qe = (value == "qe");
        has_table = true;
      } else {
        if (value != "sf" && value != "plain") {
          PRINT_INPUT_ERROR(("dftd2: unknown truncation '" + value +
                             "'; the valid values of 'trunc' are " + valid + " (lower case).")
                              .c_str());
        }
        options.trunc_plain = (value == "plain");
        has_trunc = true;
      }
      next += 2;
    } else if (is_number(key)) {
      PRINT_INPUT_ERROR(
        ("dftd2: unexpected number '" + key +
         "'; the cutoff, if given, must come directly after s6 and before the options "
         "'table' and 'trunc'.")
          .c_str());
    } else {
      PRINT_INPUT_ERROR(("dftd2: unknown token '" + key +
                         "'; the valid options are 'table grimme2006|qe' and 'trunc sf|plain'.")
                          .c_str());
    }
  }
  if ((options.table_qe || options.trunc_plain) && !has_cutoff) {
    PRINT_INPUT_ERROR(
      "dftd2: the cutoff must be given explicitly with 'table qe' or 'trunc plain' "
      "(dftd2 <s6> <cutoff> ...), so that it matches the cutoff used to generate the reference "
      "data.");
  }
}

bool DFTD2::find_in_run_input(const RunInput& run_input, double& s6, double& rc, Options& options)
{
  bool found = false;
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty() && line.tokens[0] == "dftd2") {
      if (found) {
        PRINT_INPUT_ERROR("dftd2 can only appear once.");
      }
      parse_tokens(line.tokens, s6, rc, options);
      found = true;
    }
  }
  if (found && run_input.contains("dftd3")) {
    PRINT_INPUT_ERROR("dftd2 and dftd3 cannot be used together; use only one of them.");
  }
  return found;
}

void DFTD2::initialize(
  const double s6_in,
  const double rc_in,
  const Options& options_in,
  const std::vector<int>& atomic_numbers)
{
  enabled = true;
  s6 = s6_in;
  options = options_in;
  // The device code uses single-precision distances, so the cutoff is rounded to float once here;
  // the shift terms phi(rc), phi'(rc), the cutoff test and the cell search all use this same value.
  rc = static_cast<double>(static_cast<float>(rc_in));
  atomic_numbers_ = atomic_numbers;
  num_types_ = static_cast<int>(atomic_numbers.size());

  // one block of BLOCK_SIZE threads per SM is enough to fill the device for this kernel
  int device_id = 0;
  CHECK(gpuGetDevice(&device_id));
  gpuDeviceProp device_prop;
  CHECK(gpuGetDeviceProperties(&device_prop, device_id));
  target_threads_ = device_prop.multiProcessorCount * BLOCK_SIZE;
  // Debugging/testing aid, intentionally undocumented for users: the environment variable
  // GPUMD_DFTD2_TARGET_THREADS (a positive integer) replaces the target thread count, which fixes
  // the number of threads per atom (see get_num_sub; target = N * s gives s threads per atom if the
  // candidate count allows it), so that every instantiation of the pair kernel can be
  // tested on the same system. The chosen value is then printed once.
  const char* target_env = std::getenv("GPUMD_DFTD2_TARGET_THREADS");
  if (target_env != nullptr) {
    const long long value = std::atoll(target_env);
    if (value > 0 && value <= 1000000000LL) {
      target_threads_ = static_cast<int>(value);
      debug_print_num_sub_ = true;
    }
  }

  std::vector<float4> table(static_cast<size_t>(num_types_) * num_types_);
  for (int a = 0; a < num_types_; ++a) {
    for (int b = 0; b < num_types_; ++b) {
      const int za = atomic_numbers_[a];
      const int zb = atomic_numbers_[b];
      float4 p;
      if (za < 1 || za > DFTD2_MAX_Z || zb < 1 || zb > DFTD2_MAX_Z) {
        // never used: elements without parameters are rejected in check_elements();
        // NaN makes any accidental use visible
        p.x = p.y = p.z = p.w = std::nanf("");
      } else {
        double s6c6, R0;
        if (options.table_qe) {
          // C6 of the first reference point of both atoms, R0 = rs6 * r0ab (0-based Z in the table)
          const int zs = std::min(za, zb) - 1;
          const int zl = std::max(za, zb) - 1;
          const int idx = zl * (zl + 1) / 2 + zs;
          s6c6 = s6 * DFTD2QE_C6_EH_BOHR6[idx] * QE_AUTOEV * std::pow(QE_BOHR_A, 6);
          R0 = QE_RS6 * DFTD2QE_R0AB_A[idx];
        } else {
          s6c6 = s6 * std::sqrt(DFTD2_C6_EV_A6[za] * DFTD2_C6_EV_A6[zb]);
          R0 = DFTD2_R0_A[za] + DFTD2_R0_A[zb];
        }
        double phic = 0.0, dphic = 0.0;
        if (!options.trunc_plain) {
          phi_dphi(rc, s6c6, R0, phic, dphic);
        }
        p.x = static_cast<float>(s6c6);
        p.y = static_cast<float>(R0);
        p.z = static_cast<float>(phic);
        p.w = static_cast<float>(dphic);
      }
      table[static_cast<size_t>(a) * num_types_ + b] = p;
    }
  }
  pair_param_.resize(table.size());
  pair_param_.copy_from_host(table.data());

  if (options.trunc_plain) {
    printf("Use the DFT-D2 correction (s6 = %g, cutoff = %g A).\n", s6, rc);
  } else {
    printf("Use the DFT-D2 correction (s6 = %g, shifted-force cutoff = %g A).\n", s6, rc);
  }
  if (options.table_qe) {
    printf("    table = qe (C6 of the first DFT-D3 reference point, R0 = 1.1 r0ab).\n");
  }
  if (options.trunc_plain) {
    printf("    trunc = plain (hard cutoff, no shifted-force terms).\n");
  }
  if (rc > WARN_CUTOFF) {
    printf(
      "Warning: the dftd2 cutoff (%g A) is larger than %g A; the cost grows as the cube of the "
      "cutoff and the run may be very slow or exhaust memory.\n",
      rc,
      WARN_CUTOFF);
  }
}

void DFTD2::check_elements(const GPU_Vector<int>& type)
{
  const int N = type.size();
  std::vector<int> h_type(N);
  CHECK(gpuMemcpy(h_type.data(), type.data(), sizeof(int) * N, gpuMemcpyDeviceToHost));
  std::set<int> bad;
  for (int i = 0; i < N; ++i) {
    const int t = h_type[i];
    if (t < 0 || t >= num_types_) {
      PRINT_INPUT_ERROR("dftd2: atom type out of range.");
    }
    const int z = atomic_numbers_[t];
    if (z < 1 || z > DFTD2_MAX_Z) {
      bad.insert(z);
    }
  }
  if (!bad.empty()) {
    std::string msg = "dftd2 has no parameters for element(s):";
    for (const int z : bad) {
      msg += " ";
      msg += (z >= 1 && z <= 94) ? SYMBOL[z - 1] : "(unrecognized element)";
      if (z >= 1 && z <= 94) {
        msg += " (Z=" + std::to_string(z) + ")";
      }
    }
    msg += options.table_qe ? ". The 'table qe' parameters cover Z <= 86 (H-Rn) only."
                            : ". The built-in DFT-D2 table covers Z <= 86 (H-Rn) only.";
    PRINT_INPUT_ERROR(msg.c_str());
  }
  elements_checked_ = true;
}

void DFTD2::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();
  if (N == 0) {
    return;
  }
  if (!elements_checked_) {
    check_elements(type);
  }

  Grid grid;
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  for (int a = 0; a < 9; ++a) {
    grid.H[a] = box.cpu_h[a];
    grid.Hinv[a] = box.cpu_h[9 + a];
  }
  // perpendicular heights of the cell: h_d = 1 / |row d of H^-1|
  double height[3];
  for (int d = 0; d < 3; ++d) {
    const double* r = grid.Hinv + 3 * d;
    height[d] = 1.0 / std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    grid.periodic[d] = pbc[d] ? 1 : 0;
    grid.s0[d] = 0.0;
    grid.len[d] = 1.0;
  }

  const double* x = position_per_atom.data();
  const double* y = x + N;
  const double* z = x + 2 * N;

  if (!(grid.periodic[0] && grid.periodic[1] && grid.periodic[2])) {
    ensure_size(extent_, 6);
    find_fractional_extent<<<1, REDUCE_BLOCK_SIZE>>>(N, grid, x, y, z, extent_.data());
    GPU_CHECK_KERNEL
    double ext[6];
    extent_.copy_to_host(ext, 6);
    for (int d = 0; d < 3; ++d) {
      if (!grid.periodic[d]) {
        double len = ext[2 * d + 1] - ext[2 * d];
        const double min_len = 1.0e-6 / height[d]; // at least 1e-6 A wide
        len = std::max(len, min_len) * (1.0 + 1.0e-9);
        grid.s0[d] = ext[2 * d];
        grid.len[d] = len;
      }
    }
  }

  // number of cells and neighbor range
  double width[3];
  for (int d = 0; d < 3; ++d) {
    width[d] = grid.len[d] * height[d];
  }
  const long long max_cells = std::max<long long>(1000, 2LL * N);
  const double candidates_per_atom =
    choose_cell_counts(N, width, grid.periodic, rc, max_cells, grid.n);
  for (int d = 0; d < 3; ++d) {
    const double cell_width = grid.len[d] * height[d] / grid.n[d];
    int k = static_cast<int>(std::ceil(rc / cell_width * (1.0 + 1.0e-12)));
    if (!grid.periodic[d]) {
      k = std::min(k, grid.n[d] - 1);
    }
    grid.k[d] = std::max(k, 0);
  }
  const int num_cells = grid.n[0] * grid.n[1] * grid.n[2];
  for (int d = 0; d < 3; ++d) {
    for (int a = 0; a < 3; ++a) {
      grid.C[3 * a + d] = grid.H[3 * a + d] * grid.len[d] / grid.n[d];
    }
  }
  for (int a = 0; a < 3; ++a) {
    grid.O[a] =
      grid.H[3 * a] * grid.s0[0] + grid.H[3 * a + 1] * grid.s0[1] + grid.H[3 * a + 2] * grid.s0[2];
  }

  ensure_size(wrapped_, N);
  ensure_size(sorted_, N);
  ensure_size(sorted_index_, N);
  ensure_size(listed_index_, N);
  ensure_size(cell_of_atom_, N);
  // num_cells <= max_cells always holds, so the cell arrays are allocated once at the upper bound
  // (no reallocation when the box changes, e.g. in NPT)
  ensure_size(cell_count_, max_cells);
  ensure_size(cell_start_, max_cells + 1);
  ensure_size(cell_fill_, max_cells);
  CHECK(gpuMemset(cell_count_.data(), 0, sizeof(int) * num_cells));
  CHECK(gpuMemset(cell_fill_.data(), 0, sizeof(int) * num_cells));

  const int grid_size = (N - 1) / BLOCK_SIZE + 1;
  bin_atoms<<<grid_size, BLOCK_SIZE>>>(
    N, grid, type.data(), x, y, z, wrapped_.data(), cell_of_atom_.data(), cell_count_.data());
  GPU_CHECK_KERNEL
  scan_cells<<<1, REDUCE_BLOCK_SIZE>>>(num_cells, cell_count_.data(), cell_start_.data());
  GPU_CHECK_KERNEL
  list_atoms<<<grid_size, BLOCK_SIZE>>>(
    N, cell_of_atom_.data(), cell_start_.data(), cell_fill_.data(), listed_index_.data());
  GPU_CHECK_KERNEL
  place_atoms<<<grid_size, BLOCK_SIZE>>>(
    N,
    grid,
    wrapped_.data(),
    cell_of_atom_.data(),
    cell_start_.data(),
    listed_index_.data(),
    sorted_.data(),
    sorted_index_.data());
  GPU_CHECK_KERNEL
  const int num_sub = get_num_sub(N, candidates_per_atom, target_threads_);
  if (debug_print_num_sub_) {
    printf(
      "DFTD2 debug: threads per atom = %d, cells = %d %d %d, range = %d %d %d\n",
      num_sub,
      grid.n[0],
      grid.n[1],
      grid.n[2],
      grid.k[0],
      grid.k[1],
      grid.k[2]);
    fflush(stdout);
  }
  const int pair_blocks = (N * num_sub - 1) / BLOCK_SIZE + 1;
#define DFTD2_LAUNCH_PAIR(NS)                                                                      \
  dftd2_pair_kernel<NS><<<pair_blocks, BLOCK_SIZE>>>(                                              \
    N,                                                                                             \
    num_types_,                                                                                    \
    grid,                                                                                          \
    static_cast<float>(rc),                                                                        \
    pair_param_.data(),                                                                            \
    cell_of_atom_.data(),                                                                          \
    cell_start_.data(),                                                                            \
    sorted_.data(),                                                                                \
    sorted_index_.data(),                                                                          \
    potential_per_atom.data(),                                                                     \
    force_per_atom.data(),                                                                         \
    force_per_atom.data() + N,                                                                     \
    force_per_atom.data() + 2 * N,                                                                 \
    virial_per_atom.data());
  switch (num_sub) {
    case 1:
      DFTD2_LAUNCH_PAIR(1)
      break;
    case 2:
      DFTD2_LAUNCH_PAIR(2)
      break;
    case 4:
      DFTD2_LAUNCH_PAIR(4)
      break;
    case 8:
      DFTD2_LAUNCH_PAIR(8)
      break;
    case 16:
      DFTD2_LAUNCH_PAIR(16)
      break;
    default:
      DFTD2_LAUNCH_PAIR(32)
      break;
  }
#undef DFTD2_LAUNCH_PAIR
  GPU_CHECK_KERNEL
}
