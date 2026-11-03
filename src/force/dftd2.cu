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
C6_ij = sqrt(C6_i C6_j), R0_ij = R0_i + R0_j, d = 20.
The sum runs over all pairs and periodic images with r < rc, with the shifted-force
truncation phi_sf(r) = phi(r) - phi(rc) - phi'(rc) (r - rc).

Pair search: one cell list (counting sort) in fractional coordinates; one thread per atom
loops over the neighboring cells including periodic images (the same kernel handles cells
smaller or larger than the cutoff). Forces and virials are accumulated per atom in registers
(double); there is no atomicAdd on forces.
------------------------------------------------------------------------------*/

#include "dftd2.cuh"
#include "dftd2para.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace
{
constexpr double D_DAMP = 20.0;
// a warning is printed for cutoffs above this value (Angstrom)
constexpr double WARN_CUTOFF = 200.0;
constexpr int BLOCK_SIZE = 128;
constexpr int REDUCE_BLOCK_SIZE = 256;
// target cell width = cutoff / CELL_DIVISOR (finer cells reduce the number of rejected candidates)
constexpr double CELL_DIVISOR = 2.0;

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
  const double4* g_wrapped,
  const int* g_cell,
  const int* g_start,
  const int* g_listed_index,
  double4* g_sorted,
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
  g_sorted[begin + rank] = g_wrapped[i];
  g_sorted_index[begin + rank] = i;
}

// one thread per atom (in cell order); sums over all neighbors, including periodic images
__global__ void dftd2_pair_kernel(
  const int N,
  const int num_types,
  const Grid grid,
  const float rc,
  const float4* g_pair,
  const int* g_cell,
  const int* g_start,
  const double4* g_sorted,
  const int* g_sorted_index,
  double* g_potential,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= N) {
    return;
  }
  const int i = g_sorted_index[slot];
  const double4 pi = g_sorted[slot];
  const int type_i = static_cast<int>(pi.w);
  const int cell_i = g_cell[i];
  const int cx = cell_i % grid.n[0];
  const int cy = (cell_i / grid.n[0]) % grid.n[1];
  const int cz = cell_i / (grid.n[0] * grid.n[1]);
  const float rc2 = rc * rc;

  double e_i = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
  double v_xx = 0.0, v_yy = 0.0, v_zz = 0.0, v_xy = 0.0, v_xz = 0.0, v_yz = 0.0;
  double v_yx = 0.0, v_zx = 0.0, v_zy = 0.0;

  for (int oz = -grid.k[2]; oz <= grid.k[2]; ++oz) {
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
        // translation of the image cell
        const double tx = shx * grid.H[0] + shy * grid.H[1] + shz * grid.H[2];
        const double ty = shx * grid.H[3] + shy * grid.H[4] + shz * grid.H[5];
        const double tz = shx * grid.H[6] + shy * grid.H[7] + shz * grid.H[8];
        const bool home = (shx == 0 && shy == 0 && shz == 0);
        const int j_end = g_start[cell + 1];
        for (int js = g_start[cell]; js < j_end; ++js) {
          if (home && js == slot) {
            continue;
          }
          const double4 pj = g_sorted[js];
          const float dx = static_cast<float>(pj.x + tx - pi.x);
          const float dy = static_cast<float>(pj.y + ty - pi.y);
          const float dz = static_cast<float>(pj.z + tz - pi.z);
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
          e_i += 0.5 * static_cast<double>(phi_sf);
          const float g = dphi_sf * rinv;
          const float fpx = g * dx; // force on i from this neighbor
          const float fpy = g * dy;
          const float fpz = g * dz;
          fx += fpx;
          fy += fpy;
          fz += fpz;
          v_xx -= 0.5 * static_cast<double>(dx * fpx);
          v_yy -= 0.5 * static_cast<double>(dy * fpy);
          v_zz -= 0.5 * static_cast<double>(dz * fpz);
          v_xy -= 0.5 * static_cast<double>(dx * fpy);
          v_xz -= 0.5 * static_cast<double>(dx * fpz);
          v_yz -= 0.5 * static_cast<double>(dy * fpz);
          v_yx -= 0.5 * static_cast<double>(dy * fpx);
          v_zx -= 0.5 * static_cast<double>(dz * fpx);
          v_zy -= 0.5 * static_cast<double>(dz * fpy);
        }
      }
    }
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

template <typename T>
void ensure_size(GPU_Vector<T>& v, const size_t n)
{
  if (v.size() < n) {
    v.resize(n);
  }
}
} // namespace

void DFTD2::parse_tokens(const std::vector<std::string>& tokens, double& s6, double& rc)
{
  if (tokens.size() != 2 && tokens.size() != 3) {
    PRINT_INPUT_ERROR("dftd2 requires 1 or 2 parameters: dftd2 <s6> [<cutoff in Angstrom>].");
  }
  s6 = get_double_from_token(tokens[1], __FILE__, __LINE__);
  if (!(s6 > 0.0) || !std::isfinite(s6)) {
    PRINT_INPUT_ERROR("dftd2: s6 must be a positive number.");
  }
  rc = DEFAULT_CUTOFF;
  if (tokens.size() == 3) {
    rc = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (!(rc > 0.0) || !std::isfinite(rc)) {
      PRINT_INPUT_ERROR("dftd2: cutoff must be a positive number (Angstrom).");
    }
  }
}

bool DFTD2::find_in_run_input(const RunInput& run_input, double& s6, double& rc)
{
  bool found = false;
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty() && line.tokens[0] == "dftd2") {
      if (found) {
        PRINT_INPUT_ERROR("dftd2 can only appear once.");
      }
      parse_tokens(line.tokens, s6, rc);
      found = true;
    }
  }
  if (found && run_input.contains("dftd3")) {
    PRINT_INPUT_ERROR("dftd2 and dftd3 cannot be used together; use only one of them.");
  }
  return found;
}

void DFTD2::initialize(
  const double s6_in, const double rc_in, const std::vector<int>& atomic_numbers)
{
  enabled = true;
  s6 = s6_in;
  // The device code uses single-precision distances, so the cutoff is rounded to float once here;
  // the shift terms phi(rc), phi'(rc), the cutoff test and the cell search all use this same value.
  rc = static_cast<double>(static_cast<float>(rc_in));
  atomic_numbers_ = atomic_numbers;
  num_types_ = static_cast<int>(atomic_numbers.size());

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
        const double s6c6 = s6 * std::sqrt(DFTD2_C6_EV_A6[za] * DFTD2_C6_EV_A6[zb]);
        const double R0 = DFTD2_R0_A[za] + DFTD2_R0_A[zb];
        double phic, dphic;
        phi_dphi(rc, s6c6, R0, phic, dphic);
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

  printf("Use the DFT-D2 correction (s6 = %g, shifted-force cutoff = %g A).\n", s6, rc);
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
    msg += ". The built-in DFT-D2 table covers Z <= 86 (H-Rn) only.";
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
  for (int d = 0; d < 3; ++d) {
    const double width = grid.len[d] * height[d];
    grid.n[d] = std::max(1, static_cast<int>(std::floor(CELL_DIVISOR * width / rc)));
  }
  const long long max_cells = std::max<long long>(1000, 2LL * N);
  while (static_cast<long long>(grid.n[0]) * grid.n[1] * grid.n[2] > max_cells) {
    int dmax = 0;
    for (int d = 1; d < 3; ++d) {
      if (grid.n[d] > grid.n[dmax]) {
        dmax = d;
      }
    }
    grid.n[dmax] = std::max(1, grid.n[dmax] / 2);
  }
  for (int d = 0; d < 3; ++d) {
    const double cell_width = grid.len[d] * height[d] / grid.n[d];
    int k = static_cast<int>(std::ceil(rc / cell_width * (1.0 + 1.0e-12)));
    if (!grid.periodic[d]) {
      k = std::min(k, grid.n[d] - 1);
    }
    grid.k[d] = std::max(k, 0);
  }
  const int num_cells = grid.n[0] * grid.n[1] * grid.n[2];

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
    wrapped_.data(),
    cell_of_atom_.data(),
    cell_start_.data(),
    listed_index_.data(),
    sorted_.data(),
    sorted_index_.data());
  GPU_CHECK_KERNEL
  dftd2_pair_kernel<<<grid_size, BLOCK_SIZE>>>(
    N,
    num_types_,
    grid,
    static_cast<float>(rc),
    pair_param_.data(),
    cell_of_atom_.data(),
    cell_start_.data(),
    sorted_.data(),
    sorted_index_.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}
