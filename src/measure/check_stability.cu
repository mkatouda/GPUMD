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
Run-time stability guard (DR-008): every <interval> steps evaluate the enabled
criteria (temperature, potential energy change, shortest distance, largest force
component, largest charge) and stop with a diagnostic on the first violation.
The guard only reads the state of the simulation. Implemented for the DR-008
requirement of the dielectric-response development of this fork.
------------------------------------------------------------------------------*/

#include "check_stability.cuh"
#include "force/force.cuh"
#include "force/nep/nep_multigpu.cuh"
#include "force/potential.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{
const int BLOCK_SIZE = 256;
const int MAX_BLOCKS = 128;
const int NUM_PARTIAL = 6; // kinetic energy, potential energy, max |f| and code, max |q| and index

struct Box_Data {
  double h[9];
  double inv[9];
  int pbc[3];
  int n[3]; // cells per direction
  int r[3]; // cell-offset range per direction (image range in a thin periodic cell)
};

// NaN is mapped to infinity so that it is always reported as a violation.
__device__ inline double nan_to_inf(const double a) { return isnan(a) ? INFINITY : a; }

// Per-block partial sums and maxima. Fixed grid and fixed reduction order: deterministic.
void __global__ gpu_reduce_state(
  const int N,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_u,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz,
  const float* g_q,
  double* g_partial)
{
  __shared__ double s_ke[BLOCK_SIZE];
  __shared__ double s_u[BLOCK_SIZE];
  __shared__ double s_f[BLOCK_SIZE];
  __shared__ double s_fc[BLOCK_SIZE];
  __shared__ double s_q[BLOCK_SIZE];
  __shared__ double s_qi[BLOCK_SIZE];
  const int tid = threadIdx.x;
  double ke = 0.0, u = 0.0, fmax = -1.0, fcode = -1.0, qmax = -1.0, qidx = -1.0;
  for (int n = blockIdx.x * blockDim.x + tid; n < N; n += gridDim.x * blockDim.x) {
    const double vx = g_vx[n], vy = g_vy[n], vz = g_vz[n];
    ke += 0.5 * g_mass[n] * (vx * vx + vy * vy + vz * vz);
    u += g_u[n];
    const double fa[3] = {g_fx[n], g_fy[n], g_fz[n]};
    for (int c = 0; c < 3; ++c) {
      const double a = nan_to_inf(fabs(fa[c]));
      if (a > fmax) {
        fmax = a;
        fcode = n * 3 + c;
      }
    }
    if (g_q) {
      const double a = nan_to_inf(fabs(static_cast<double>(g_q[n])));
      if (a > qmax) {
        qmax = a;
        qidx = n;
      }
    }
  }
  s_ke[tid] = ke;
  s_u[tid] = u;
  s_f[tid] = fmax;
  s_fc[tid] = fcode;
  s_q[tid] = qmax;
  s_qi[tid] = qidx;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_ke[tid] += s_ke[tid + offset];
      s_u[tid] += s_u[tid + offset];
      if (
        s_f[tid + offset] > s_f[tid] ||
        (s_f[tid + offset] == s_f[tid] && s_fc[tid + offset] >= 0.0 &&
         s_fc[tid + offset] < s_fc[tid])) {
        s_f[tid] = s_f[tid + offset];
        s_fc[tid] = s_fc[tid + offset];
      }
      if (
        s_q[tid + offset] > s_q[tid] ||
        (s_q[tid + offset] == s_q[tid] && s_qi[tid + offset] >= 0.0 &&
         s_qi[tid + offset] < s_qi[tid])) {
        s_q[tid] = s_q[tid + offset];
        s_qi[tid] = s_qi[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    double* p = g_partial + blockIdx.x * NUM_PARTIAL;
    p[0] = s_ke[0];
    p[1] = s_u[0];
    p[2] = s_f[0];
    p[3] = s_fc[0];
    p[4] = s_q[0];
    p[5] = s_qi[0];
  }
}

// Fractional coordinates; periodic directions are wrapped into [0, 1).
__device__ inline bool
get_fractional(const Box_Data& b, const double x, const double y, const double z, double s[3])
{
  for (int a = 0; a < 3; ++a) {
    s[a] = b.inv[a * 3 + 0] * x + b.inv[a * 3 + 1] * y + b.inv[a * 3 + 2] * z;
    if (!isfinite(s[a]))
      return false;
    if (b.pbc[a])
      s[a] -= floor(s[a]);
  }
  return true;
}

__device__ inline int get_cell_coordinate(const Box_Data& b, const int a, const double s)
{
  int c = static_cast<int>(floor(s * b.n[a]));
  c = c < 0 ? 0 : c;
  c = c > b.n[a] - 1 ? b.n[a] - 1 : c;
  return c;
}

void __global__ gpu_build_cells(
  const int N,
  const Box_Data box,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  int* g_head,
  int* g_next)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double s[3];
    int c[3] = {0, 0, 0};
    if (get_fractional(box, g_x[i], g_y[i], g_z[i], s)) {
      for (int a = 0; a < 3; ++a)
        c[a] = get_cell_coordinate(box, a, s[a]);
    }
    const int cell = c[0] + box.n[0] * (c[1] + box.n[1] * c[2]);
    g_next[i] = atomicExch(&g_head[cell], i);
  }
}

// Shortest distance of each atom to any atom (or periodic image) closer than rmin.
// Cell offsets -r..r are scanned per direction and wrapped with explicit image shifts, where
// r = ceil(rmin * n / width) (width: perpendicular thickness), so every image closer than rmin
// is visited also in a periodic box thinner than rmin or 2 * rmin.
void __global__ gpu_find_pairs(
  const int N,
  const Box_Data box,
  const double rmin,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const int* g_head,
  const int* g_next,
  double* g_d2,
  int* g_partner)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  double best = INFINITY;
  int partner = -1;
  double si[3];
  if (!get_fractional(box, g_x[i], g_y[i], g_z[i], si)) {
    g_d2[i] = NAN;
    g_partner[i] = -1;
    return;
  }
  int ci[3];
  for (int a = 0; a < 3; ++a)
    ci[a] = get_cell_coordinate(box, a, si[a]);
  const double rmin2 = rmin * rmin;
  for (int oz = -box.r[2]; oz <= box.r[2]; ++oz) {
    for (int oy = -box.r[1]; oy <= box.r[1]; ++oy) {
      for (int ox = -box.r[0]; ox <= box.r[0]; ++ox) {
        const int off[3] = {ox, oy, oz};
        int cj[3];
        int shift[3];
        bool valid = true;
        for (int a = 0; a < 3; ++a) {
          int c = ci[a] + off[a];
          shift[a] = 0;
          if (c < 0 || c >= box.n[a]) {
            if (box.pbc[a]) {
              shift[a] = c < 0 ? -((-c + box.n[a] - 1) / box.n[a]) : c / box.n[a];
              c -= shift[a] * box.n[a];
            } else {
              valid = false;
            }
          }
          cj[a] = c;
        }
        if (!valid)
          continue;
        const int cell = cj[0] + box.n[0] * (cj[1] + box.n[1] * cj[2]);
        for (int j = g_head[cell]; j >= 0; j = g_next[j]) {
          double sj[3];
          if (!get_fractional(box, g_x[j], g_y[j], g_z[j], sj))
            continue;
          if (j == i && shift[0] == 0 && shift[1] == 0 && shift[2] == 0)
            continue;
          double ds[3];
          for (int a = 0; a < 3; ++a) {
            ds[a] = box.pbc[a] ? sj[a] + shift[a] - si[a] : sj[a] - si[a];
          }
          const double dx = box.h[0] * ds[0] + box.h[1] * ds[1] + box.h[2] * ds[2];
          const double dy = box.h[3] * ds[0] + box.h[4] * ds[1] + box.h[5] * ds[2];
          const double dz = box.h[6] * ds[0] + box.h[7] * ds[1] + box.h[8] * ds[2];
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < rmin2 && (d2 < best || (d2 == best && j < partner))) {
            best = d2;
            partner = j;
          }
        }
      }
    }
  }
  g_d2[i] = best;
  g_partner[i] = partner;
}

} // namespace

Check_Stability::Check_Stability(
  const std::vector<std::string>& tokens, const std::string& potential_filename)
{
  action_name = "check_stability";
  const int num_tokens = static_cast<int>(tokens.size());
  if (num_tokens < 3) {
    PRINT_INPUT_ERROR(
      "check_stability needs an interval and at least one criterion "
      "(max_temperature, max_potential_change, min_distance, max_force, max_charge).");
  }
  if (!is_valid_int(tokens[1], &interval_) || interval_ <= 0) {
    PRINT_INPUT_ERROR("check_stability: interval should be a positive integer.");
  }
  bool has_keep_frames = false;
  int n = 2;
  while (n < num_tokens) {
    const std::string& key = tokens[n];
    if (n + 1 >= num_tokens) {
      std::string message = "check_stability: missing value after " + key + ".";
      PRINT_INPUT_ERROR(message.c_str());
    }
    const std::string& value = tokens[n + 1];
    double* target = nullptr;
    bool* flag = nullptr;
    if (key == "max_temperature") {
      target = &max_temperature_;
      flag = &use_temperature_;
    } else if (key == "max_potential_change") {
      target = &max_potential_change_;
      flag = &use_potential_change_;
    } else if (key == "min_distance") {
      target = &min_distance_;
      flag = &use_distance_;
    } else if (key == "max_force") {
      target = &max_force_;
      flag = &use_force_;
    } else if (key == "max_charge") {
      target = &max_charge_;
      flag = &use_charge_;
    } else if (key == "keep_frames") {
      if (has_keep_frames) {
        PRINT_INPUT_ERROR("check_stability: keep_frames given twice.");
      }
      if (!is_valid_int(value, &keep_frames_) || keep_frames_ < 0) {
        PRINT_INPUT_ERROR("check_stability: keep_frames should be a non-negative integer.");
      }
      has_keep_frames = true;
    } else {
      std::string message = "check_stability: unknown token '" + key + "'.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    if (target) {
      if (*flag) {
        std::string message = "check_stability: " + key + " given twice.";
        PRINT_INPUT_ERROR(message.c_str());
      }
      if (!is_valid_real(value, target) || !std::isfinite(*target) || *target <= 0.0) {
        std::string message = "check_stability: " + key + " should be a positive real number.";
        PRINT_INPUT_ERROR(message.c_str());
      }
      *flag = true;
    }
    n += 2;
  }
  if (!(use_temperature_ || use_potential_change_ || use_distance_ || use_force_ || use_charge_)) {
    PRINT_INPUT_ERROR("check_stability: at least one criterion is required.");
  }
  if (use_charge_ && !is_nep_charge_potential(potential_filename)) {
    PRINT_INPUT_ERROR("check_stability: max_charge requires a charge model (qnep1, qnep2, qnep3); "
                      "the potential has no charges.");
  }
  printf("Check stability every %d steps (keep %d frames):", interval_, keep_frames_);
  if (use_temperature_)
    printf(" max_temperature=%g K", max_temperature_);
  if (use_potential_change_)
    printf(" max_potential_change=%g eV/atom", max_potential_change_);
  if (use_distance_)
    printf(" min_distance=%g A", min_distance_);
  if (use_force_)
    printf(" max_force=%g eV/A", max_force_);
  if (use_charge_)
    printf(" max_charge=%g e", max_charge_);
  printf("\n");
}

void Check_Stability::pre_run(
  const int,
  const double,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box&,
  Force& force)
{
  for (int i = 0; i < force.get_number_of_potentials(); ++i) {
    if (dynamic_cast<NEP_MULTIGPU*>(&force.get_potential(i)) != nullptr) {
      PRINT_INPUT_ERROR("check_stability does not support multi-GPU NEP.");
    }
  }
  if (is_pimd(integrate.get_type())) {
    // positions, velocities, forces and energies of the beads are not in the arrays read here
    PRINT_INPUT_ERROR("check_stability does not support path-integral (PIMD) ensembles.");
  }
  num_atoms_ = atom.number_of_atoms;
  if (num_atoms_ <= 0) {
    PRINT_INPUT_ERROR("check_stability: the system has no atoms.");
  }
  if (use_charge_) {
    if (
      force.get_number_of_potentials() < 1 ||
      static_cast<int>(force.get_potential(0).get_charge_reference().size()) != num_atoms_) {
      PRINT_INPUT_ERROR(
        "check_stability: max_charge requires the first potential to be a charge model.");
    }
  }
  has_reference_ = false;
  potential_reference_ = 0.0;
  num_stored_ = 0;
  next_slot_ = 0;
  const int capacity = keep_frames_ > 0 ? keep_frames_ : 1;
  frame_info_.assign(capacity, Frame_Info());
  frame_state_.resize(static_cast<size_t>(capacity) * num_atoms_ * 9);
  if (use_charge_)
    frame_charge_.resize(static_cast<size_t>(capacity) * num_atoms_);
  num_blocks_ = (num_atoms_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
  num_blocks_ = num_blocks_ > MAX_BLOCKS ? MAX_BLOCKS : num_blocks_;
  partial_.resize(static_cast<size_t>(num_blocks_) * NUM_PARTIAL);
  cpu_partial_.resize(static_cast<size_t>(num_blocks_) * NUM_PARTIAL);
  if (use_distance_) {
    cell_next_.resize(num_atoms_);
    pair_d2_.resize(num_atoms_);
    pair_partner_.resize(num_atoms_);
    cpu_pair_d2_.resize(num_atoms_);
    cpu_pair_partner_.resize(num_atoms_);
  }
}

void Check_Stability::store_frame(const int step, Atom& atom, const Box& box, const float* charge)
{
  const int capacity = static_cast<int>(frame_info_.size());
  const int slot = next_slot_;
  const size_t N = num_atoms_;
  double* dst = frame_state_.data() + static_cast<size_t>(slot) * N * 9;
  CHECK(
    gpuMemcpy(dst, atom.position_per_atom.data(), sizeof(double) * N * 3, gpuMemcpyDeviceToDevice));
  CHECK(gpuMemcpy(
    dst + 3 * N, atom.velocity_per_atom.data(), sizeof(double) * N * 3, gpuMemcpyDeviceToDevice));
  CHECK(gpuMemcpy(
    dst + 6 * N, atom.force_per_atom.data(), sizeof(double) * N * 3, gpuMemcpyDeviceToDevice));
  if (charge) {
    CHECK(gpuMemcpy(
      frame_charge_.data() + static_cast<size_t>(slot) * N,
      charge,
      sizeof(float) * N,
      gpuMemcpyDeviceToDevice));
  }
  Frame_Info& info = frame_info_[slot];
  info.step = step;
  info.pbc[0] = box.pbc_x;
  info.pbc[1] = box.pbc_y;
  info.pbc[2] = box.pbc_z;
  for (int k = 0; k < 9; ++k)
    info.h[k] = box.cpu_h[k];
  next_slot_ = (slot + 1) % capacity;
  if (num_stored_ < capacity)
    ++num_stored_;
}

void Check_Stability::write_frames(const Atom& atom)
{
  const size_t N = num_atoms_;
  const int capacity = static_cast<int>(frame_info_.size());
  std::vector<double> state(static_cast<size_t>(capacity) * N * 9);
  frame_state_.copy_to_host(state.data());
  std::vector<float> charge;
  if (use_charge_) {
    charge.resize(static_cast<size_t>(capacity) * N);
    frame_charge_.copy_to_host(charge.data());
  }
  FILE* fid = fopen("stability_frames.xyz", "w");
  if (fid == NULL) {
    printf("check_stability: cannot open stability_frames.xyz for writing.\n");
    return;
  }
  const bool has_symbol = atom.cpu_atom_symbol.size() == N;
  for (int k = 0; k < num_stored_; ++k) {
    // oldest first
    const int slot = (next_slot_ - num_stored_ + k + 2 * capacity) % capacity;
    const Frame_Info& info = frame_info_[slot];
    const double* s = state.data() + static_cast<size_t>(slot) * N * 9;
    fprintf(fid, "%d\n", num_atoms_);
    fprintf(
      fid,
      "Lattice=\"%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\" "
      "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3%s pbc=\"%c %c %c\" step=%d\n",
      info.h[0],
      info.h[3],
      info.h[6],
      info.h[1],
      info.h[4],
      info.h[7],
      info.h[2],
      info.h[5],
      info.h[8],
      use_charge_ ? ":charge:R:1" : "",
      info.pbc[0] ? 'T' : 'F',
      info.pbc[1] ? 'T' : 'F',
      info.pbc[2] ? 'T' : 'F',
      info.step);
    for (size_t n = 0; n < N; ++n) {
      fprintf(fid, "%s", has_symbol ? atom.cpu_atom_symbol[n].c_str() : "X");
      for (int c = 0; c < 9; ++c) {
        // velocities in A/fs, as in dump_xyz
        const double scale = (c >= 3 && c < 6) ? 1.0 / TIME_UNIT_CONVERSION : 1.0;
        fprintf(fid, " %.16g", s[c * N + n] * scale);
      }
      if (use_charge_)
        fprintf(fid, " %.9g", static_cast<double>(charge[static_cast<size_t>(slot) * N + n]));
      fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

double
Check_Stability::find_shortest_distance(Atom& atom, const Box& box, int& index_1, int& index_2)
{
  const int N = num_atoms_;
  Box_Data bd;
  for (int k = 0; k < 9; ++k) {
    bd.h[k] = box.cpu_h[k];
    bd.inv[k] = box.cpu_h[9 + k];
  }
  bd.pbc[0] = box.pbc_x;
  bd.pbc[1] = box.pbc_y;
  bd.pbc[2] = box.pbc_z;
  double widths[3];
  for (int a = 0; a < 3; ++a) {
    const double norm = sqrt(
      bd.inv[a * 3] * bd.inv[a * 3] + bd.inv[a * 3 + 1] * bd.inv[a * 3 + 1] +
      bd.inv[a * 3 + 2] * bd.inv[a * 3 + 2]);
    const double width = norm > 0.0 ? 1.0 / norm : 0.0;
    widths[a] = width;
    // clamp the ratio before the int conversion (a huge ratio would overflow)
    const double ratio = std::isfinite(width) ? width / min_distance_ : 0.0;
    const int n = ratio > 1.0e6 ? 1000000 : static_cast<int>(ratio);
    bd.n[a] = n > 1 ? n : 1;
  }
  // limit the number of cells to O(N); larger cells stay correct
  const long long max_cells = 2LL * N + 27;
  while (static_cast<long long>(bd.n[0]) * bd.n[1] * bd.n[2] > max_cells) {
    int a = 0;
    for (int b = 1; b < 3; ++b)
      if (bd.n[b] > bd.n[a])
        a = b;
    bd.n[a] = bd.n[a] > 1 ? bd.n[a] / 2 : 1;
  }
  // offset range: cells are at least rmin / width wide (in fractional coordinates), so +-1 is
  // enough unless a periodic box is thinner than min_distance; then more images are scanned
  long long num_offsets = 1;
  for (int a = 0; a < 3; ++a) {
    bd.r[a] = 1;
    if (bd.pbc[a] && widths[a] > 0.0 && std::isfinite(widths[a])) {
      const double need = std::ceil(min_distance_ * bd.n[a] / widths[a]);
      if (need > 1.0)
        bd.r[a] = need > 1.0e6 ? 1000000 : static_cast<int>(need);
    }
    num_offsets *= 2LL * bd.r[a] + 1;
  }
  if (num_offsets > 100000000LL) {
    PRINT_INPUT_ERROR(
      "check_stability: the periodic box is too thin compared with min_distance (more than "
      "1e8 image cells per atom).");
  }
  const int num_cells = bd.n[0] * bd.n[1] * bd.n[2];
  if (cell_head_.size() != static_cast<size_t>(num_cells))
    cell_head_.resize(num_cells);
  cell_head_.fill(-1);
  const double* x = atom.position_per_atom.data();
  const double* y = x + N;
  const double* z = y + N;
  const int grid = (N - 1) / 128 + 1;
  gpu_build_cells<<<grid, 128>>>(N, bd, x, y, z, cell_head_.data(), cell_next_.data());
  GPU_CHECK_KERNEL
  gpu_find_pairs<<<grid, 128>>>(
    N,
    bd,
    min_distance_,
    x,
    y,
    z,
    cell_head_.data(),
    cell_next_.data(),
    pair_d2_.data(),
    pair_partner_.data());
  GPU_CHECK_KERNEL
  pair_d2_.copy_to_host(cpu_pair_d2_.data());
  pair_partner_.copy_to_host(cpu_pair_partner_.data());
  double best = INFINITY;
  index_1 = -1;
  index_2 = -1;
  for (int i = 0; i < N; ++i) {
    const double d2 = cpu_pair_d2_[i];
    if (std::isnan(d2)) { // non-finite position
      index_1 = i;
      index_2 = -1;
      return NAN;
    }
    if (d2 < best) {
      best = d2;
      index_1 = i;
      index_2 = cpu_pair_partner_[i];
    }
  }
  if (index_2 >= 0 && index_2 < index_1) {
    const int t = index_1;
    index_1 = index_2;
    index_2 = t;
  }
  return std::isfinite(best) ? sqrt(best) : INFINITY;
}

void Check_Stability::end_of_step(
  const int,
  int step,
  const int,
  const int,
  const double,
  const double,
  Integrate&,
  Box& box,
  std::vector<Group>&,
  GPU_Vector<double>&,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % interval_ != 0)
    return;
  const int N = num_atoms_;
  const int run_step = step + 1;
  const float* charge = nullptr;
  if (use_charge_)
    charge = force.get_potential(0).get_charge_reference().data();

  gpu_reduce_state<<<num_blocks_, BLOCK_SIZE>>>(
    N,
    atom.mass.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + N,
    atom.velocity_per_atom.data() + 2 * N,
    atom.potential_per_atom.data(),
    atom.force_per_atom.data(),
    atom.force_per_atom.data() + N,
    atom.force_per_atom.data() + 2 * N,
    charge,
    partial_.data());
  GPU_CHECK_KERNEL
  partial_.copy_to_host(cpu_partial_.data());
  double ke = 0.0, u = 0.0, fmax = -1.0, fcode = -1.0, qmax = -1.0, qidx = -1.0;
  for (int b = 0; b < num_blocks_; ++b) {
    const double* p = cpu_partial_.data() + b * NUM_PARTIAL;
    ke += p[0];
    u += p[1];
    if (p[2] > fmax || (p[2] == fmax && p[3] >= 0.0 && p[3] < fcode)) {
      fmax = p[2];
      fcode = p[3];
    }
    if (p[4] > qmax || (p[4] == qmax && p[5] >= 0.0 && p[5] < qidx)) {
      qmax = p[4];
      qidx = p[5];
    }
  }
  const double temperature = 2.0 * ke / (3.0 * N * K_B);
  if (!has_reference_) {
    potential_reference_ = u;
    has_reference_ = true;
  }
  const double du = fabs(u - potential_reference_) / N;

  int d_1 = -1, d_2 = -1;
  double shortest = INFINITY;
  if (use_distance_)
    shortest = find_shortest_distance(atom, box, d_1, d_2);

  store_frame(run_step, atom, box, charge);

  // The first violated criterion (fixed order) is named in the final error; all are printed.
  // Comparisons are written so that NaN counts as a violation.
  bool violated = false;
  std::string first;
  char line[512];
  auto report = [&](const char* text) {
    if (!violated) {
      first = text;
      violated = true;
    }
    printf("check_stability: %s\n", text);
  };
  if (use_temperature_ && !(temperature <= max_temperature_)) {
    snprintf(
      line,
      sizeof(line),
      "step %d: temperature %.10g K exceeds max_temperature %.10g K",
      run_step,
      temperature,
      max_temperature_);
    report(line);
  }
  if (use_potential_change_ && !(du <= max_potential_change_)) {
    snprintf(
      line,
      sizeof(line),
      "step %d: |U - U0|/N = %.10g eV/atom exceeds max_potential_change %.10g eV/atom "
      "(U = %.10g eV, U0 = %.10g eV)",
      run_step,
      du,
      max_potential_change_,
      u,
      potential_reference_);
    report(line);
  }
  if (use_distance_ && !(shortest >= min_distance_)) {
    if (d_2 >= 0) {
      snprintf(
        line,
        sizeof(line),
        "step %d: shortest distance %.10g A between atoms %d and %d is below min_distance %.10g A "
        "(atom indices start at 0)",
        run_step,
        shortest,
        d_1,
        d_2,
        min_distance_);
    } else {
      snprintf(
        line,
        sizeof(line),
        "step %d: non-finite position of atom %d (min_distance %g A, atom indices start at 0)",
        run_step,
        d_1,
        min_distance_);
    }
    report(line);
  }
  if (use_force_ && !(fmax <= max_force_)) {
    const long long code = static_cast<long long>(fcode);
    snprintf(
      line,
      sizeof(line),
      "step %d: force component %.10g eV/A of atom %lld (component %lld) exceeds max_force %.10g "
      "eV/A "
      "(atom indices start at 0)",
      run_step,
      fmax,
      code / 3,
      code % 3,
      max_force_);
    report(line);
  }
  if (use_charge_ && !(qmax <= max_charge_)) {
    snprintf(
      line,
      sizeof(line),
      "step %d: |q| = %.10g e of atom %lld exceeds max_charge %.10g e (atom indices start at 0)",
      run_step,
      qmax,
      static_cast<long long>(qidx),
      max_charge_);
    report(line);
  }
  if (violated) {
    write_frames(atom);
    printf(
      "check_stability: wrote %d frame(s) to stability_frames.xyz; no restart file is written.\n",
      num_stored_);
    fflush(stdout);
    std::string message = "check_stability violated: " + first;
    PRINT_INPUT_ERROR(message.c_str());
  }
}
