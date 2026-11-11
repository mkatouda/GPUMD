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

// charge_transfer kernels (DR-005 / ADR-0006, DES-0005 sections 2-3).
//   q_a = sum_b phi_ab (h_a - h_b),  V_a = sum_b phi_ab (r_a - r_b)
//   Z_k[beta,alpha] = dP_beta/dr_k,alpha = Term A + Term B (index 3 * beta + alpha)
//   Term A = sum_a V_a,beta dh_a/dr_k,alpha ; Term B = sum_b M_kb (h_k - h_b),
//   M_kb = phi'(r) xhat (x) x + phi(r) 1, x = r_k - r_b  (contains the diagonal q_k 1)
// h is the charge output of the per-atom network (find_descriptor); q overwrites nep_data.charge.
// Neighbours are the radial neighbour list (rc_phi <= rc_radial); phi vanishes beyond rc_phi.
// Large-box kernels take r12 from the positions with MIC, small-box kernels from the stored r12.
#pragma once

#include "model/box.cuh"
#include "nep_charge.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"

// phi_{t1 t2}(d) and its derivative: the radial basis f_k(d) fc(d) with cutoff rc_phi, using the
// same find_fc_and_fcp / find_fn_and_fnp as the radial descriptor. Symmetric pair index
// p = t1 * nt - t1 * (t1 - 1) / 2 + (t2 - t1) for t1 <= t2.
static __device__ __forceinline__ void ct_find_phi(
  const NEP_Charge::ParaMB& paramb,
  const NEP_Charge::ANN& annmb,
  const int t1,
  const int t2,
  const float d12,
  float& phi,
  float& dphi)
{
  phi = 0.0f;
  dphi = 0.0f;
  if (d12 >= paramb.rc_phi) {
    return;
  }
  const float rc = paramb.rc_phi;
  const float rcinv = 1.0f / rc;
  float fc12, fcp12;
  find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
  float fn12[MAX_NUM_N];
  float fnp12[MAX_NUM_N];
  find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
  const int ta = t1 < t2 ? t1 : t2;
  const int tb = t1 < t2 ? t2 : t1;
  const int pair = ta * paramb.num_types - ta * (ta - 1) / 2 + (tb - ta);
  for (int k = 0; k <= paramb.basis_size_radial; ++k) {
    const float c = annmb.phi_coeff[pair * paramb.num_phi_per_pair + k];
    phi += fn12[k] * c;
    dphi += fnp12[k] * c;
  }
}

// Term B contribution of one pair to atom k (all nine components, index 3 * beta + alpha).
// x = r_k - r_b = -r12 leaves x (x) x unchanged.
static __device__ __forceinline__ void ct_add_term_B(
  const float phi,
  const float dphi,
  const float r12[3],
  const float d12,
  const float dh,
  double Z[9])
{
  const float d12inv = 1.0f / d12;
  for (int b = 0; b < 3; ++b) {
    for (int a = 0; a < 3; ++a) {
      float m = dphi * r12[b] * r12[a] * d12inv;
      if (a == b) {
        m += phi;
      }
      Z[3 * b + a] += double(m) * double(dh);
    }
  }
}

static __global__ void ct_find_charge_large_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_h,
  float* g_charge,
  float* g_V)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    double q = 0.0;
    double V[3] = {0.0, 0.0, 0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - g_x[n1];
      float y12 = g_y[n2] - g_y[n1];
      float z12 = g_z[n2] - g_z[n1];
      apply_mic(box, x12, y12, z12);
      const float r12[3] = {x12, y12, z12};
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      q += double(phi) * (double(g_h[n1]) - double(g_h[n2]));
      for (int d = 0; d < 3; ++d) {
        V[d] -= double(phi) * double(r12[d]);
      }
    }
    g_charge[n1] = float(q);
    for (int d = 0; d < 3; ++d) {
      g_V[n1 + N * d] = float(V[d]);
    }
  }
}

static __global__ void ct_find_charge_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_h,
  float* g_charge,
  float* g_V)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    double q = 0.0;
    double V[3] = {0.0, 0.0, 0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      q += double(phi) * (double(g_h[n1]) - double(g_h[n2]));
      for (int d = 0; d < 3; ++d) {
        V[d] -= double(phi) * double(r12[d]);
      }
    }
    g_charge[n1] = float(q);
    for (int d = 0; d < 3; ++d) {
      g_V[n1 + N * d] = float(V[d]);
    }
  }
}

// Term B: writes (not accumulates) all nine components of g_bec
static __global__ void ct_find_bec_B_large_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_h,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    double Z[9] = {0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - g_x[n1];
      float y12 = g_y[n2] - g_y[n1];
      float z12 = g_z[n2] - g_z[n1];
      apply_mic(box, x12, y12, z12);
      const float r12[3] = {x12, y12, z12};
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      ct_add_term_B(phi, dphi, r12, d12, g_h[n1] - g_h[n2], Z);
    }
    for (int c = 0; c < 9; ++c) {
      g_bec[n1 + N * c] = float(Z[c]);
    }
  }
}

static __global__ void ct_find_bec_B_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_h,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    double Z[9] = {0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      ct_add_term_B(phi, dphi, r12, d12, g_h[n1] - g_h[n2], Z);
    }
    for (int c = 0; c < 9; ++c) {
      g_bec[n1 + N * c] = float(Z[c]);
    }
  }
}

// Term A: Z_n2 += V_n1 (x) dh_n1/dr_n2 and Z_n1 -= the same. One pass gives all nine components
// (equivalent to three passes with the weights V_x, V_y, V_z).
static __device__ __forceinline__ void ct_scatter_term_A(
  const int N, const int n1, const int n2, const float V[3], const float f12[3], float* g_bec)
{
  for (int b = 0; b < 3; ++b) {
    for (int a = 0; a < 3; ++a) {
      const float z = V[b] * f12[a];
      atomicAdd(&g_bec[n2 + N * (3 * b + a)], z);
      atomicAdd(&g_bec[n1 + N * (3 * b + a)], -z);
    }
  }
}

// radial part of dh_a/dr_b for the pair (n1, n2), r12 = r_n2 - r_n1
static __device__ __forceinline__ void ct_radial_dh(
  const NEP_Charge::ParaMB& paramb,
  const NEP_Charge::ANN& annmb,
  const int N,
  const int n1,
  const int t1,
  const int t2,
  const float r12[3],
  const float d12,
  const float* g_charge_derivative,
  float f12[3])
{
  const float d12inv = 1.0f / d12;
  float fc12, fcp12;
  const float rc = paramb.rc_radial;
  const float rcinv = 1.0f / rc;
  find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
  float fn12[MAX_NUM_N];
  float fnp12[MAX_NUM_N];
  find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    float gnp12 = 0.0f;
    for (int k = 0; k <= paramb.basis_size_radial; ++k) {
      int c_index = get_c_index(
        t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
      gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
    }
    const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
    for (int d = 0; d < 3; ++d) {
      f12[d] += tmp12 * r12[d];
    }
  }
}

static __global__ void ct_find_bec_A_radial_large_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  const float* g_V,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - g_x[n1];
      float y12 = g_y[n2] - g_y[n1];
      float z12 = g_z[n2] - g_z[n1];
      apply_mic(box, x12, y12, z12);
      const float r12[3] = {x12, y12, z12};
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float f12[3] = {0.0f};
      ct_radial_dh(paramb, annmb, N, n1, t1, g_type[n2], r12, d12, g_charge_derivative, f12);
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}

static __global__ void ct_find_bec_A_radial_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_charge_derivative,
  const float* g_V,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      ct_radial_dh(paramb, annmb, N, n1, t1, g_type[n2], r12, d12, g_charge_derivative, f12);
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}

// angular part of dh_a/dr_b for the pair (n1, n2); Fp and sum_fxyz are those of atom n1
static __device__ __forceinline__ void ct_angular_dh(
  const NEP_Charge::ParaMB& paramb,
  const NEP_Charge::ANN& annmb,
  const int t1,
  const int t2,
  const float r12[3],
  const float d12,
  float* Fp,
  float* sum_fxyz,
  float f12[3])
{
  float fc12, fcp12;
  const float rc = paramb.rc_angular;
  const float rcinv = 1.0f / rc;
  find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
  float fn12[MAX_NUM_N];
  float fnp12[MAX_NUM_N];
  find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float gn12 = 0.0f;
    float gnp12 = 0.0f;
    for (int k = 0; k <= paramb.basis_size_angular; ++k) {
      int c_index = get_c_index(
        t1 * paramb.num_types + t2,
        n,
        k,
        paramb.n_max_angular,
        paramb.basis_size_angular,
        paramb.num_c_radial);
      gn12 += fn12[k] * annmb.c_type_pair[c_index];
      gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
    }
    accumulate_f12(
      paramb.L_max,
      paramb.has_q_222,
      paramb.has_q_1111,
      paramb.has_q_112,
      paramb.has_q_123,
      paramb.has_q_233,
      paramb.has_q_134,
      paramb.num_L,
      n,
      paramb.n_max_angular + 1,
      d12,
      r12,
      gn12,
      gnp12,
      Fp,
      sum_fxyz,
      f12);
  }
}

static __device__ __forceinline__ void ct_load_angular_atom(
  const NEP_Charge::ParaMB& paramb,
  const int N,
  const int n1,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* Fp,
  float* sum_fxyz)
{
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
  }
  for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
      sum_fxyz[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
    }
  }
}

static __global__ void ct_find_bec_A_angular_large_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  const float* g_V,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    ct_load_angular_atom(paramb, N, n1, g_charge_derivative, g_sum_fxyz, Fp, sum_fxyz);
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      const int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - g_x[n1];
      float y12 = g_y[n2] - g_y[n1];
      float z12 = g_z[n2] - g_z[n1];
      apply_mic(box, x12, y12, z12);
      const float r12[3] = {x12, y12, z12};
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float f12[3] = {0.0f};
      ct_angular_dh(paramb, annmb, t1, g_type[n2], r12, d12, Fp, sum_fxyz, f12);
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}

static __global__ void ct_find_bec_A_angular_small_box(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  const float* g_V,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    ct_load_angular_atom(paramb, N, n1, g_charge_derivative, g_sum_fxyz, Fp, sum_fxyz);
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL_angular[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      ct_angular_dh(paramb, annmb, t1, g_type[n2], r12, d12, Fp, sum_fxyz, f12);
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}
