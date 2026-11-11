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

// charge_transfer kernels of the NEP trainer (DR-005 stage 3a, ADR-0006, DES-0005).
// Forward-only port of src/force/nep_charge/nep_charge_transfer.cuh (same arithmetic; the types
// NEP_Charge::ParaMB / ANN differ between gpumd and nep, so the header cannot be shared directly):
//   q_a = sum_b phi_ab (h_a - h_b),  V_a = sum_b phi_ab (r_a - r_b)
//   Z_k[beta,alpha] = dP_beta/dr_k,alpha = Term A + Term B (index 3 * beta + alpha)
//   Term B = sum_b M_kb (h_k - h_b), M_kb = phi' xhat (x) x + phi 1 ; Term A = sum_a V_a,beta
//   dh_a/dr_k,alpha Dt_k = sum_b phi_kb (D_k - D_b), pair force F_k += G v/d, virial -0.5 G v (x) v
//   / d, G = phi'(d) (h_k - h_b) (D_k - D_b), v = r_b - r_k.
// Neighbour lists are the trainer lists (compact index NN_sum[n1] + i1, stored r12 = r_n2 - r_n1).
// The trainer virial has six components (xx yy zz xy yz zx).
#pragma once

#include "nep_charge.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"

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

static __global__ void ct_find_charge(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_h,
  float* g_charge,
  float* g_charge_shifted,
  float* g_V)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    double q = 0.0;
    double V[3] = {0.0, 0.0, 0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = g_NN_sum[n1] + i1;
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
    // q replaces the network output as "the charge" (loss, prediction output, long range);
    // no total-charge shift: sum q = 0 by construction
    g_charge[n1] = float(q);
    g_charge_shifted[n1] = float(q);
    for (int d = 0; d < 3; ++d) {
      g_V[n1 + N * d] = float(V[d]);
    }
  }
}

// Term B: writes (not accumulates) all nine components of g_bec (includes q 1)
static __global__ void ct_find_bec_B(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_h,
  float* g_bec)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    double Z[9] = {0.0};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = g_NN_sum[n1] + i1;
      const int n2 = g_NL[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      const float dh = g_h[n1] - g_h[n2];
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
    for (int c = 0; c < 9; ++c) {
      g_bec[n1 + N * c] = float(Z[c]);
    }
  }
}

// Term A: Z_n2 += V_n1 (x) dh_n1/dr_n2 and Z_n1 -= the same (all nine components in one pass)
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

static __global__ void ct_find_bec_A_radial(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int* g_NN_sum,
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
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = g_NN_sum[n1] + i1;
      const int n2 = g_NL[index];
      const int t2 = g_type[n2];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      const float d12inv = 1.0f / d12;
      float fc12, fcp12;
      const float rc = paramb.rc_radial;
      const float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          const int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}

static __global__ void ct_find_bec_A_angular(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  const float* g_V,
  float* g_bec)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    const int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int abc = 0; abc < num_abc; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] = g_sum_fxyz[(n * num_abc + abc) * N + n1];
      }
    }
    const int t1 = g_type[n1];
    const float V[3] = {g_V[n1], g_V[n1 + N], g_V[n1 + N * 2]};
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = g_NN_sum[n1] + i1;
      const int n2 = g_NL[index];
      const int t2 = g_type[n2];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      const float rc = paramb.rc_angular;
      const float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          const int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
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
      ct_scatter_term_A(N, n1, n2, V, f12, g_bec);
    }
  }
}

// Dt (weight of the descriptor force kernels) and the explicit pair force/virial.
// D = dE/dq must be final (after the long-range solver, real-space and self-energy parts).
static __global__ void ct_find_Dt_and_pair_force(
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_h,
  const float* g_D,
  float* g_Dt,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    double Dt = 0.0;
    double f[3] = {0.0, 0.0, 0.0};
    double w[6] = {0.0}; // xx yy zz xy yz zx
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      const int index = g_NN_sum[n1] + i1;
      const int n2 = g_NL[index];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float phi, dphi;
      ct_find_phi(paramb, annmb, t1, g_type[n2], d12, phi, dphi);
      if (phi == 0.0f && dphi == 0.0f) {
        continue;
      }
      const double dh = double(g_h[n1]) - double(g_h[n2]);
      const double dD = double(g_D[n1]) - double(g_D[n2]);
      Dt += double(phi) * dD;
      const double g = double(dphi) * dh * dD / double(d12);
      const double v[3] = {double(r12[0]), double(r12[1]), double(r12[2])};
      for (int a = 0; a < 3; ++a) {
        f[a] += g * v[a];
      }
      w[0] -= 0.5 * g * v[0] * v[0];
      w[1] -= 0.5 * g * v[1] * v[1];
      w[2] -= 0.5 * g * v[2] * v[2];
      w[3] -= 0.5 * g * v[0] * v[1];
      w[4] -= 0.5 * g * v[1] * v[2];
      w[5] -= 0.5 * g * v[2] * v[0];
    }
    g_Dt[n1] = float(Dt);
    g_fx[n1] += f[0];
    g_fy[n1] += f[1];
    g_fz[n1] += f[2];
    for (int c = 0; c < 6; ++c) {
      g_virial[n1 + N * c] += w[c];
    }
  }
}
