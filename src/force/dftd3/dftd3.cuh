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

#pragma once
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;

class DFTD3
{
public:
  struct DFTD3_Para {
    float s6 = 0.0;
    float s8 = 0.0;
    float a1 = 0.0;
    float a2 = 0.0;
    int damping = 0; // 0: BJ damping, 1: zero damping, D3(0) (DR-003)
    float sr6 = 0.0; // zero damping only: scaling of R0 in f6 (alpha6 = 14)
    float sr8 = 1.0; // zero damping only: scaling of R0 in f8 (alpha8 = 16)
    int atomic_number[94];
  };

  // Damping selector of the dftd3 keyword (ADR-0003): `dftd3 <functional> <rc> <rc_cn> [zero|bj]`.
  // Checks the number of tokens, exits with a message on error, and returns "bj" or "zero".
  static std::string parse_damping(const std::vector<std::string>& tokens);

  DFTD3_Para dftd3_para;

  bool enabled = false;
  bool elements_checked = false; // zero damping: atom types of the structure checked (Z <= 86)
  float rc_radial = 15.0;
  float rc_angular = 10.0;
  GPU_Vector<float> cn;
  GPU_Vector<float> cn_weight; // N * max_cn Gaussian weights of the C6 interpolation
  GPU_Vector<float> c6_ref;
  GPU_Vector<float> dc6_sum;
  GPU_Vector<float> dc8_sum;
  GPU_Vector<int> cn_list;       // accepted angular partners per (atom, thread) slot, large box
  GPU_Vector<int> cn_list_count; // number of accepted partners per slot

  GPU_Vector<int> NN_radial;
  GPU_Vector<int> NL_radial;
  GPU_Vector<int> NN_angular;
  GPU_Vector<int> NL_angular;
  GPU_Vector<float> r12;

  GPU_Vector<int> cell_count_radial;
  GPU_Vector<int> cell_count_sum_radial;
  GPU_Vector<int> cell_contents_radial;
  GPU_Vector<int> cell_count_angular;
  GPU_Vector<int> cell_count_sum_angular;
  GPU_Vector<int> cell_contents_angular;

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  ExpandedBox ebox;

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void initialize(
    std::string& ex_functional,
    const float rc_radial,
    const float rc_angular,
    const std::string& potential_file_name,
    const std::string& damping = "bj");

  void check_elements(const GPU_Vector<int>& type);
};
