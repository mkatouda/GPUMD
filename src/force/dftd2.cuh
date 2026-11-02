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
class RunInput;

// DFT-D2 pairwise dispersion correction (S. Grimme, J. Comput. Chem. 27, 1787 (2006)),
// added externally to a NEP model (DR-002). Keyword: dftd2 <s6> [<cutoff in Angstrom>].
class DFTD2
{
public:
  // PROVISIONAL (2026-11-03): default cutoff used when the cutoff is omitted.
  // To be decided in DES-0002 Phase 2 (cutoff study); do not rely on this value.
  static constexpr double DEFAULT_CUTOFF = 25.0;

  bool enabled = false;
  double s6 = 0.0;
  double rc = DEFAULT_CUTOFF;

  // Parse the tokens of one "dftd2" line (exits with a specific message on invalid input).
  static void parse_tokens(const std::vector<std::string>& tokens, double& s6, double& rc);

  // Find and parse the (single) dftd2 line of run.in; returns false if there is none.
  static bool find_in_run_input(const RunInput& run_input, double& s6, double& rc);

  // atomic_numbers[t] is the atomic number of potential type t.
  void initialize(const double s6, const double rc, const std::vector<int>& atomic_numbers);

  // Adds energy, force and virial of the D2 correction to the per-atom arrays.
  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

private:
  std::vector<int> atomic_numbers_;
  bool elements_checked_ = false;
  int num_types_ = 0;

  // float4 per type pair: {s6*C6_ij, R0_ij, phi(rc), phi'(rc)}
  GPU_Vector<float4> pair_param_;

  GPU_Vector<double> extent_;    // min/max fractional coordinates (non-periodic directions)
  GPU_Vector<double4> wrapped_;  // wrapped position (x, y, z) and type (w), original order
  GPU_Vector<double4> sorted_;   // the same in cell order
  GPU_Vector<int> sorted_index_; // original atom index in cell order
  GPU_Vector<int> cell_of_atom_;
  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_start_;
  GPU_Vector<int> cell_fill_;

  void check_elements(const GPU_Vector<int>& type);
};
