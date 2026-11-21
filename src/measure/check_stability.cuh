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

#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Check_Stability : public Action
{
public:
  Check_Stability(const std::vector<std::string>& tokens, const std::string& potential_filename);

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void end_of_step(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

private:
  struct Frame_Info {
    int step = 0;
    int pbc[3] = {1, 1, 1};
    double h[9] = {0.0};
  };

  void store_frame(const int step, Atom& atom, const Box& box, const float* charge);
  void write_frames(const Atom& atom);
  double find_shortest_distance(Atom& atom, const Box& box, int& index_1, int& index_2);

  int interval_ = 0;
  int keep_frames_ = 3;
  bool use_temperature_ = false;
  bool use_potential_change_ = false;
  bool use_distance_ = false;
  bool use_force_ = false;
  bool use_charge_ = false;
  double max_temperature_ = 0.0;
  double max_potential_change_ = 0.0;
  double min_distance_ = 0.0;
  double max_force_ = 0.0;
  double max_charge_ = 0.0;

  // state of the current run
  bool has_reference_ = false;
  double potential_reference_ = 0.0;
  int num_atoms_ = 0;
  int num_blocks_ = 0;
  int num_stored_ = 0;
  int next_slot_ = 0;
  std::vector<Frame_Info> frame_info_;
  GPU_Vector<double> frame_state_; // positions, velocities, forces of the stored frames
  GPU_Vector<float> frame_charge_;
  GPU_Vector<double> partial_; // per-block partial results of the reductions
  std::vector<double> cpu_partial_;
  GPU_Vector<int> cell_head_;
  GPU_Vector<int> cell_next_;
  GPU_Vector<double> pair_d2_;
  GPU_Vector<int> pair_partner_;
  std::vector<double> cpu_pair_d2_;
  std::vector<int> cpu_pair_partner_;
};
