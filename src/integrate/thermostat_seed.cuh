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
Optional trailing "seed <n>" token of the ensemble line (DR-009), shared by
nvt_bdp and npt_scr. The token pair is removed from the token list so that the
usual parameter-count checks of the ensemble apply to the remaining tokens.
------------------------------------------------------------------------------*/

#pragma once
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <string>
#include <vector>

// Returns the seed (> 0) and strips the two trailing tokens, or returns 0 and leaves
// the tokens unchanged when there is no trailing "seed" keyword.
static inline int extract_thermostat_seed(std::vector<std::string>& tokens)
{
  if (tokens.size() < 3) {
    return 0;
  }
  if (tokens.back() == "seed") {
    PRINT_INPUT_ERROR("The seed keyword of the ensemble needs a positive integer value.");
  }
  if (tokens[tokens.size() - 2] != "seed") {
    return 0;
  }
  int seed = 0;
  if (!is_valid_int(tokens.back(), &seed)) {
    PRINT_INPUT_ERROR("The seed of the ensemble should be an integer.");
  }
  if (seed <= 0) {
    PRINT_INPUT_ERROR("The seed of the ensemble should be a positive integer.");
  }
  tokens.pop_back();
  tokens.pop_back();
  if (tokens.size() >= 3 && tokens[tokens.size() - 2] == "seed") {
    PRINT_INPUT_ERROR("The seed keyword of the ensemble can be given only once.");
  }
  return seed;
}
