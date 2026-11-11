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

#include "nep_charge_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

bool NEP_Charge_Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "lambda_q") == 0) {
    parse_lambda_q(param, num_param);
  } else if (strcmp(param[0], "lambda_z") == 0) {
    parse_lambda_z(param, num_param);
  } else if (strcmp(param[0], "charge_transfer") == 0) {
    parse_charge_transfer(param, num_param);
  } else {
    return NEP_Energy_Parameters::parse_keyword(param, num_param);
  }
  return true;
}

void NEP_Charge_Parameters::parse_lambda_q(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_q should have 1 parameter.\n");
  }

  double lambda_q_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_q_tmp)) {
    PRINT_INPUT_ERROR("Charge loss weight should be a number.\n");
  }
  lambda_q = lambda_q_tmp;

  if (lambda_q < 0.0f) {
    PRINT_INPUT_ERROR("Charge loss weight should >= 0.");
  }
}

void NEP_Charge_Parameters::parse_lambda_z(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_z should have 1 parameter.\n");
  }

  double lambda_z_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_z_tmp)) {
    PRINT_INPUT_ERROR("BEC loss weight should be a number.\n");
  }
  lambda_z = lambda_z_tmp;

  if (lambda_z < 0.0f) {
    PRINT_INPUT_ERROR("BEC loss weight should >= 0.");
  }
}

void NEP_Charge_Parameters::parse_charge_transfer(const char** param, int num_param)
{
  if (rc_phi > 0.0f) {
    PRINT_INPUT_ERROR("Specify charge_transfer only once.");
  }
  if (num_param != 2) {
    PRINT_INPUT_ERROR("charge_transfer should have 1 parameter (rc_phi, in Angstrom).\n");
  }
  double rc_phi_tmp = 0.0;
  if (!is_valid_real(param[1], &rc_phi_tmp)) {
    PRINT_INPUT_ERROR("charge_transfer rc_phi should be a number.\n");
  }
  if (
    !std::isfinite(rc_phi_tmp) || !(rc_phi_tmp > 0.0) || !(float(rc_phi_tmp) > 0.0f) ||
    rc_phi_tmp > 100.0) {
    PRINT_INPUT_ERROR("charge_transfer rc_phi should be > 0 and <= 100 Angstrom.");
  }
  rc_phi = rc_phi_tmp;
}

void NEP_Charge_Parameters::report_model_inputs() const
{
  NEP_Energy_Parameters::report_model_inputs();
  if (rc_phi > 0.0f) {
    printf("    (input)   charge_transfer rc_phi = %g A.\n", rc_phi);
  }
  printf("        lambda_q = %g.\n", lambda_q);
  printf("        lambda_z = %g.\n", lambda_z);
}
