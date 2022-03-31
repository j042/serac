// Copyright (c) 2019-2022, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file equation_solver.hpp
 *
 * @brief This file contains the declaration of an equation solver wrapper
 */

#pragma once

#include "mfem.hpp"
#include "serac/numerics/functional/tuple.hpp"

namespace serac::solid_util {

/**
 * @brief Calculate the deformation gradient from the displacement gradient (F = H + I)
 *
 * @param[in] du_dX the displacement gradient (du_dX)
 * @param[out] F the deformation gradient (dx_dX)
 */
void calcDeformationGradient(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& F);

/**
 * @brief Calculate the linearized strain tensor (epsilon = 1/2 * (du_dX + du_dX^T))
 *
 * @param[in] du_dX the displacement gradient (du_dX)
 * @param[out] epsilon the linearized strain tensor epsilon = 1/2 * (du_dX + du_dX^T)
 */
void calcLinearizedStrain(const mfem::DenseMatrix& du_dX, mfem::DenseMatrix& epsilon);

/**
 * @brief Calculate the Cauchy stress from the PK1 stress
 *
 * @param[in] F the deformation gradient dx_dX
 * @param[in] P the first Piola-Kirchoff stress tensor
 * @param[out] sigma the Cauchy stress tensor
 */
void calcCauchyStressFromPK1Stress(const mfem::DenseMatrix& F, const mfem::DenseMatrix& P, mfem::DenseMatrix& sigma);

/**
 * @brief Adjust the displacement and displacement gradient with a shape displacement field
 *
 * @note If shape_index = -1, the original displacement is returned.
 *
 * @tparam shape_index The index of the shape parameter in the @ params parameter pack
 * @tparam T Displacement type
 * @tparam Ts Parameter types
 * @param displacement A tuple of the displacement and displacement gradient
 * @param params A parameter pack containing the shape displacement parameter
 * @return The modified displacement containing the shape displacement term if appropriate.
 */
template <int shape_index, typename T, typename... Ts>
auto adjustDisplacementWithShape(T displacement, Ts... params)
{
  if constexpr (shape_index != -1) {
    auto shape_param = serac::get<shape_index>(serac::tuple{params...});
    return serac::tuple{displacement[0] + shape_param[0], displacement[1] + shape_param[1]};
  }
  return displacement;
}

}  // namespace serac::solid_util
