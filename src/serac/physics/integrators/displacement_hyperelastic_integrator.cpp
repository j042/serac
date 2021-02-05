// Copyright (c) 2019-2021, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/integrators/displacement_hyperelastic_integrator.hpp"

#include "serac/infrastructure/profiling.hpp"
#include "serac/numerics/expr_template_ops.hpp"
#include "serac/numerics/array_4D.hpp"

namespace serac::mfem_ext {

void DisplacementHyperelasticIntegrator::CalcDeformationGradient(
    const mfem::FiniteElement& element, const mfem::IntegrationPoint& int_point,
    mfem::ElementTransformation& parent_to_reference_transformation)
{
  // Calculate the reference to stress-free transformation
  CalcInverse(parent_to_reference_transformation.Jacobian(), Jrp_);

  // Calculate the derivatives of the shape functions in the reference space
  element.CalcDShape(int_point, DSh_);

  // Calculate the derivatives of the shape functions in the stress free configuration
  Mult(DSh_, Jrp_, DS_);

  // Get the dimesion of the problem (2 or 3)
  int dim = Jrp_.Width();

  // Calculate the displacement gradient using the current DOFs
  MultAtB(input_state_matrix_, DS_, H_);

  // Add the identity matrix to calculate the deformation gradient F_
  F_ = H_;
  for (int i = 0; i < dim; ++i) {
    F_(i, i) += 1.0;
  }

  // Calculate the inverse of the deformation gradient
  mfem::CalcInverse(F_, Finv_);

  // Calculate the B matrix (dN/Dx where x is the current configuration)

  if (geom_nonlin_) {
    // If we're including geometric nonlinearities, we integrate on the current deformed configuration
    mfem::Mult(DS_, Finv_, B_);
    detF_ = F_.Det();
  } else {
    // If not, we integrate on the undeformed stress-free configuration
    B_    = DS_;
    detF_ = 1.0;
  }
}

double DisplacementHyperelasticIntegrator::GetElementEnergy(
    const mfem::FiniteElement& element, mfem::ElementTransformation& parent_to_reference_transformation,
    const mfem::Vector& state_vector)
{
  int dof = element.GetDof();
  int dim = element.GetDim();

  // Reshape the state vector
  input_state_matrix_.UseExternalData(state_vector.GetData(), dof, dim);

  // Determine the integration rule from the order of the FE space
  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3));
  }

  double energy = 0.0;

  // Set the transformation for the underlying material. This is required for coefficient evaluation.
  material_.SetTransformation(parent_to_reference_transformation);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& int_point = ir->IntPoint(i);
    parent_to_reference_transformation.SetIntPoint(&int_point);

    // Calculate the deformation gradent and accumulate the strain energy at the current integration point
    CalcDeformationGradient(element, int_point, parent_to_reference_transformation);
    energy += detF_ * int_point.weight * parent_to_reference_transformation.Weight() * material_.EvalStrainEnergy(F_);
  }

  return energy;
}

void DisplacementHyperelasticIntegrator::AssembleElementVector(
    const mfem::FiniteElement& element, mfem::ElementTransformation& parent_to_reference_transformation,
    const mfem::Vector& state_vector, mfem::Vector& residual_vector)
{
  int dof = element.GetDof();
  int dim = element.GetDim();

  // Ensure that the working vectors are sized appropriately
  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  B_.SetSize(dof, dim);
  Jrp_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  H_.SetSize(dim);
  sigma_.SetSize(dim);

  // Reshape the input state and residual vectors as matrices
  input_state_matrix_.UseExternalData(state_vector.GetData(), dof, dim);
  residual_vector.SetSize(dof * dim);
  output_residual_matrix_.UseExternalData(residual_vector.GetData(), dof, dim);

  // Select an integration rule based on the order of the underlying FE space
  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3));
  }

  // Set the transformation for the underlying material. This is required for coefficient evaluation.
  material_.SetTransformation(parent_to_reference_transformation);

  output_residual_matrix_ = 0.0;

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Set the current integration point
    const mfem::IntegrationPoint& int_point = ir->IntPoint(i);
    parent_to_reference_transformation.SetIntPoint(&int_point);

    // Calculate the deformation gradient at the current integration point
    CalcDeformationGradient(element, int_point, parent_to_reference_transformation);

    // Evaluate the Cauchy stress using the calculated deformation gradient
    material_.EvalStress(F_, sigma_);

    // Accumulate the residual using the Cauchy stress and the B matrix
    sigma_ *= detF_ * int_point.weight * parent_to_reference_transformation.Weight();
    mfem::AddMult(B_, sigma_, output_residual_matrix_);
  }
}

void DisplacementHyperelasticIntegrator::AssembleElementGrad(
    const mfem::FiniteElement& element, mfem::ElementTransformation& parent_to_reference_transformation,
    const mfem::Vector& state_vector, mfem::DenseMatrix& stiffness_matrix)
{
  SERAC_MARK_FUNCTION;

  int dof = element.GetDof();
  int dim = element.GetDim();

  // Ensure that the working vectors are sized appropriately
  DSh_.SetSize(dof, dim);
  DS_.SetSize(dof, dim);
  B_.SetSize(dof, dim);
  Jrp_.SetSize(dim);
  F_.SetSize(dim);
  Finv_.SetSize(dim);
  H_.SetSize(dim);
  sigma_.SetSize(dim);
  stiffness_matrix.SetSize(dof * dim);
  C_.SetSize(dim, dim, dim, dim);

  // Reshape the input state as a matrix
  input_state_matrix_.UseExternalData(state_vector.GetData(), dof, dim);

  // Select an integration rule based on the order of the underlying FE space
  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(element.GetGeomType(), 2 * element.GetOrder() + 3));  // <---
  }

  stiffness_matrix = 0.0;

  // Set the transformation for the underlying material. This is required for coefficient evaluation.
  material_.SetTransformation(parent_to_reference_transformation);
  SERAC_MARK_LOOP_START(ip_loop_id, "IntegrationPt Loop");

  for (int ip_num = 0; ip_num < ir->GetNPoints(); ip_num++) {
    // Set the integration point and calculate the deformation gradient
    SERAC_MARK_LOOP_ITER(ip_loop_id, i);
    const mfem::IntegrationPoint& int_point = ir->IntPoint(ip_num);
    parent_to_reference_transformation.SetIntPoint(&int_point);
    CalcDeformationGradient(element, int_point, parent_to_reference_transformation);

    // Assemble the spatial tangent moduli at the current integration point
    material_.AssembleTangentModuli(F_, C_);

    // Accumulate the material stiffness using the spatial tangent moduli and the B matrix
    for (int a = 0; a < dof; ++a) {
      for (int i = 0; i < dim; ++i) {
        for (int b = 0; b < dof; ++b) {
          for (int k = 0; k < dim; ++k) {
            for (int j = 0; j < dim; ++j) {
              for (int l = 0; l < dim; ++l) {
                stiffness_matrix(i * dof + a, k * dof + b) += C_(i, j, k, l) * B_(a, j) * B_(b, l) * int_point.weight *
                                                              parent_to_reference_transformation.Weight();
              }
            }
          }
        }
      }
    }

    // Accumulate the geometric stiffness if desired
    if (geom_nonlin_) {
      material_.EvalStress(F_, sigma_);
      for (int a = 0; a < dof; ++a) {
        for (int i = 0; i < dim; ++i) {
          for (int b = 0; b < dof; ++b) {
            for (int k = 0; k < dim; ++k) {
              for (int j = 0; j < dim; ++j) {
                stiffness_matrix(i * dof + a, k * dof + b) -= detF_ * sigma_(i, j) * B_(a, k) * B_(b, j) *
                                                              int_point.weight *
                                                              parent_to_reference_transformation.Weight();
              }
            }
          }
        }
      }
    }
  }
  SERAC_MARK_LOOP_END(ip_loop_id);
}

}  // namespace serac::mfem_ext
