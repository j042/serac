// Copyright (c) 2019, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file nonlinear_solid_solver.hpp
 *
 * @brief The solver object for finite deformation hyperelasticity
 */

#ifndef NONLIN_SOLID
#define NONLIN_SOLID

#include "infrastructure/input.hpp"
#include "mfem.hpp"
#include "physics/base_physics.hpp"
#include "physics/operators/nonlinear_solid_operators.hpp"

namespace serac {

/**
 * @brief The nonlinear solid solver class
 *
 * The nonlinear hyperelastic quasi-static and dynamic
 * hyperelastic solver object. It is derived from MFEM
 * example 10p.
 */
class NonlinearSolid : public BasePhysics {
public:
  /**
   * @brief A timestep method and config for the M solver
   */
  struct DynamicSolverParameters {
    TimestepMethod         timestepper;
    LinearSolverParameters M_params;
  };
  /**
   * @brief A configuration variant for the various solves
   * Either quasistatic, or time-dependent with timestep and M params
   */
  struct SolverParameters {
    LinearSolverParameters                 H_lin_params;
    NonlinearSolverParameters              H_nonlin_params;
    std::optional<DynamicSolverParameters> dyn_params = std::nullopt;
  };

  /**
   * @brief Construct a new Nonlinear Solid Solver object
   *
   * @param[in] order The order of the displacement field
   * @param[in] mesh The MFEM parallel mesh to solve on
   * @param[in] solver The system solver parameters
   */
  NonlinearSolid(int order, std::shared_ptr<mfem::ParMesh> mesh, const SolverParameters& params);

  /**
   * @brief Set displacement boundary conditions
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::VectorCoefficient> disp_bdr_coef);

  /**
   * @brief Set the displacement essential boundary conditions on a single component
   *
   * @param[in] disp_bdr The set of boundary attributes to set the displacement on
   * @param[in] disp_bdr_coef The vector coefficient containing the set displacement values
   * @param[in] component The component to set the displacment on
   */
  void setDisplacementBCs(const std::set<int>& disp_bdr, std::shared_ptr<mfem::Coefficient> disp_bdr_coef,
                          int component);

  /**
   * @brief Set the traction boundary conditions
   *
   * @param[in] trac_bdr The set of boundary attributes to apply a traction to
   * @param[in] trac_bdr_coef The vector valued traction coefficient
   * @param[in] component The component to apply the traction on
   */
  void setTractionBCs(const std::set<int>& trac_bdr, std::shared_ptr<mfem::VectorCoefficient> trac_bdr_coef,
                      int component = -1);

  /**
   * @brief Set the viscosity coefficient
   *
   * @param[in] visc_coef The abstract viscosity coefficient
   */
  void setViscosity(std::unique_ptr<mfem::Coefficient>&& visc_coef);

  /**
   * @brief Set the hyperelastic material parameters
   *
   * @param[in] mu Set the mu Lame parameter for the hyperelastic solid
   * @param[in] K Set the K Lame parameter for the hyperelastic solid
   */
  void setHyperelasticMaterialParameters(double mu, double K);

  /**
   * @brief Set the initial displacement value
   *
   * @param[in] disp_state The initial displacement state
   */
  void setDisplacement(mfem::VectorCoefficient& disp_state);

  /**
   * @brief Set the velocity state
   *
   * @param[in] velo_state The velocity state
   */
  void setVelocity(mfem::VectorCoefficient& velo_state);

  /**
   * @brief Get the displacement state
   *
   * @return The displacement state field
   */
  std::shared_ptr<FiniteElementState> displacement() { return displacement_; };

  /**
   * @brief Get the velocity state
   *
   * @return The velocity state field
   */
  std::shared_ptr<FiniteElementState> velocity() { return velocity_; };

  /**
   * @brief Complete the setup of all of the internal MFEM objects and prepare for timestepping
   */
  void completeSetup() override;

  /**
   * @brief Advance the timestep
   *
   * @param[inout] dt The timestep to attempt. This will return the actual timestep for adaptive timestepping schemes
   */
  void advanceTimestep(double& dt) override;

  /**
   * @brief Destroy the Nonlinear Solid Solver object
   */
  virtual ~NonlinearSolid();

  /**
   * @brief Input file parameters specific to this class
   *
   * @param[in] schema_creator Inlet's SchemaCreator that input files will be added too
   **/
  static void defineInputFileSchema(std::shared_ptr<axom::inlet::SchemaCreator> schema_creator);

protected:
  /**
   * @brief Velocity field
   */
  std::shared_ptr<FiniteElementState> velocity_;

  /**
   * @brief Displacement field
   */
  std::shared_ptr<FiniteElementState> displacement_;

  /**
   * @brief The quasi-static operator for use with the MFEM newton solvers
   */
  std::unique_ptr<mfem::Operator> nonlinear_oper_;

  /**
   * @brief Configuration for dynamic equation solver
   */
  std::optional<LinearSolverParameters> timedep_oper_params_;

  /**
   * @brief The time dependent operator for use with the MFEM ODE solvers
   */
  std::unique_ptr<mfem::TimeDependentOperator> timedep_oper_;

  /**
   * @brief The viscosity coefficient
   */
  std::unique_ptr<mfem::Coefficient> viscosity_;

  /**
   * @brief The hyperelastic material model
   */
  std::unique_ptr<mfem::HyperelasticModel> model_;

  /**
   * @brief Pointer to the reference mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> reference_nodes_;

  /**
   * @brief Pointer to the deformed mesh data
   */
  std::unique_ptr<mfem::ParGridFunction> deformed_nodes_;

  /**
   * @brief Solve the Quasi-static operator
   */
  void quasiStaticSolve();

  /**
   * @brief Nonlinear system solver instance
   */
  EquationSolver nonlin_solver_;
};

}  // namespace serac

#endif