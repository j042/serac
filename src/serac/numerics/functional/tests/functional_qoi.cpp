// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>
#include <iostream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils_base.hpp"
#include "serac/numerics/functional/functional.hpp"
#include "serac/numerics/functional/shape_aware_functional.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/infrastructure/profiling.hpp"

#include "serac/numerics/functional/tests/check_gradient.hpp"

using namespace serac;
using namespace serac::profiling;

int num_procs, myid;
int nsamples = 1;  // because mfem doesn't take in unsigned int

double t = 0.0;

std::unique_ptr<mfem::ParMesh> mesh2D;
std::unique_ptr<mfem::ParMesh> mesh3D;

double measure_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm mass_lf(&fespace);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_lf.Assemble();

  mfem::ParGridFunction one_gf(&fespace);
  one_gf.ProjectCoefficient(one);

  return mass_lf(one_gf);
}

double x_moment_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm mass_lf(&fespace);
  mass_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  mass_lf.Assemble();

  mfem::FunctionCoefficient x_coordinate([](mfem::Vector x) { return x[0]; });
  mfem::ParGridFunction     x_gf(&fespace);
  x_gf.ProjectCoefficient(x_coordinate);

  return mass_lf(x_gf);
}

double sum_of_measures_mfem(mfem::ParMesh& mesh)
{
  mfem::ConstantCoefficient one(1.0);

  auto                        fec = mfem::H1_FECollection(1, mesh.Dimension());
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParLinearForm lf(&fespace);
  lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
  lf.AddBoundaryIntegrator(new mfem::BoundaryLFIntegrator(one));
  lf.Assemble();

  mfem::ParGridFunction one_gf(&fespace);
  one_gf.ProjectCoefficient(one);

  return lf(one_gf);
}

enum class WhichTest
{
  Measure,
  Moment,
  SumOfMeasures,
  Nonlinear
};

// note: the exact answers are generated by a Mathematica script that
// integrates the qoi for these domains to machine precision
//
// see scripts/wolfram/qoi_examples.nb for more info
template <int p, int dim>
void qoi_test(mfem::ParMesh& mesh, H1<p> trial, Dimension<dim>, WhichTest which)
{
  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParGridFunction     U_gf(&fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);

  mfem::HypreParVector* tmp = fespace.NewTrueDofVector();
  mfem::HypreParVector  U   = *tmp;
  U_gf.GetTrueDofs(U);

  mfem::ParGridFunction     V_gf(&fespace);
  mfem::FunctionCoefficient x_coord([](mfem::Vector x) { return x[0]; });
  V_gf.ProjectCoefficient(x_coord);

  mfem::HypreParVector* tmp2 = fespace.NewTrueDofVector();
  mfem::HypreParVector  V    = *tmp2;
  V_gf.GetTrueDofs(V);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = decltype(trial);

  switch (which) {
    case WhichTest::Measure: {
      Functional<double(trial_space)> measure({&fespace});
      measure.AddDomainIntegral(
          Dimension<dim>{}, DependsOn<>{}, [&](double /*t*/, auto /*x*/) { return 1.0; }, mesh);

      constexpr double expected[] = {1.0, 16.0};

      double relative_error = (measure(t, U) - expected[dim - 2]) / expected[dim - 2];
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

      relative_error = (measure(t, U) - measure_mfem(mesh)) / measure(t, U);
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

    } break;

    case WhichTest::Moment: {
      Functional<double(trial_space)> x_moment({&fespace});
      x_moment.AddDomainIntegral(
          Dimension<dim>{}, DependsOn<>{}, [&](double /*t*/, auto position) { return get<VALUE>(position)[0]; }, mesh);

      constexpr double expected[] = {0.5, 40.0};

      double relative_error = (x_moment(t, U) - expected[dim - 2]) / expected[dim - 2];
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

      relative_error = (x_moment(t, U) - x_moment_mfem(mesh)) / x_moment(t, U);
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

      Functional<double(trial_space)> x_moment_2({&fespace});
      x_moment_2.AddDomainIntegral(
          Dimension<dim>{}, DependsOn<0>{}, [&](double /*t*/, auto, auto u) { return get<0>(u); }, mesh);

      relative_error = (x_moment_2(t, V) - expected[dim - 2]) / expected[dim - 2];
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

      relative_error = (x_moment_2(t, V) - x_moment_mfem(mesh)) / x_moment_2(t, V);
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

    } break;

    case WhichTest::SumOfMeasures: {
      Functional<double(trial_space)> sum_of_measures({&fespace});
      sum_of_measures.AddDomainIntegral(
          Dimension<dim>{}, DependsOn<>{}, [&](double /*t*/, auto /*x*/) { return 1.0; }, mesh);
      sum_of_measures.AddBoundaryIntegral(
          Dimension<dim - 1>{}, DependsOn<>{}, [&](double /*t*/, auto /*x*/) { return 1.0; }, mesh);

      constexpr double expected[] = {5.0, 64.0};

      double relative_error = (sum_of_measures(t, U) - expected[dim - 2]) / expected[dim - 2];
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

      relative_error = (sum_of_measures(t, U) - sum_of_measures_mfem(mesh)) / sum_of_measures(t, U);
      EXPECT_NEAR(0.0, relative_error, 1.0e-10);

    } break;

    case WhichTest::Nonlinear: {
      Functional<double(trial_space)> f({&fespace});
      f.AddDomainIntegral(
          Dimension<dim>{}, DependsOn<0>{},
          [&](double /*t*/, auto position, auto temperature) {
            auto X           = get<VALUE>(position);
            auto [u, grad_u] = temperature;
            return X[0] * X[0] + sin(X[1]) + X[0] * u * u * u;
          },
          mesh);
      f.AddBoundaryIntegral(
          Dimension<dim - 1>{}, DependsOn<0>{},
          [&](double /*t*/, auto position, auto temperature) {
            auto [X, dX_dxi] = position;
            auto [u, unused] = temperature;
            return X[0] - X[1] + cos(u * X[1]);
          },
          mesh);

      constexpr double expected[] = {4.6640262484879, 192400.1149761554};

      double relative_error = (f(t, U) - expected[dim - 2]) / expected[dim - 2];

      // the tolerance on this one isn't very tight since
      // we're using a pretty the coarse integration rule
      // that doesn't capture the features in the trigonometric integrands
      EXPECT_NEAR(0.0, relative_error, 3.0e-2);

      check_gradient(f, t, U);

    } break;
  }

  delete tmp;
}

template <int p1, int p2, int dim>
void qoi_test(mfem::ParMesh& mesh, H1<p1> trial1, H1<p2> trial2, Dimension<dim>)
{
  auto                        fec1 = mfem::H1_FECollection(p1, dim);
  mfem::ParFiniteElementSpace fespace1(&mesh, &fec1);

  auto                        fec2 = mfem::H1_FECollection(p2, dim);
  mfem::ParFiniteElementSpace fespace2(&mesh, &fec2);

  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  mfem::FunctionCoefficient y([](mfem::Vector x) { return x[1]; });

  mfem::ParGridFunction U1_gf(&fespace1);
  U1_gf.ProjectCoefficient(x_squared);

  mfem::ParGridFunction U2_gf(&fespace2);
  U2_gf.ProjectCoefficient(y);

  std::unique_ptr<mfem::HypreParVector> tmp;

  tmp.reset(fespace1.NewTrueDofVector());
  mfem::HypreParVector U1 = *tmp;
  U1_gf.GetTrueDofs(U1);

  tmp.reset(fespace2.NewTrueDofVector());
  mfem::HypreParVector U2 = *tmp;
  U2_gf.GetTrueDofs(U2);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space1 = decltype(trial1);
  using trial_space2 = decltype(trial2);

  Functional<double(trial_space1, trial_space2)> f({&fespace1, &fespace2});
  f.AddDomainIntegral(
      Dimension<dim>{}, DependsOn<0, 1>{},
      [&](double /*t*/, auto position, auto temperature, auto dtemperature_dt) {
        auto [X, dX_dxi]     = position;
        auto [u, grad_u]     = temperature;
        auto [du_dt, unused] = dtemperature_dt;
        return X[0] * X[0] + sin(du_dt) + X[0] * u * u * u;
      },
      mesh);
  f.AddBoundaryIntegral(
      Dimension<dim - 1>{}, DependsOn<0, 1>{},
      [&](double /*t*/, auto position, auto temperature, auto dtemperature_dt) {
        auto [X, dX_dxi]     = position;
        auto [u, grad_u]     = temperature;
        auto [du_dt, unused] = dtemperature_dt;
        return X[0] - X[1] + cos(u * du_dt);
      },
      mesh);

  // note: these answers are generated by a Mathematica script that
  // integrates the qoi for these domains to machine precision
  //
  // see scripts/wolfram/qoi_examples.nb for more info
  constexpr double expected[]     = {4.6640262484879, 192400.1149761554};
  double           relative_error = (f(t, U1, U2) - expected[dim - 2]) / expected[dim - 2];

  // the tolerance on this one isn't very tight since
  // we're using a pretty the coarse integration rule
  // that doesn't capture the features in the trigonometric integrands
  EXPECT_NEAR(0.0, relative_error, 3.0e-2);

  check_gradient(f, t, U1, U2);
}

TEST(QoI, DependsOnVectorValuedInput)
{
  constexpr int p   = 2;
  constexpr int dim = 3;

  mfem::ParMesh& mesh = *mesh3D;

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec, dim);

  mfem::ParGridFunction           U_gf(&fespace);
  mfem::VectorFunctionCoefficient x_squared(dim, [](const mfem::Vector x, mfem::Vector& y) {
    y    = 0.0;
    y[0] = x[0] * x[0];
  });
  U_gf.ProjectCoefficient(x_squared);

  mfem::HypreParVector* tmp = fespace.NewTrueDofVector();
  mfem::HypreParVector  U   = *tmp;
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = H1<p, dim>;

  Functional<double(trial_space)> f({&fespace});
  f.AddVolumeIntegral(
      DependsOn<0>{}, [&](double /*t*/, auto /*x*/, auto u) { return norm(serac::get<0>(u)); }, mesh);

  double exact_answer   = 141.3333333333333;
  double relative_error = (f(t, U) - exact_answer) / exact_answer;
  EXPECT_NEAR(0.0, relative_error, 1.0e-10);

  delete tmp;
}

TEST(QoI, AddAreaIntegral)
{
  constexpr int p   = 1;
  constexpr int dim = 2;

  mfem::ParMesh& mesh = *mesh2D;

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParGridFunction     U_gf(&fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);

  mfem::HypreParVector* tmp = fespace.NewTrueDofVector();
  mfem::HypreParVector  U   = *tmp;
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = H1<p>;

  Functional<double(trial_space)> measure({&fespace});
  measure.AddAreaIntegral(
      DependsOn<>{}, [&](double /*t*/, auto /*x*/) { return 1.0; }, mesh);
  double relative_error = (measure(t, U) - measure_mfem(mesh)) / measure(t, U);
  EXPECT_NEAR(0.0, relative_error, 1.0e-10);

  delete tmp;
}

TEST(QoI, AddVolumeIntegral)
{
  constexpr int p   = 1;
  constexpr int dim = 3;

  mfem::ParMesh& mesh = *mesh3D;

  auto                        fec = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace(&mesh, &fec);

  mfem::ParGridFunction     U_gf(&fespace);
  mfem::FunctionCoefficient x_squared([](mfem::Vector x) { return x[0] * x[0]; });
  U_gf.ProjectCoefficient(x_squared);

  mfem::HypreParVector* tmp = fespace.NewTrueDofVector();
  mfem::HypreParVector  U   = *tmp;
  U_gf.GetTrueDofs(U);

  // Define the types for the test and trial spaces using the function arguments
  using trial_space = H1<p>;

  Functional<double(trial_space)> measure({&fespace});
  measure.AddVolumeIntegral(
      DependsOn<>{}, [&](double /*t*/, auto /*x*/) { return 1.0; }, mesh);
  double relative_error = (measure(t, U) - measure_mfem(mesh)) / measure(t, U);
  EXPECT_NEAR(0.0, relative_error, 1.0e-10);

  delete tmp;
}

TEST(QoI, UsingL2)
{
  constexpr int p   = 1;
  constexpr int dim = 3;

  mfem::ParMesh& mesh = *mesh3D;

  auto                        fec0 = mfem::H1_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace_0(&mesh, &fec0);

  auto                        fec1 = mfem::L2_FECollection(p, dim);
  mfem::ParFiniteElementSpace fespace_1(&mesh, &fec1);

  std::unique_ptr<mfem::HypreParVector> U0(fespace_0.NewTrueDofVector());
  U0->Randomize(0);

  std::unique_ptr<mfem::HypreParVector> U1(fespace_1.NewTrueDofVector());
  U1->Randomize(1);

  // Define the types for the test spaces
  using trial_space_0 = H1<p>;
  using trial_space_1 = L2<0>;

  // this tests a fix for the QoI constructor segfaulting when using L2 spaces
  Functional<double(trial_space_0, trial_space_1)> f({&fespace_0, &fespace_1});

  f.AddVolumeIntegral(
      DependsOn<1>{}, [&](auto...) { return 1.0; }, mesh);
  f.AddSurfaceIntegral(
      DependsOn<0>{}, [&](auto...) { return 1.0; }, mesh);

  check_gradient(f, t, *U0, *U1);
}

TEST(QoI, ShapeAndParameter)
{
  // _average_start
  // Define the compile-time finite element spaces for the shape displacement and parameter fields
  static constexpr int dim{3};

  // The shape displacement must be vector-valued H1
  using shape_space = H1<2, dim>;

  // Shape-aware functional only supports H1 and L2 trial functions
  using parameter_space = H1<1>;

  // Define the QOI type. Note that the shape-aware functional has an extra template argument
  // for the shape displacement finite element space
  using qoi_type = serac::ShapeAwareFunctional<shape_space, double(parameter_space)>;

  // Define the mesh and runtime finite element spaces for the calculation
  mfem::ParMesh& mesh = *mesh3D;

  auto [shape_fe_space, shape_fe_coll]         = generateParFiniteElementSpace<shape_space>(&mesh);
  auto [parameter_fe_space, parameter_fe_coll] = generateParFiniteElementSpace<parameter_space>(&mesh);

  std::array<const mfem::ParFiniteElementSpace*, 1> trial_fes = {parameter_fe_space.get()};
  const mfem::ParFiniteElementSpace*                shape_fes = shape_fe_space.get();

  auto   everything = [](std::vector<tensor<double, dim>> /*X*/, int /* attr */) { return true; };
  Domain whole_mesh = Domain::ofElements(mesh, everything);

  // Define the shape-aware QOI objects
  qoi_type serac_qoi(shape_fes, trial_fes);

  // Note that the integral does not have a shape parameter field. The transformations are handled under the hood
  // so the user only sees the modified x = X + p input arguments
  serac_qoi.AddDomainIntegral(
      serac::Dimension<dim>{}, serac::DependsOn<0>{},
      [](double /*t*/, auto /*x*/, auto param) { return serac::get<0>(param); }, whole_mesh);

  qoi_type serac_volume(shape_fes, trial_fes);

  serac_volume.AddDomainIntegral(
      serac::Dimension<dim>{}, serac::DependsOn<>{}, [](double /*t*/, auto /*x*/) { return 1.0; }, whole_mesh);

  std::unique_ptr<mfem::HypreParVector> shape_displacement(shape_fe_space->NewTrueDofVector());
  *shape_displacement = 1.0;

  std::unique_ptr<mfem::HypreParVector> parameter(parameter_fe_space->NewTrueDofVector());
  *parameter = 0.1;

  // Note that the first argument after time is always the shape displacement field
  double val    = serac_qoi(t, *shape_displacement, *parameter);
  double volume = serac_volume(t, *shape_displacement, *parameter);

  double average = val / volume;
  // _average_end

  constexpr double expected_vol = 16.0;  // volume of 2 2x2x2 cubes == 16, so expected is 0.1 * 16
  EXPECT_NEAR(volume, expected_vol, 3.0e-14);

  constexpr double expected_val = 1.6;
  EXPECT_NEAR(val, expected_val, 1.0e-14);

  constexpr double expected_avg = 0.1;
  EXPECT_NEAR(average, expected_avg, 1.0e-14);
}

TEST(QoI, ShapeAndParameterBoundary)
{
  // _boundary_start
  // Define the compile-time finite element spaces for the shape displacement and parameter fields
  static constexpr int dim{3};

  // The shape displacement must be vector-valued H1
  using shape_space = H1<2, dim>;

  // Shape-aware functional only supports H1 and L2 trial functions
  using parameter_space = H1<1, dim>;

  // Define the QOI type. Note that the shape-aware functional has an extra template argument
  // for the shape displacement finite element space
  using qoi_type = serac::ShapeAwareFunctional<shape_space, double(parameter_space)>;

  // Define the mesh and runtime finite element spaces for the calculation
  mfem::ParMesh& mesh = *mesh3D;

  auto [shape_fe_space, shape_fe_coll]         = generateParFiniteElementSpace<shape_space>(&mesh);
  auto [parameter_fe_space, parameter_fe_coll] = generateParFiniteElementSpace<parameter_space>(&mesh);

  std::array<const mfem::ParFiniteElementSpace*, 1> trial_fes = {parameter_fe_space.get()};
  const mfem::ParFiniteElementSpace*                shape_fes = shape_fe_space.get();

  // Define the shape-aware QOI objects
  qoi_type serac_qoi(shape_fes, trial_fes);

  auto   everything     = [](std::vector<tensor<double, dim>> /*X*/, int /* attr */) { return true; };
  Domain whole_boundary = Domain::ofBoundaryElements(mesh, everything);

  // Note that the integral does not have a shape parameter field. The transformations are handled under the hood
  // so the user only sees the modified x = X + p input arguments
  serac_qoi.AddBoundaryIntegral(
      serac::Dimension<dim - 1>{}, serac::DependsOn<0>{},
      [](double /*t*/, auto x, auto param) {
        using std::abs;
        auto n = normalize(cross(get<DERIVATIVE>(x)));
        return abs(dot(serac::get<VALUE>(param), n));
      },
      whole_boundary);

  qoi_type serac_area(shape_fes, trial_fes);

  serac_area.AddBoundaryIntegral(
      serac::Dimension<dim - 1>{}, serac::DependsOn<>{}, [](double /*t*/, auto /*x*/) { return 1.0; }, whole_boundary);

  std::unique_ptr<mfem::HypreParVector> shape_displacement(shape_fe_space->NewTrueDofVector());
  *shape_displacement = 1.0;

  std::unique_ptr<mfem::HypreParVector> parameter(parameter_fe_space->NewTrueDofVector());
  *parameter = 5.0;

  // Note that the first argument after time is always the shape displacement field
  double val    = serac_qoi(t, *shape_displacement, *parameter);
  double volume = serac_area(t, *shape_displacement, *parameter);

  double average = val / volume;
  // _boundary_end

  constexpr double expected_vol = 48.0;  // volume of 2 2x2x2 cubes == 16, so expected is 0.1 * 16
  EXPECT_NEAR(volume, expected_vol, 3.0e-14);

  constexpr double expected_val = 240.0;
  EXPECT_NEAR(val, expected_val, 1.5e-13);

  constexpr double expected_avg = 5.0;
  EXPECT_NEAR(average, expected_avg, 1.0e-14);
}

// clang-format off
TEST(Measure, 2DLinear   ) { qoi_test(*mesh2D, H1<1>{}, Dimension<2>{}, WhichTest::Measure); }
TEST(Measure, 2DQuadratic) { qoi_test(*mesh2D, H1<2>{}, Dimension<2>{}, WhichTest::Measure); }
TEST(Measure, 3DLinear   ) { qoi_test(*mesh3D, H1<1>{}, Dimension<3>{}, WhichTest::Measure); }
TEST(Measure, 3DQuadratic) { qoi_test(*mesh3D, H1<2>{}, Dimension<3>{}, WhichTest::Measure); }

TEST(Moment, 2DLinear   ) { qoi_test(*mesh2D, H1<1>{}, Dimension<2>{}, WhichTest::Moment); }
TEST(Moment, 2DQuadratic) { qoi_test(*mesh2D, H1<2>{}, Dimension<2>{}, WhichTest::Moment); }
TEST(Moment, 3DLinear   ) { qoi_test(*mesh3D, H1<1>{}, Dimension<3>{}, WhichTest::Moment); }
TEST(Moment, 3DQuadratic) { qoi_test(*mesh3D, H1<2>{}, Dimension<3>{}, WhichTest::Moment); }

TEST(SumOfMeasures, 2DLinear   ) { qoi_test(*mesh2D, H1<1>{}, Dimension<2>{}, WhichTest::SumOfMeasures); }
TEST(SumOfMeasures, 2DQuadratic) { qoi_test(*mesh2D, H1<2>{}, Dimension<2>{}, WhichTest::SumOfMeasures); }
TEST(SumOfMeasures, 3DLinear   ) { qoi_test(*mesh3D, H1<1>{}, Dimension<3>{}, WhichTest::SumOfMeasures); }
TEST(SumOfMeasures, 3DQuadratic) { qoi_test(*mesh3D, H1<2>{}, Dimension<3>{}, WhichTest::SumOfMeasures); }

TEST(Nonlinear, 2DLinear   ) { qoi_test(*mesh2D, H1<1>{}, Dimension<2>{}, WhichTest::Nonlinear); }
TEST(Nonlinear, 2DQuadratic) { qoi_test(*mesh2D, H1<2>{}, Dimension<2>{}, WhichTest::Nonlinear); }
TEST(Nonlinear, 3DLinear   ) { qoi_test(*mesh3D, H1<1>{}, Dimension<3>{}, WhichTest::Nonlinear); }
TEST(Nonlinear, 3DQuadratic) { qoi_test(*mesh3D, H1<2>{}, Dimension<3>{}, WhichTest::Nonlinear); }

TEST(Variadic, 2DLinear   ) { qoi_test(*mesh2D, H1<1>{}, H1<1>{}, Dimension<2>{}); }
TEST(Variadic, 2DQuadratic) { qoi_test(*mesh2D, H1<2>{}, H1<2>{}, Dimension<2>{}); }
TEST(Variadic, 3DLinear   ) { qoi_test(*mesh3D, H1<1>{}, H1<1>{}, Dimension<3>{}); }
TEST(Variadic, 3DQuadratic) { qoi_test(*mesh3D, H1<2>{}, H1<2>{}, Dimension<3>{}); }
// clang-format on

// TODO Functional currently doesn't support HCurl. When it does, this test should work without other changes as
// shape-aware functional already contains the appropriate Hcurl transformations

// TEST(QoI, MixedShapeAware)
// {
//   // _boundary_start
//   static constexpr int dim{3};

//   using shape_space = H1<2, dim>;

//   using parameter_1_space = H1<1>;
//   using parameter_2_space = Hcurl<1>;

//   // Define the QOI type. Note that the shape aware functional has an extra template argument
//   // for the shape displacement finite element space
//   using qoi_type = serac::ShapeAwareFunctional<shape_space, double(parameter_1_space, parameter_2_space)>;

//   // Define the mesh and runtime finite element spaces for the calculation
//   mfem::ParMesh& mesh = *mesh3D;

//   auto [shape_fe_space, shape_fe_coll]             = generateParFiniteElementSpace<shape_space>(&mesh);
//   auto [parameter_1_fe_space, parameter_1_fe_coll] = generateParFiniteElementSpace<parameter_1_space>(&mesh);
//   auto [parameter_2_fe_space, parameter_2_fe_coll] = generateParFiniteElementSpace<parameter_2_space>(&mesh);

//   std::array<const mfem::ParFiniteElementSpace*, 2> trial_fes = {parameter_1_fe_space.get(),
//                                                                  parameter_2_fe_space.get()};
//   const mfem::ParFiniteElementSpace*                shape_fes = shape_fe_space.get();

//   // Define the shape-aware QOI objects
//   qoi_type serac_qoi(shape_fes, trial_fes);

//   serac_qoi.AddDomainIntegral(
//       serac::Dimension<dim>{}, serac::DependsOn<0>{},
//       [](double /*t*/, auto /*x*/, auto scalar_param) { return serac::get<0>(scalar_param); }, mesh);

//   serac_qoi.AddDomainIntegral(
//       serac::Dimension<dim>{}, serac::DependsOn<1>{},
//       [](double /*t*/, auto /*x*/, auto vector_hcurl_param) { return norm(serac::get<0>(vector_hcurl_param)); },
//       mesh);

//   std::unique_ptr<mfem::HypreParVector> shape_displacement(shape_fe_space->NewTrueDofVector());
//   *shape_displacement = 1.0;

//   std::unique_ptr<mfem::HypreParVector> parameter_1(parameter_1_fe_space->NewTrueDofVector());
//   *parameter_1 = 0.1;

//   std::unique_ptr<mfem::HypreParVector> parameter_2(parameter_2_fe_space->NewTrueDofVector());
//   *parameter_2 = 0.5;

//   double val = serac_qoi(t, *shape_displacement, *parameter_1, *parameter_2);
// }

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  axom::slic::SimpleLogger logger;

  int serial_refinement   = 1;
  int parallel_refinement = 0;

  std::string meshfile2D = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";
  mesh2D = mesh::refineAndDistribute(buildMeshFromFile(meshfile2D), serial_refinement, parallel_refinement);

  std::string meshfile3D = SERAC_REPO_DIR "/data/meshes/patch3D_tets_and_hexes.mesh";
  mesh3D = mesh::refineAndDistribute(buildMeshFromFile(meshfile3D), serial_refinement, parallel_refinement);

  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
